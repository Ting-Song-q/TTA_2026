#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从观察位直接抓取红色方块。

前提：机械臂已处于观察位姿（开爪、红块已在正前方）。
本脚本不移动到 initial/observe，也不下降：
  空握则开爪，基于当前位姿重新视觉规划（不回观察位；最多 --grasp-retries 次）
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_TTA = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from grasp_3d import (  # noqa: E402
    DEFAULT_OG,
    DEFAULT_V3D,
    full_pose_from_ik,
    go_pose_strict,
    hold_forever,
    ik_route,
    merge_configs,
)
from og import JOINTS, TicksArm, clamp_pose, pose_i, print_pose  # noqa: E402
from pixel_to_base import (  # noqa: E402
    anchor_grasp_forward,
    grasp_z,
    load_handeye,
    load_intrinsics,
    pixel_to_base,
)
from so101_fk import JOINT_ORDER, SO101FK  # noqa: E402
from so101_ik import SO101IK  # noqa: E402
from so101_red_block_camera_test import (  # noqa: E402
    RedBlockDetection,
    RedBlockDetector,
    capture_frame,
)
from start import DEFAULT_BAUD, move_to  # noqa: E402


def build_forward_route_xyz(
    cur_xyz: np.ndarray,
    target_xyz: np.ndarray,
    n_points: int,
) -> List[np.ndarray]:
    """前进到目标 XY；Z 用目标值（默认同观察位，可由 --z-offset 微调）。"""
    goal = np.array(
        [float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])],
        dtype=float,
    )
    n = max(2, int(n_points))
    route: List[np.ndarray] = []
    for i in range(1, n + 1):
        a = i / float(n)
        route.append(cur_xyz + a * (goal - cur_xyz))
    return route


def offset_grasp_left(
    ee_xyz: np.ndarray,
    grasp_xyz: np.ndarray,
    left_m: float,
) -> np.ndarray:
    """相对前进方向（夹爪→抓取点水平）向左偏移；Z 朝上时左 = 前进方向水平逆时针 90°。"""
    p = np.asarray(grasp_xyz, dtype=float).reshape(3).copy()
    if abs(float(left_m)) < 1e-9:
        return p
    fwd = p[:2] - np.asarray(ee_xyz, dtype=float).reshape(3)[:2]
    n = float(np.linalg.norm(fwd))
    if n < 1e-6:
        left_xy = np.array([1.0, 0.0], dtype=float)  # 前进 -Y 时左为 +X
    else:
        fwd = fwd / n
        left_xy = np.array([-fwd[1], fwd[0]], dtype=float)
    p[0] += float(left_m) * float(left_xy[0])
    p[1] += float(left_m) * float(left_xy[1])
    return p


def grasp_target_uv(
    frame_shape: tuple,
    *,
    target_u: Optional[float],
    target_v: Optional[float],
) -> tuple[float, float]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    tu = float(target_u) if target_u is not None else 0.5 * w
    tv = float(target_v) if target_v is not None else 0.55 * h
    return tu, tv


def is_block_graspable(
    det: Optional[RedBlockDetection],
    frame: np.ndarray,
    *,
    target_u: float,
    target_v: float,
    tol_u: float,
    tol_v: float,
    min_area: float,
    max_area: float,
) -> tuple[bool, str]:
    """闭爪前判定：红块可见且中心落在夹爪对准窗口内。"""
    if det is None:
        return False, "未检测到红块"
    if det.area < min_area:
        return False, f"面积过小 area={det.area:.0f}<{min_area:.0f}"
    if det.area > max_area:
        return False, f"面积过大 area={det.area:.0f}>{max_area:.0f}"
    du = float(det.center_u) - float(target_u)
    dv = float(det.center_v) - float(target_v)
    if abs(du) > tol_u or abs(dv) > tol_v:
        return False, f"中心偏离窗口 du={du:+.1f} dv={dv:+.1f} (tol_u={tol_u:.0f},tol_v={tol_v:.0f})"
    return True, f"可抓取 du={du:+.1f} dv={dv:+.1f} area={det.area:.0f}"


def pixel_error_to_base_delta(
    err_u: float,
    err_v: float,
    T_base_cam: np.ndarray,
    *,
    m_per_px: float,
    max_step_m: float,
) -> np.ndarray:
    """眼在手上：图像误差 → 基座水平微调（相机 x/y 投到水平面）。"""
    d_cam = np.array(
        [float(err_u) * m_per_px, float(err_v) * m_per_px, 0.0],
        dtype=float,
    )
    d_base = T_base_cam[:3, :3] @ d_cam
    d_xy = np.array([d_base[0], d_base[1], 0.0], dtype=float)
    n = float(np.linalg.norm(d_xy[:2]))
    if n > max_step_m > 0:
        d_xy[:2] *= max_step_m / n
    return d_xy


def refine_pose_until_graspable(
    *,
    arm: TicksArm,
    fk: SO101FK,
    ik: SO101IK,
    detector: RedBlockDetector,
    camera_cfg: dict,
    T_ee_cam: np.ndarray,
    limits: dict,
    g_open: float,
    observe_arm: dict,
    save_dir: Path,
    no_save: bool,
    ik_tol: float,
    max_horiz_m: float,
    z_hold: float,
    target_u: Optional[float],
    target_v: Optional[float],
    tol_u: float,
    tol_v: float,
    min_area: float,
    max_area: float,
    m_per_px: float,
    max_step_m: float,
    max_iters: int,
    settle_s: float,
) -> bool:
    """闭爪前循环：识别 → 判定可抓 → 否则按像素误差微调位姿。"""
    for it in range(max(1, int(max_iters))):
        time.sleep(float(settle_s))
        frame = capture_frame(camera_cfg)
        det = detector.detect(frame)
        tu, tv = grasp_target_uv(frame.shape, target_u=target_u, target_v=target_v)
        if not no_save:
            vis = detector.draw(frame, det)
            cv2.rectangle(
                vis,
                (int(tu - tol_u), int(tv - tol_v)),
                (int(tu + tol_u), int(tv + tol_v)),
                (0, 255, 0),
                2,
            )
            cv2.drawMarker(vis, (int(tu), int(tv)), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
            save_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = save_dir / f"preclose{it+1}_{stamp}.jpg"
            cv2.imwrite(str(path), vis)
            print(f"[saved] 闭爪前检测: {path}")

        ok, reason = is_block_graspable(
            det,
            frame,
            target_u=tu,
            target_v=tv,
            tol_u=tol_u,
            tol_v=tol_v,
            min_area=min_area,
            max_area=max_area,
        )
        print(f"[preclose {it+1}/{max_iters}] {reason}")
        if ok:
            return True
        if det is None:
            print("[preclose] 看不见红块，小幅后退后再试")
            ticks = arm.read()
            seed = {n: float(ticks[n]) for n in JOINT_ORDER}
            T = fk.forward_ticks(seed)
            cur = T[:3, 3].copy()
            # 沿 -前进（基座 +Y 近似）退 1cm
            target = np.array([cur[0], cur[1] + 0.01, z_hold], dtype=float)
        else:
            ticks = arm.read()
            seed = {n: float(ticks[n]) for n in JOINT_ORDER}
            T_base_ee = fk.forward_ticks(seed)
            T_base_cam = T_base_ee @ T_ee_cam
            cur = T_base_ee[:3, 3].copy()
            du = float(det.center_u) - tu
            dv = float(det.center_v) - tv
            delta = pixel_error_to_base_delta(
                du, dv, T_base_cam, m_per_px=m_per_px, max_step_m=max_step_m
            )
            target = np.array(
                [cur[0] + delta[0], cur[1] + delta[1], z_hold],
                dtype=float,
            )
            print(
                f"[preclose] 微调 Δxy=({delta[0]*1000:+.1f},{delta[1]*1000:+.1f}) mm "
                f"→ ({target[0]:.4f},{target[1]:.4f},{target[2]:.4f})"
            )

        route = build_forward_route_xyz(cur, target, n_points=3)
        route_ticks = ik_route(
            ik,
            route,
            seed,
            observe_arm,
            cur,
            limits,
            float(ik_tol),
            False,
            float(max_horiz_m),
        )
        for j, arm_ticks in enumerate(route_ticks):
            pose = full_pose_from_ik(arm_ticks, g_open, ticks)
            go_pose_strict(arm, f"preclose{it+1}_wp{j+1}", pose)
            time.sleep(0.03)

    print("[preclose] 达到最大调整次数，仍判定不可抓")
    return False


def set_gripper(arm: TicksArm, ticks: float, limits: dict, settle_s: float) -> None:
    pose = dict(arm.goal) if arm.goal else arm.read()
    pose["gripper"] = float(ticks)
    limited = clamp_pose(pose, limits)
    move_to(arm.bus, pose_i(limited), wait_s=0.0)
    arm.goal = dict(limited)
    time.sleep(float(settle_s))


def gripper_closed_empty(actual_g: float, g_close: float, miss_tol: float) -> bool:
    """实际 ticks 很接近设定闭爪值 → 空握（未夹到物体）。"""
    return abs(float(actual_g) - float(g_close)) <= float(miss_tol)


def save_detection_image(
    out_dir: Path,
    frame: np.ndarray,
    detector: RedBlockDetector,
    det: Optional[RedBlockDetection],
    *,
    tag: str = "det",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis = detector.draw(frame, det)
    if det is not None:
        cv2.drawMarker(
            vis,
            (int(round(det.center_u)), int(round(det.center_v))),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            24,
            2,
        )
        path = out_dir / f"{tag}_{stamp}.jpg"
    else:
        path = out_dir / f"miss_{stamp}.jpg"
    cv2.imwrite(str(path), vis)
    print(f"[saved] 检测图: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SO-101：观察位仅前进抓取红块（不下降）"
    )
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--og-config", type=Path, default=DEFAULT_OG)
    parser.add_argument("--vision-config", type=Path, default=DEFAULT_V3D)
    parser.add_argument("--port", default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--ik-tol", type=float, default=0.012)
    parser.add_argument("--max-horiz-m", type=float, default=0.28)
    parser.add_argument("--route-points", type=int, default=5, help="水平前进路点数")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--range-mode",
        choices=("forward", "vision"),
        default="forward",
        help="forward=视觉定方向+前方距离锚定(默认); vision=纯桌面平面交点",
    )
    parser.add_argument(
        "--forward-m",
        type=float,
        default=0.04,
        help="观察位时木块相对夹爪正前方的水平距离 (m)，默认 0.04≈4cm",
    )
    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.0,
        help="相对观察位高度的 Z 修正 (m)，上为正；偏下2cm用 0.02",
    )
    parser.add_argument(
        "--left-m",
        type=float,
        default=0.005,
        help="木块中心确定后，抓取点相对前进方向向左偏移 (m)，默认 0.005",
    )
    parser.add_argument("--gripper-open", type=float, default=None)
    parser.add_argument(
        "--gripper-close",
        type=float,
        default=800.0,
        help="闭爪 ticks，默认 800",
    )
    parser.add_argument(
        "--open-gripper",
        action="store_true",
        help="抓取前先开爪（默认保持当前夹爪）",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="检测图保存目录（默认 output/grasp_from_observe）",
    )
    parser.add_argument("--no-save-image", action="store_true", help="不保存检测图")
    parser.add_argument(
        "--no-preclose-check",
        action="store_true",
        help="关闭闭爪前二次识别",
    )
    parser.add_argument("--preclose-iters", type=int, default=6, help="闭爪前最大调整次数")
    parser.add_argument("--preclose-tol-u", type=float, default=40.0, help="可抓窗口半宽(px)")
    parser.add_argument("--preclose-tol-v", type=float, default=50.0, help="可抓窗口半高(px)")
    parser.add_argument("--preclose-target-u", type=float, default=None, help="对准目标 u（默认图像中心）")
    parser.add_argument("--preclose-target-v", type=float, default=None, help="对准目标 v（默认约 0.55*高）")
    parser.add_argument("--preclose-m-per-px", type=float, default=0.00012, help="像素→米增益")
    parser.add_argument("--preclose-max-step", type=float, default=0.012, help="单次最大水平微调(m)")
    parser.add_argument("--preclose-min-area", type=float, default=800.0)
    parser.add_argument("--preclose-max-area", type=float, default=120000.0)
    parser.add_argument(
        "--close-even-if-ungraspable",
        action="store_true",
        help="调整失败仍强制闭爪",
    )
    parser.add_argument(
        "--grasp-miss-tol",
        type=float,
        default=50.0,
        help="闭爪后实际ticks与gripper-close差≤该值视为未夹到（空握），默认50",
    )
    parser.add_argument(
        "--grasp-retries",
        type=int,
        default=3,
        help="未夹到时重新抓取最大次数（含首次），默认3",
    )
    parser.add_argument("--release-on-exit", action="store_true")
    args = parser.parse_args()

    if not args.yes:
        print("Refusing motion: pass --yes")
        return 2

    cfg = merge_configs(args.og_config, args.vision_config)
    port = args.port or cfg.get("port", "/dev/ttyACM1")
    baud = int(cfg.get("baud") or DEFAULT_BAUD)
    motion = cfg.get("motion") or {}
    limits = motion.get("joint_limits") or {}
    poses = cfg.get("poses") or {}

    scene = cfg.get("scene") or {}
    table_z = float(scene.get("table_z_m", 0.25))
    table_normal = np.asarray(scene.get("table_normal", [0, 0, 1]), dtype=float)
    _, grasp_z_m, grasp_label = grasp_z({**scene, "table_z_m": table_z})

    ag = cfg.get("auto_grasp") or {}
    observe_pose = poses.get("grasp") or {}
    g_open = float(
        args.gripper_open
        if args.gripper_open is not None
        else ag.get("gripper_open", observe_pose.get("gripper", 1200))
    )
    g_close = float(args.gripper_close)

    K, dist = load_intrinsics(
        Path(cfg.get("intrinsics", "output/camera_calib/camera_intrinsics.yaml"))
    )
    T_ee_cam = load_handeye(Path(cfg.get("handeye", "output/handeye_ee_cam.yaml")))

    camera_cfg = dict(cfg.get("camera") or {})
    if args.camera is not None:
        camera_cfg["index_or_path"] = args.camera
    camera_cfg.setdefault("width", 640)
    camera_cfg.setdefault("height", 480)
    camera_cfg.setdefault("fps", 30)
    camera_cfg.setdefault("settle_frames", 8)
    camera_cfg.setdefault("frame_timeout_s", 5.0)

    detector = RedBlockDetector(cfg)
    fk = SO101FK.from_config(cfg) if isinstance(cfg.get("fk"), dict) else SO101FK()
    ik = SO101IK(fk)

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = Path(
            cfg.get("grasp_observe_output", "output/grasp_from_observe")
        )
    if not save_dir.is_absolute():
        save_dir = (_TTA / save_dir).resolve()

    print(
        f"[scene] table_z={table_z:.3f} grasp_z={float(grasp_z_m):.3f} ({grasp_label}) "
        f"motion=forward-only close={int(g_close)}"
    )
    print("[info] 假定已在观察位；仅水平前进，不下降")
    if not args.no_save_image:
        print(f"[info] 检测图目录: {save_dir}")

    arm = TicksArm(port, baud, motion)
    try:
        arm.connect()
        ticks_now = arm.read()
        print_pose("当前关节(观察位):", ticks_now)
        arm.goal = {j: float(ticks_now.get(j, 2048)) for j in JOINTS}
        arm.current_name = "observe"

        if args.open_gripper:
            set_gripper(arm, g_open, limits, 0.4)
            ticks_now = arm.read()

        observe_arm_cfg = {
            n: float(observe_pose.get(n, 2048)) for n in JOINT_ORDER
        } if observe_pose else None

        do_preclose = not bool(args.no_preclose_check)
        miss_tol = float(args.grasp_miss_tol)
        max_tries = max(1, int(args.grasp_retries))
        grasped = False

        for attempt in range(1, max_tries + 1):
            print(f"\n[grasp] ===== 抓取尝试 {attempt}/{max_tries}（当前位姿视觉重规划）=====")
            set_gripper(arm, g_open, limits, 0.35)

            ticks_now = arm.read()
            seed = {n: float(ticks_now[n]) for n in JOINT_ORDER}
            observe_arm = dict(seed)
            if observe_arm_cfg is not None:
                observe_arm = dict(observe_arm_cfg)

            frame = capture_frame(camera_cfg)
            det = detector.detect(frame)
            tag = f"det{attempt}" if det else f"miss{attempt}"
            if not args.no_save_image:
                save_detection_image(save_dir, frame, detector, det, tag=tag)
            if det is None:
                print("[error] 未检测到红块")
                if attempt < max_tries:
                    print("[grasp] 将重试完整规划")
                    continue
                return 1
            print(
                f"[vision] center=({det.center_u:.1f},{det.center_v:.1f}) area={det.area:.0f}"
            )

            T_base_ee = fk.forward_ticks(seed)
            cur_xyz = T_base_ee[:3, 3].copy()

            res = pixel_to_base(
                float(det.center_u),
                float(det.center_v),
                T_base_ee,
                T_ee_cam,
                K,
                dist,
                table_z,
                table_normal,
                float(grasp_z_m),
            )
            p = res["p_grasp"]
            horiz_vis = float(np.hypot(p[0] - cur_xyz[0], p[1] - cur_xyz[1]))
            print(
                f"[3d] 视觉桌面交点 XYZ=({p[0]:.4f},{p[1]:.4f},{p[2]:.4f}) m "
                f"距夹爪水平 {horiz_vis*1000:.0f} mm"
            )
            print(
                f"[fk] ee_xyz=({cur_xyz[0]:.4f},{cur_xyz[1]:.4f},{cur_xyz[2]:.4f}) m"
            )

            if args.range_mode == "forward":
                anchored = anchor_grasp_forward(
                    cur_xyz, p, float(args.forward_m), float(grasp_z_m)
                )
                p = anchored["p_grasp"]
                print(
                    f"[3d] 前方锚定 forward={args.forward_m*1000:.0f} mm "
                    f"→ XY=({p[0]:.4f},{p[1]:.4f}) "
                    f"(视觉水平 {float(anchored['vision_horiz_m'][0])*1000:.0f} mm → "
                    f"{args.forward_m*1000:.0f} mm)"
                )
                if horiz_vis > max(0.12, 2.5 * float(args.forward_m)):
                    print(
                        "[warn] 视觉尺度明显偏大（手眼/桌高可能不准）；"
                        "已用 --forward-m 纠正，建议稍后重标定"
                    )
            else:
                print("[3d] range-mode=vision，使用桌面交点 XY（不锚定）")

            p_center = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=float)
            p = offset_grasp_left(cur_xyz, p_center, float(args.left_m))
            if abs(float(args.left_m)) >= 1e-9:
                print(
                    f"[3d] 抓取点左偏 {args.left_m*1000:.1f} mm: "
                    f"center XY=({p_center[0]:.4f},{p_center[1]:.4f}) → "
                    f"grasp XY=({p[0]:.4f},{p[1]:.4f})"
                )

            target_grasp = np.array(
                [
                    float(p[0]),
                    float(p[1]),
                    float(cur_xyz[2]) + float(args.z_offset),
                ],
                dtype=float,
            )
            print(
                f"[plan] 前进目标 ee=({target_grasp[0]:.4f},{target_grasp[1]:.4f},{target_grasp[2]:.4f}) "
                f"(Z=观察位{float(args.z_offset):+.3f} m)"
            )

            route_xyz = build_forward_route_xyz(
                cur_xyz, target_grasp, int(args.route_points)
            )
            try:
                route_ticks = ik_route(
                    ik,
                    route_xyz,
                    seed,
                    observe_arm,
                    cur_xyz,
                    limits,
                    float(args.ik_tol),
                    bool(args.allow_partial),
                    float(args.max_horiz_m),
                )
            except RuntimeError as exc:
                print(f"[error] {exc}")
                if attempt < max_tries:
                    continue
                return 1

            for i, arm_ticks in enumerate(route_ticks):
                pose = full_pose_from_ik(arm_ticks, g_open, ticks_now)
                go_pose_strict(arm, f"a{attempt}_wp{i+1}", pose)
                time.sleep(0.05)

            if do_preclose:
                ticks_now = arm.read()
                T_now = fk.forward_ticks({n: float(ticks_now[n]) for n in JOINT_ORDER})
                z_hold = float(T_now[2, 3])
                graspable = refine_pose_until_graspable(
                    arm=arm,
                    fk=fk,
                    ik=ik,
                    detector=detector,
                    camera_cfg=camera_cfg,
                    T_ee_cam=T_ee_cam,
                    limits=limits,
                    g_open=g_open,
                    observe_arm=observe_arm,
                    save_dir=save_dir,
                    no_save=bool(args.no_save_image),
                    ik_tol=float(args.ik_tol),
                    max_horiz_m=float(args.max_horiz_m),
                    z_hold=z_hold,
                    target_u=args.preclose_target_u,
                    target_v=args.preclose_target_v,
                    tol_u=float(args.preclose_tol_u),
                    tol_v=float(args.preclose_tol_v),
                    min_area=float(args.preclose_min_area),
                    max_area=float(args.preclose_max_area),
                    m_per_px=float(args.preclose_m_per_px),
                    max_step_m=float(args.preclose_max_step),
                    max_iters=int(args.preclose_iters),
                    settle_s=0.35,
                )
                if not graspable and not args.close_even_if_ungraspable:
                    print("[grasp] 闭爪前判定不可抓 → 将基于当前位姿视觉重规划")
                    if attempt < max_tries:
                        continue
                    print("[grasp] 多次对准失败，取消闭爪并保持")
                    hold_forever(arm, motion)
                    return 1
                if not graspable:
                    print("[grasp] 仍不可抓，按 --close-even-if-ungraspable 强制闭爪")

            print(f"[grasp] 闭爪 -> {int(g_close)}")
            set_gripper(arm, g_close, limits, 0.85)
            actual = arm.read()
            actual_g = float(actual.get("gripper", g_close))
            gap = abs(actual_g - float(g_close))
            print(
                f"[grasp] 闭爪后实际 gripper={actual_g:.0f} "
                f"目标={float(g_close):.0f} |Δ|={gap:.0f} (miss_tol={miss_tol:.0f})"
            )

            if gripper_closed_empty(actual_g, g_close, miss_tol):
                print(
                    "[grasp] 空握（接近 gripper-close）→ 开爪并基于当前位姿视觉重规划"
                )
                set_gripper(arm, g_open, limits, 0.45)
                continue

            print("[grasp] |Δ| 足够大 → 判定已夹住物体")
            grasped = True
            break

        if not grasped:
            print("[grasp] 多次当前位姿重规划后仍失败，保持当前状态")
            hold_forever(arm, motion)
            return 1

        hold_forever(arm, motion)
        return 0
    except KeyboardInterrupt:
        print("\n用户中断")
        return 130
    finally:
        arm.disconnect(release_torque=bool(args.release_on_exit))
        if not args.release_on_exit:
            print("已断开串口；力矩未主动关闭（可用 --release-on-exit）")


if __name__ == "__main__":
    raise SystemExit(main())
