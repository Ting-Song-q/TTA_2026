#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 三维抓取。

示教位仅用：
  - initial  初始
  - grasp    观察位（开爪）

观察位识别红块后：
  - 像素→基座三维
  - 自动生成笛卡尔接近路线并 IK 成关节指令（无固定抓取位姿）
  - 平滑跟踪目标 ticks（不因偏差改持实际位）

用法:
  python3 grasp_3d.py --yes
  python3 grasp_3d.py --yes --route-points 6
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from og import JOINTS, TicksArm, clamp_pose, pose_i, print_pose  # noqa: E402
from pixel_to_base import (  # noqa: E402
    grasp_z,
    load_handeye,
    load_intrinsics,
    load_yaml,
    pixel_to_base,
    resolve_path,
)
from so101_fk import JOINT_ORDER, SO101FK  # noqa: E402
from so101_ik import SO101IK  # noqa: E402
from so101_red_block_camera_test import RedBlockDetector, capture_frame  # noqa: E402
from start import DEFAULT_BAUD, move_to  # noqa: E402

DEFAULT_OG = _HERE / "og.yaml"
DEFAULT_V3D = _HERE / "vision_3d.yaml"


def deep_update(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def merge_configs(og_path: Path, v3d_path: Path) -> dict:
    cfg: dict = {}
    if og_path.exists():
        cfg = deep_update(cfg, load_yaml(resolve_path(og_path)))
    if v3d_path.exists():
        cfg = deep_update(cfg, load_yaml(resolve_path(v3d_path)))
    return cfg


def full_pose_from_ik(
    arm_ticks: Mapping[str, float],
    gripper: float,
    seed: Mapping[str, float],
) -> Dict[str, float]:
    pose = {n: float(seed.get(n, 2048)) for n in JOINTS}
    for n in JOINT_ORDER:
        pose[n] = float(arm_ticks[n])
    pose["gripper"] = float(gripper)
    return pose


def go_pose_strict(arm: TicksArm, name: str, target: Dict[str, float]) -> Dict[str, float]:
    """平滑到目标 ticks，并始终把 goal 设为设定值（不因跟踪误差改持实际位）。"""
    assert arm.bus is not None
    limits = arm.motion.get("joint_limits") or {}
    target = clamp_pose({k: float(v) for k, v in target.items()}, limits)
    step = float(arm.motion["step_ticks"])
    step_s = float(arm.motion["step_s"])
    tol = float(arm.motion["arrive_tol_ticks"])

    print(f"\n>>> 跟踪设定 ticks: {arm.current_name or 'current'} -> {name}")
    print_pose("目标 ticks:", target)

    current = arm.read()
    max_delta = max(
        abs(float(target[n]) - float(current.get(n, target[n]))) for n in target
    )
    steps = max(1, int(np.ceil(max_delta / max(step, 1.0))))

    for i in range(steps):
        actual = arm.read()
        command: Dict[str, float] = {}
        done = True
        for joint, goal in target.items():
            cur = float(actual.get(joint, current.get(joint, goal)))
            err = float(goal) - cur
            if abs(err) > tol:
                done = False
            command[joint] = cur + max(-step, min(step, err))
        limited = clamp_pose(command, limits)
        move_to(arm.bus, pose_i(limited), wait_s=0.0)
        arm.goal = dict(target)
        if done:
            break
        time.sleep(step_s)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  step {i + 1}/{steps}")

    limited_t = clamp_pose(target, limits)
    move_to(arm.bus, pose_i(limited_t), wait_s=0.0)
    arm.goal = dict(limited_t)
    time.sleep(float(arm.motion["settle_s"]))
    final = arm.read()
    print_pose("实际到达:", final)
    max_track_err = max(
        abs(float(final.get(j, g)) - float(g)) for j, g in limited_t.items()
    )
    if max_track_err > max(tol * 4.0, 80.0):
        print(
            f"[warn] 实际偏差 {max_track_err:.0f} ticks，仍继续持设定目标 "
            f"（不考虑过载回退）"
        )
    arm.current_name = name
    print(f"[holding] 位姿={name}，持设定 ticks")
    return final


def build_grasp_route_xyz(
    cur_xyz: np.ndarray,
    target_grasp: np.ndarray,
    approach_z: float,
    n_horiz: int,
    n_down: int,
) -> List[np.ndarray]:
    """笛卡尔路线：水平靠近目标 XY（保持当前高度）→ 再竖直下降到抓取高度。"""
    z_pre = float(cur_xyz[2])
    pre = np.array([target_grasp[0], target_grasp[1], z_pre], dtype=float)
    # 若当前已经很高，可略降到 grasp+approach 再水平
    z_alt = float(target_grasp[2] + max(approach_z, 0.03))
    if z_pre > z_alt + 0.04:
        pre = np.array([target_grasp[0], target_grasp[1], z_alt], dtype=float)

    route: List[np.ndarray] = []
    n_h = max(2, int(n_horiz))
    for i in range(1, n_h + 1):
        a = i / float(n_h)
        route.append(cur_xyz + a * (pre - cur_xyz))
    n_d = max(1, int(n_down))
    start_down = route[-1]
    for i in range(1, n_d + 1):
        a = i / float(n_d)
        route.append(start_down + a * (target_grasp - start_down))
    return route


def ik_route(
    ik: SO101IK,
    route_xyz: List[np.ndarray],
    seed: Dict[str, float],
    observe_arm: Dict[str, float],
    start_xyz: np.ndarray,
    limits: dict,
    ik_tol: float,
    allow_partial: bool,
    max_horiz_m: float,
) -> List[Dict[str, float]]:
    """将笛卡尔路点逐点 IK；误差再大也继续用当前最优解（不中断、不截断）。"""
    horiz = float(np.hypot(route_xyz[-1][0], route_xyz[-1][1]))
    print(f"[route] 路点数={len(route_xyz)} 终点水平半径={horiz*1000:.0f} mm")
    if horiz > max_horiz_m:
        print("[warn] 终点偏远，可能超出工作空间")

    seeds: List[Dict[str, float]] = [seed, observe_arm]
    stretched = dict(observe_arm)
    stretched["shoulder_lift"] = float(np.clip(stretched["shoulder_lift"] + 200, 600, 3400))
    stretched["elbow_flex"] = float(np.clip(stretched["elbow_flex"] - 200, 600, 3400))
    seeds.append(stretched)

    ticks_list: List[Dict[str, float]] = []
    cur_seed = dict(seed)
    accept = max(ik_tol, 0.02)

    for i, xyz in enumerate(route_xyz):
        ticks, info = ik.solve_position_best(
            xyz,
            [cur_seed] + seeds,
            joint_limits=limits,
            pos_tol_m=ik_tol,
            max_iters=120,
        )
        err = float(info["pos_err_m"])
        print(
            f"[ik] wp{i+1}/{len(route_xyz)} "
            f"xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) "
            f"err={err*1000:.1f}mm ok={bool(info['ok'])}"
        )
        if err > accept:
            print(f"[ik] wp{i+1} 误差偏大，仍继续执行")

        ticks_list.append(ticks)
        cur_seed = ticks
        seeds = [ticks, observe_arm, stretched]

    if not ticks_list:
        raise RuntimeError("未生成任何路径点")
    return ticks_list


def hold_forever(arm: TicksArm, motion: dict) -> None:
    print("[hold] 抓取完成，保持设定 ticks；Ctrl+C 退出")
    hz = float(motion.get("hold_hz", 10.0))
    period = 1.0 / max(hz, 0.5)
    while True:
        arm.hold_tick()
        time.sleep(period)


def main() -> int:
    parser = argparse.ArgumentParser(description="SO-101：观察后自动规划抓取路线")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--og-config", type=Path, default=DEFAULT_OG)
    parser.add_argument("--vision-config", type=Path, default=DEFAULT_V3D)
    parser.add_argument("--port", default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--approach-z", type=float, default=0.05)
    parser.add_argument("--ee-z-offset", type=float, default=0.04)
    parser.add_argument("--ik-tol", type=float, default=0.012)
    parser.add_argument("--max-horiz-m", type=float, default=0.28)
    parser.add_argument("--route-horiz", type=int, default=5, help="水平段路点数")
    parser.add_argument("--route-down", type=int, default=4, help="下降段路点数")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--gripper-open", type=float, default=None)
    parser.add_argument("--gripper-close", type=float, default=None)
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
    if "initial" not in poses or "grasp" not in poses:
        print("[error] og.yaml 需要 poses.initial 与 poses.grasp（观察位）")
        return 2

    scene = cfg.get("scene") or {}
    table_z = float(scene.get("table_z_m", 0.25))
    table_normal = np.asarray(scene.get("table_normal", [0, 0, 1]), dtype=float)
    _, grasp_z_m, grasp_label = grasp_z({**scene, "table_z_m": table_z})

    ag = cfg.get("auto_grasp") or {}
    g_open = float(
        args.gripper_open
        if args.gripper_open is not None
        else ag.get("gripper_open", poses["grasp"].get("gripper", 1200))
    )
    g_close = float(
        args.gripper_close
        if args.gripper_close is not None
        else ag.get("gripper_close", 2100)
    )

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

    print(
        f"[scene] table_z={table_z:.3f} grasp_z={float(grasp_z_m):.3f} ({grasp_label}) "
        f"ee_z_offset={args.ee_z_offset:.3f} approach={args.approach_z:.3f}"
    )

    arm = TicksArm(port, baud, motion)
    try:
        arm.connect()

        # 仅示教：初始 → 观察
        initial = {j: float(poses["initial"][j]) for j in JOINTS}
        observe = {j: float(poses["grasp"][j]) for j in JOINTS}
        observe["gripper"] = g_open

        print_pose("初始位(示教):", initial)
        go_pose_strict(arm, "initial", initial)
        time.sleep(0.3)

        print_pose("观察位(示教):", observe)
        go_pose_strict(arm, "observe", observe)
        time.sleep(0.5)

        # 识别后自动规划路线（非固定抓取位姿）
        frame = capture_frame(camera_cfg)
        det = detector.detect(frame)
        if det is None:
            print("[error] 观察位未检测到红块，取消")
            return 1
        print(
            f"[vision] center=({det.center_u:.1f},{det.center_v:.1f}) area={det.area:.0f}"
        )

        ticks_now = arm.read()
        seed = {n: float(ticks_now[n]) for n in JOINT_ORDER}
        observe_arm = {n: float(observe[n]) for n in JOINT_ORDER}
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
        print(
            f"[3d] 方块中心 XYZ=({p[0]:.4f},{p[1]:.4f},{p[2]:.4f}) m "
            f"→ mm ({p[0]*1000:.1f},{p[1]*1000:.1f},{p[2]*1000:.1f})"
        )

        target_grasp = np.array(
            [float(p[0]), float(p[1]), float(p[2]) + float(args.ee_z_offset)],
            dtype=float,
        )
        print(
            f"[plan] 抓取目标 ee=({target_grasp[0]:.4f},{target_grasp[1]:.4f},{target_grasp[2]:.4f})"
        )

        route_xyz = build_grasp_route_xyz(
            cur_xyz,
            target_grasp,
            float(args.approach_z),
            int(args.route_horiz),
            int(args.route_down),
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
            return 1

        # 按自动路线逐点跟踪设定 ticks
        for i, arm_ticks in enumerate(route_ticks):
            pose = full_pose_from_ik(arm_ticks, g_open, ticks_now)
            go_pose_strict(arm, f"wp{i+1}", pose)
            time.sleep(0.05)

        print(f"[grasp] 闭爪 -> {int(g_close)}")
        closed = dict(arm.goal)
        closed["gripper"] = g_close
        limited = clamp_pose(closed, limits)
        move_to(arm.bus, pose_i(limited), wait_s=0.0)
        arm.goal = dict(limited)
        time.sleep(0.85)

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
