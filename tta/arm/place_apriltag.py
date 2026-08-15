#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 AprilTag 放置。

流程:
  1. 进入放置位姿（持块）
  2. 识别 AprilTag，每次识别后存图
  3. 像素视觉伺服：Tag 中心 → 夹爪尖端 (320,400)
     - 左右：相机 X 水平投影
     - 前后：相机光轴 Z 水平投影（Tag 在上方则前移靠近）
     - 保持 Z，不用关节 reach 上抬
  4. 连续若干帧落入容差窗即到位；丢标则回退上一步缩小步长重试
  5. 开爪放下

用法:
  python3 place_apriltag.py --yes
  python3 place_apriltag.py --yes --target-u 320 --target-v 400
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from grasp_from_observe import build_forward_route_xyz  # noqa: E402
from og import JOINTS, TicksArm, clamp_pose, pose_i, print_pose  # noqa: E402
from pixel_to_base import load_handeye, load_intrinsics  # noqa: E402
from so101_fk import JOINT_ORDER, SO101FK  # noqa: E402
from so101_ik import SO101IK  # noqa: E402
from so101_red_block_camera_test import camera_source  # noqa: E402
from start import DEFAULT_BAUD, move_to  # noqa: E402

# 放置位姿（持块）
PLACE_POSE: Dict[str, float] = {
    "shoulder_pan": 2045,
    "shoulder_lift": 1379,
    "elbow_flex": 1942,
    "wrist_flex": 3337,
    "wrist_roll": 1985,
    "gripper": 799,
}

DETECT_MAX_WIDTH = 960

# 夹爪尖端在腕部相机图像上的投影（放置位姿标定）
DEFAULT_GRIPPER_TIP_U = 320.0
DEFAULT_GRIPPER_TIP_V = 400.0


# ---------------------------------------------------------------------------
# AprilTag 检测
# ---------------------------------------------------------------------------


class AprilTagDetector:
    def __init__(self, family: str = "tag36h11"):
        self.family = family
        self.kind = ""
        self.impl = None
        self._init_backend()

    def _init_backend(self) -> None:
        for mod_name, kind in (("pupil_apriltags", "pupil"), ("pyapriltags", "pyapriltags")):
            try:
                mod = __import__(mod_name)
                self.impl = mod.Detector(families=self.family)
                self.kind = kind
                print(f"[apriltag] backend={kind}")
                return
            except Exception:
                pass
        try:
            import apriltag

            self.impl = apriltag.Detector(apriltag.DetectorOptions(families=self.family))
            self.kind = "apriltag"
            print("[apriltag] backend=apriltag")
            return
        except Exception:
            pass
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "无可用 AprilTag 后端。请: pip3 install pupil-apriltags 或 opencv-contrib-python"
            )
        name = "DICT_APRILTAG_36h11"
        dict_const = getattr(cv2.aruco, name, None) or getattr(
            cv2.aruco, "DICT_APRILTAG_36H11", None
        )
        if dict_const is None:
            raise RuntimeError("OpenCV 无 AprilTag 字典，请安装 opencv-contrib-python")
        get = getattr(cv2.aruco, "getPredefinedDictionary", None)
        dictionary = get(dict_const) if get else cv2.aruco.Dictionary_get(dict_const)
        params = cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.impl = cv2.aruco.ArucoDetector(dictionary, params)
        else:
            self.impl = (dictionary, params)
        self.kind = "opencv"
        print("[apriltag] backend=opencv")

    def detect(self, frame: np.ndarray, tag_id: Optional[int] = None) -> Optional[dict]:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = 1.0
        gray = gray_full
        if gray_full.shape[1] > DETECT_MAX_WIDTH:
            scale = DETECT_MAX_WIDTH / float(gray_full.shape[1])
            gray = cv2.resize(
                gray_full,
                (DETECT_MAX_WIDTH, int(gray_full.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        raw = []
        if self.kind in {"pupil", "pyapriltags", "apriltag"}:
            raw = list(self.impl.detect(gray))
        else:
            if hasattr(self.impl, "detectMarkers"):
                corners, ids, _ = self.impl.detectMarkers(gray)
            else:
                dictionary, params = self.impl
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, dictionary, parameters=params
                )
            if ids is not None:
                for i, c in enumerate(corners):
                    t = type("T", (), {})()
                    t.tag_id = int(ids[i][0])
                    t.corners = np.asarray(c, dtype=float).reshape(4, 2)
                    raw.append(t)

        best = None
        for tag in raw:
            tid = int(getattr(tag, "tag_id", -1))
            if tag_id is not None and tid != int(tag_id):
                continue
            corners = np.asarray(tag.corners, dtype=float) / scale
            area = float(cv2.contourArea(corners.astype(np.float32)))
            if best is None or area > best["area"]:
                c = corners.mean(axis=0)
                best = {
                    "tag_id": tid,
                    "corners": corners,
                    "center_u": float(c[0]),
                    "center_v": float(c[1]),
                    "area": area,
                }
        return best


# ---------------------------------------------------------------------------
# 相机 / 工具
# ---------------------------------------------------------------------------


def _list_video_candidates() -> list:
    """列出本机可能的摄像头候选（路径 + 索引）。"""
    cands: list = []
    if os.name != "nt":
        try:
            for p in sorted(Path("/dev").glob("video*")):
                cands.append(str(p))
        except Exception:
            pass
    for i in range(0, 16):
        cands.append(i)
    return cands


def _try_open_capture(source) -> Optional[cv2.VideoCapture]:
    """尝试多种后端打开相机；失败返回 None。"""
    backends: list = []
    if isinstance(source, int):
        if os.name == "nt":
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        # 设备节点路径也优先 V4L2
        if os.name != "nt":
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]
        # 纯数字字符串已在 camera_source 转 int；这里再兜底一次
        text = str(source)
        if text.isdigit():
            return _try_open_capture(int(text))

    for be in backends:
        cap = cv2.VideoCapture(source, be)
        if cap is not None and cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            cap.release()
        elif cap is not None:
            cap.release()
    # 最后再试默认后端
    cap = cv2.VideoCapture(source)
    if cap is not None and cap.isOpened():
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        cap.release()
    return None


class LiveCamera:
    def __init__(self, camera_cfg: dict) -> None:
        self.cfg = camera_cfg
        preferred = camera_source(camera_cfg["index_or_path"])
        self.cap: Optional[cv2.VideoCapture] = None
        self.source_used = preferred

        # 1) 先开配置指定的相机
        self.cap = _try_open_capture(preferred)
        # 2) 失败则自动扫描其它节点（跳过已试过的）
        if self.cap is None:
            print(f"[camera] 无法打开 {preferred!r}，尝试自动扫描...")
            tried = {str(preferred)}
            for cand in _list_video_candidates():
                if str(cand) in tried:
                    continue
                # 路径不存在则跳过
                if isinstance(cand, str) and cand.startswith("/dev/") and not Path(cand).exists():
                    continue
                cap = _try_open_capture(cand)
                if cap is not None:
                    self.cap = cap
                    self.source_used = cand
                    print(f"[camera] 自动选用 {cand!r}")
                    break

        if self.cap is None or not self.cap.isOpened():
            avail = []
            for cand in _list_video_candidates():
                if isinstance(cand, str) and Path(cand).exists():
                    avail.append(cand)
            hint = ", ".join(avail[:12]) if avail else "无 /dev/video*"
            raise RuntimeError(
                f"cannot open wrist camera: {preferred!r}\n"
                f"  现有设备: {hint}\n"
                f"  请改 og.yaml 的 camera.index_or_path，或命令行指定，例如:\n"
                f"    python3 place_apriltag.py --yes --camera 0\n"
                f"    python3 place_apriltag.py --yes --camera /dev/video0"
            )

        print(f"[camera] opened {self.source_used!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_cfg["width"]))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_cfg["height"]))
        self.cap.set(cv2.CAP_PROP_FPS, int(camera_cfg["fps"]))
        for _ in range(max(1, int(camera_cfg.get("settle_frames", 1)))):
            self.cap.read()

    def grab(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("wrist camera returned empty frame")
        return frame

    def close(self) -> None:
        self.cap.release()


def set_gripper(arm: TicksArm, value: float, limits: dict, settle_s: float = 0.25) -> None:
    pose = dict(arm.goal) if arm.goal else arm.read()
    pose["gripper"] = float(value)
    limited = clamp_pose(pose, limits)
    move_to(arm.bus, pose_i(limited), wait_s=0.0)
    arm.goal = dict(limited)
    time.sleep(settle_s)


def snapshot_pose(arm: TicksArm) -> Dict[str, float]:
    src = arm.goal if arm.goal else arm.read()
    return {k: float(v) for k, v in src.items()}


def gripper_tip_uv(
    frame_shape: tuple,
    tip_u: Optional[float],
    tip_v: Optional[float],
) -> Tuple[float, float]:
    """夹爪尖端在图像中的目标像素（非摄像头中心）。"""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    tu = float(tip_u) if tip_u is not None else float(DEFAULT_GRIPPER_TIP_U)
    tv = float(tip_v) if tip_v is not None else float(DEFAULT_GRIPPER_TIP_V)
    if tip_u is None and w > 0 and abs(w - 640) > 1:
        tu = tu * (w / 640.0)
    if tip_v is None and h > 0 and abs(h - 480) > 1:
        tv = tv * (h / 480.0)
    return tu, tv


def save_debug(
    out_dir: Path,
    frame: np.ndarray,
    det: Optional[dict],
    tu: float,
    tv: float,
    tol_u: float,
    tol_v: float,
    tag: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    vis = frame.copy()
    cv2.drawMarker(vis, (int(tu), int(tv)), (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.rectangle(
        vis,
        (int(tu - tol_u), int(tv - tol_v)),
        (int(tu + tol_u), int(tv + tol_v)),
        (0, 255, 0),
        2,
    )
    cv2.putText(
        vis,
        "gripper tip",
        (int(tu) + 10, max(16, int(tv) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
    cx_img = frame.shape[1] // 2
    cy_img = frame.shape[0] // 2
    cv2.drawMarker(vis, (cx_img, cy_img), (128, 128, 128), cv2.MARKER_TILTED_CROSS, 14, 1)
    cv2.putText(
        vis,
        "cam center",
        (cx_img + 8, cy_img + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (128, 128, 128),
        1,
    )
    if det is not None:
        pts = np.round(det["corners"]).astype(int)
        cv2.polylines(vis, [pts], True, (0, 165, 255), 2)
        cv2.drawMarker(
            vis,
            (int(det["center_u"]), int(det["center_v"])),
            (255, 0, 0),
            cv2.MARKER_CROSS,
            18,
            2,
        )
        cv2.putText(
            vis,
            f"id={det['tag_id']} tip_err=({det['center_u']-tu:+.0f},{det['center_v']-tv:+.0f})",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (50, 220, 50),
            2,
        )
    else:
        cv2.putText(vis, "no tag", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
    path = out_dir / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    cv2.imwrite(str(path), vis)
    print(f"[saved] {path}")
    return path


def detect_and_save(
    *,
    detector: AprilTagDetector,
    camera: LiveCamera,
    tag_id: Optional[int],
    target_u: Optional[float],
    target_v: Optional[float],
    tol_u: float,
    tol_v: float,
    save_dir: Path,
    no_save: bool,
    tag: str,
    flush: int = 2,
) -> Tuple[Optional[dict], np.ndarray, float, float]:
    for _ in range(max(0, int(flush))):
        try:
            camera.grab()
        except Exception:
            break
    frame = camera.grab()
    tu, tv = gripper_tip_uv(frame.shape, target_u, target_v)
    det = detector.detect(frame, tag_id=tag_id)
    if not no_save:
        save_debug(save_dir, frame, det, tu, tv, tol_u, tol_v, tag)
    return det, frame, tu, tv


def ee_xyz_now(arm: TicksArm, fk: SO101FK) -> Tuple[Dict[str, float], np.ndarray]:
    ticks = arm.read()
    seed = {n: float(ticks[n]) for n in JOINT_ORDER}
    xyz = fk.forward_ticks(seed)[:3, 3].copy()
    return seed, xyz


def pixel_error_to_approach_xy(
    err_u: float,
    err_v: float,
    T_base_cam: np.ndarray,
    *,
    m_per_px: float,
    max_step_m: float,
    approach_sign: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """大像素误差用「左右 + 光轴前移」，不用图像平面 xy 平移。

    - err_u → 相机 X 投到水平面（左右）
    - err_v → 相机 Z 投到水平面（前后）：Tag 在尖端上方(err_v<0) 时前移，
      把 Tag 拉向图像下方靠近夹爪尖端
    """
    R = np.asarray(T_base_cam[:3, :3], dtype=float)
    cam_x = R[:, 0]
    cam_z = R[:, 2]
    lat = np.array([cam_x[0], cam_x[1], 0.0], dtype=float)
    fwd = np.array([cam_z[0], cam_z[1], 0.0], dtype=float)
    nl = float(np.linalg.norm(lat[:2]))
    nf = float(np.linalg.norm(fwd[:2]))
    if nl < 1e-4:
        lat = np.array([1.0, 0.0, 0.0], dtype=float)
        nl = 1.0
    if nf < 1e-4:
        # 光轴近竖直时，SO-101 放置位常用 -Y 为前方
        fwd = np.array([0.0, -1.0, 0.0], dtype=float)
        nf = 1.0
    lat[:2] /= nl
    fwd[:2] /= nf

    # Tag 偏右(err_u>0)→右移；Tag 偏上(err_v<0)→沿 fwd 前移
    d = (float(err_u) * float(m_per_px)) * lat + (
        float(-err_v) * float(m_per_px) * float(approach_sign)
    ) * fwd
    d[2] = 0.0
    n = float(np.linalg.norm(d[:2]))
    if n > float(max_step_m) > 0:
        d[:2] *= float(max_step_m) / n
    return d, lat, fwd


def move_xy_hold_z(
    *,
    arm: TicksArm,
    fk: SO101FK,
    ik: SO101IK,
    target_xyz: np.ndarray,
    place_arm: Dict[str, float],
    limits: dict,
    g_hold: float,
    ik_tol: float,
    allow_partial: bool,
    max_horiz_m: float,
    name: str,
    n_points: int = 3,
) -> None:
    """水平面移动到 target（保持 Z）；末点用 go_pose_strict 确保真正到位。"""
    ticks_now = arm.read()
    seed = {n: float(ticks_now[n]) for n in JOINT_ORDER}
    cur = fk.forward_ticks(seed)[:3, 3].copy()
    goal = np.asarray(target_xyz, dtype=float).copy()
    goal[2] = float(cur[2])
    # 若水平位移极小，强制至少沿目标方向走一点，避免 IK 原地踏步
    horiz = float(np.hypot(goal[0] - cur[0], goal[1] - cur[1]))
    if horiz < 1e-4:
        print(f"[align] skip tiny move {horiz*1000:.2f}mm")
        return

    route = build_forward_route_xyz(cur, goal, max(2, int(n_points)))
    route_ticks = ik_route(
        ik,
        route,
        seed,
        place_arm,
        cur,
        limits,
        float(ik_tol),
        bool(allow_partial),
        float(max_horiz_m),
    )
    for j, arm_ticks in enumerate(route_ticks):
        pose = full_pose_from_ik(arm_ticks, g_hold, ticks_now)
        if j < len(route_ticks) - 1:
            limited = clamp_pose(pose, limits)
            move_to(arm.bus, pose_i(limited), wait_s=0.0)
            arm.goal = dict(limited)
            time.sleep(0.03)
        else:
            go_pose_strict(arm, name, pose)


# ---------------------------------------------------------------------------
# 视觉伺服对齐：水平前移（参考 grasp_from_observe）
# ---------------------------------------------------------------------------


def align_apriltag(
    *,
    arm: TicksArm,
    fk: SO101FK,
    ik: SO101IK,
    detector: AprilTagDetector,
    camera: LiveCamera,
    T_ee_cam: np.ndarray,
    limits: dict,
    g_hold: float,
    place_arm: Dict[str, float],
    tag_id: Optional[int],
    target_u: Optional[float],
    target_v: Optional[float],
    tol_u: float,
    tol_v: float,
    m_per_px: float,
    max_step_m: float,
    stable_need: int,
    timeout_s: float,
    settle_s: float,
    loop_s: float,
    lose_grace_s: float,
    step_retries: int,
    ik_tol: float,
    max_horiz_m: float,
    allow_partial: bool,
    xy_sign: float,
    save_dir: Path,
    no_save: bool,
    save_every: int,
) -> bool:
    """像素误差 → 沿相机光轴水平前移 + 左右平移，保持 Z 对齐夹爪尖端。"""
    _, place_xyz = ee_xyz_now(arm, fk)
    z_hold = float(place_xyz[2])

    stable = 0
    step_i = 0
    deadline = time.monotonic() + float(timeout_s)
    save_every = max(1, int(save_every))
    # approach_sign: Tag 在上方时沿光轴前移的方向；发散则自动翻转
    approach_sign = float(xy_sign)
    if abs(approach_sign) < 1e-6:
        approach_sign = 1.0
    approach_sign = 1.0 if approach_sign > 0 else -1.0
    flipped = False

    last_seen_u: Optional[float] = None
    last_seen_v: Optional[float] = None
    last_seen_t = 0.0
    last_delta = np.zeros(3, dtype=float)

    probe = camera.grab()
    tu0, tv0 = gripper_tip_uv(probe.shape, target_u, target_v)
    print(
        f"[align] 光轴前移伺服 夹爪尖端=({tu0:.0f},{tv0:.0f}) "
        f"z_hold={z_hold:.4f}m tol=({tol_u:.0f},{tol_v:.0f}) "
        f"approach_sign={approach_sign:+.0f} max_step={max_step_m*1000:.0f}mm "
        f"timeout={timeout_s:.0f}s"
    )

    def _detect(tag: str, do_save: bool) -> Tuple[Optional[dict], float, float]:
        det, _, tu, tv = detect_and_save(
            detector=detector,
            camera=camera,
            tag_id=tag_id,
            target_u=target_u,
            target_v=target_v,
            tol_u=tol_u,
            tol_v=tol_v,
            save_dir=save_dir,
            no_save=(no_save or not do_save),
            tag=tag,
            flush=2,
        )
        return det, tu, tv

    def _err_norm(eu: float, ev: float) -> float:
        return float(np.hypot(eu, ev))

    def _move_delta(delta_xy: np.ndarray, name: str) -> None:
        _, cur = ee_xyz_now(arm, fk)
        target = np.array(
            [cur[0] + float(delta_xy[0]), cur[1] + float(delta_xy[1]), z_hold],
            dtype=float,
        )
        move_xy_hold_z(
            arm=arm,
            fk=fk,
            ik=ik,
            target_xyz=target,
            place_arm=place_arm,
            limits=limits,
            g_hold=g_hold,
            ik_tol=ik_tol,
            allow_partial=allow_partial,
            max_horiz_m=max_horiz_m,
            name=name,
            n_points=3,
        )
        time.sleep(float(settle_s))

    while time.monotonic() < deadline:
        arm.hold_tick()
        step_i += 1
        do_save = (step_i % save_every == 0) or (step_i <= 2)
        det, tu, tv = _detect(f"align{step_i}", do_save)

        if det is None:
            if last_seen_u is not None and (time.time() - last_seen_t) < float(lose_grace_s):
                print(
                    f"[align] 短时丢标，反向半步找回 "
                    f"({-0.5*last_delta[0]*1000:+.1f},{-0.5*last_delta[1]*1000:+.1f}) mm"
                )
                prev = snapshot_pose(arm)
                try:
                    # 丢标多半因走过头/方向反，回半步
                    _move_delta(-0.5 * last_delta, f"dash{step_i}")
                except Exception as exc:
                    print(f"[warn] 冲刺失败: {exc}")
                    go_pose_strict(arm, "dash_rb", prev)
                    continue
                det2, _, _ = _detect(f"dash{step_i}", True)
                if det2 is None:
                    print("[align] 仍无标，恢复上一步并反号")
                    go_pose_strict(arm, "dash_rb", prev)
                    time.sleep(0.3)
                    if not flipped:
                        approach_sign *= -1.0
                        flipped = True
                        print(f"[align] approach_sign → {approach_sign:+.0f}")
                else:
                    last_seen_u = float(det2["center_u"])
                    last_seen_v = float(det2["center_v"])
                    last_seen_t = time.time()
            else:
                print(f"[align {step_i}] 未检测到 Tag，等待...")
                time.sleep(float(loop_s))
            continue

        cu = float(det["center_u"])
        cv_ = float(det["center_v"])
        last_seen_u, last_seen_v = cu, cv_
        last_seen_t = time.time()

        err_u = cu - tu
        err_v = cv_ - tv
        err0 = _err_norm(err_u, err_v)
        print(
            f"[align {step_i}] tag={det['tag_id']} tip_err=({err_u:+.1f},{err_v:+.1f}) "
            f"|e|={err0:.1f} approach_sign={approach_sign:+.0f}"
        )

        if abs(err_u) <= tol_u and abs(err_v) <= tol_v:
            stable += 1
            print(f"[align] 窗口内 stable={stable}/{stable_need}")
            if stable >= int(stable_need):
                print("[align] Tag 已对准夹爪尖端")
                return True
            time.sleep(float(loop_s))
            continue
        stable = 0

        ticks = arm.read()
        seed = {n: float(ticks[n]) for n in JOINT_ORDER}
        T_base_ee = fk.forward_ticks(seed)
        T_base_cam = T_base_ee @ T_ee_cam
        delta, lat, fwd = pixel_error_to_approach_xy(
            err_u,
            err_v,
            T_base_cam,
            m_per_px=m_per_px,
            max_step_m=max_step_m,
            approach_sign=approach_sign,
        )
        last_delta = delta.copy()
        print(
            f"[align] 光轴前移 Δxy=({delta[0]*1000:+.1f},{delta[1]*1000:+.1f}) mm "
            f"fwd=({fwd[0]:+.2f},{fwd[1]:+.2f}) lat=({lat[0]:+.2f},{lat[1]:+.2f}) "
            f"(保 Z={z_hold:.4f}, approach_sign={approach_sign:+.0f})"
        )

        prev_pose = snapshot_pose(arm)
        _, prev_xyz = ee_xyz_now(arm, fk)
        alpha = 1.0
        moved = False
        for attempt in range(1, max(1, int(step_retries)) + 1):
            step = alpha * delta
            print(
                f"[align] 前移 attempt={attempt}/{step_retries} α={alpha:.2f} "
                f"Δ=({step[0]*1000:+.1f},{step[1]*1000:+.1f}) mm"
            )
            try:
                _move_delta(step, f"align{step_i}a{attempt}")
            except Exception as exc:
                print(f"[warn] 移动失败: {exc}，回退")
                go_pose_strict(arm, f"align_rb_ik{step_i}a{attempt}", prev_pose)
                alpha *= 0.5
                continue

            _, now_xyz = ee_xyz_now(arm, fk)
            moved_mm = float(np.hypot(now_xyz[0] - prev_xyz[0], now_xyz[1] - prev_xyz[1])) * 1000.0
            print(f"[align] 实际水平位移 {moved_mm:.1f} mm")

            det2, tu2, tv2 = _detect(f"align{step_i}chk{attempt}", True)
            if det2 is None:
                print("[align] 移动后丢失视野 → 恢复上一步")
                go_pose_strict(arm, f"align_rb{step_i}a{attempt}", prev_pose)
                time.sleep(max(0.25, float(settle_s)))
                det_back, _, _ = _detect(f"align_rb{step_i}a{attempt}", True)
                if det_back is None:
                    time.sleep(0.35)
                    det_back, _, _ = _detect(f"align_rb{step_i}b{attempt}", True)
                if det_back is None:
                    print("[warn] 回退后暂无 Tag，反号后续续试（不立即失败）")
                    if not flipped:
                        approach_sign *= -1.0
                        flipped = True
                        print(f"[align] approach_sign → {approach_sign:+.0f}")
                    moved = True
                    break
                last_seen_u = float(det_back["center_u"])
                last_seen_v = float(det_back["center_v"])
                last_seen_t = time.time()
                if not flipped:
                    approach_sign *= -1.0
                    flipped = True
                    delta = -delta
                    last_delta = delta.copy()
                    print(f"[align] 丢标触发反号 approach_sign → {approach_sign:+.0f}")
                alpha *= 0.5
                if alpha < 0.08:
                    moved = True
                    break
                continue

            eu2 = float(det2["center_u"]) - tu2
            ev2 = float(det2["center_v"]) - tv2
            err1 = _err_norm(eu2, ev2)
            last_seen_u = float(det2["center_u"])
            last_seen_v = float(det2["center_v"])
            last_seen_t = time.time()
            print(f"[align] 移动后 |e| {err0:.1f} → {err1:.1f} (Δv {err_v:+.1f}→{ev2:+.1f})")

            # 几乎没动 / 误差未下降 / 发散 → 回退并反向前移方向
            no_progress = (moved_mm < 2.0) or (err1 >= err0 - 1.0)
            diverged = err1 > err0 * 1.05
            if diverged or no_progress:
                reason = "发散" if diverged and err1 > err0 else "几乎无改善"
                print(f"[align] {reason} → 回退并反向前移方向")
                go_pose_strict(arm, f"align_div{step_i}a{attempt}", prev_pose)
                time.sleep(float(settle_s))
                if not flipped:
                    approach_sign *= -1.0
                    flipped = True
                    delta, lat, fwd = pixel_error_to_approach_xy(
                        err_u,
                        err_v,
                        T_base_cam,
                        m_per_px=m_per_px,
                        max_step_m=max_step_m,
                        approach_sign=approach_sign,
                    )
                    last_delta = delta.copy()
                    print(f"[align] approach_sign → {approach_sign:+.0f}")
                else:
                    alpha *= 0.5
                if alpha < 0.08:
                    moved = True
                    break
                continue

            moved = True
            break

        if not moved:
            print("[align] 本步多次重试失败")
            return False

        time.sleep(float(loop_s))

    print("[align] 超时未对准")
    return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="放置位姿 → AprilTag 水平前移对齐夹爪尖端 → 开爪"
    )
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--og-config", type=Path, default=DEFAULT_OG)
    parser.add_argument("--vision-config", type=Path, default=DEFAULT_V3D)
    parser.add_argument("--port", default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--tag-id", type=int, default=None)
    parser.add_argument("--tag-family", default="tag36h11")
    parser.add_argument("--target-u", type=float, default=None, help="夹爪尖端 u（默认 320）")
    parser.add_argument("--target-v", type=float, default=None, help="夹爪尖端 v（默认 400）")
    parser.add_argument("--pixel-tol", type=float, default=None, help="到位容差(px)，覆盖 u/v")
    parser.add_argument("--tol-u", type=float, default=20.0)
    parser.add_argument("--tol-v", type=float, default=25.0)
    parser.add_argument("--m-per-px", type=float, default=0.00018, help="像素→米增益")
    parser.add_argument("--max-step-m", type=float, default=0.012, help="单次最大水平步长(m)")
    parser.add_argument(
        "--xy-sign",
        type=float,
        default=1.0,
        help="光轴前移方向：+1=Tag在上方时沿相机光轴前移；发散会自动翻转，也可手动 --xy-sign -1",
    )
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--settle-s", type=float, default=0.20)
    parser.add_argument("--loop-s", type=float, default=0.05)
    parser.add_argument("--lose-grace-s", type=float, default=1.0)
    parser.add_argument("--step-retries", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--ik-tol", type=float, default=0.012)
    parser.add_argument("--max-horiz-m", type=float, default=0.28)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--gripper-open", type=float, default=1200.0)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--no-save-image", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--release-on-exit", action="store_true")
    args = parser.parse_args()

    if not args.yes:
        print("Refusing motion: pass --yes")
        return 2

    cfg = merge_configs(args.og_config, args.vision_config)
    port = args.port or cfg.get("port", "/dev/ttyACM1")
    baud = int(cfg.get("baud") or DEFAULT_BAUD)
    motion = dict(cfg.get("motion") or {})
    limits = motion.get("joint_limits") or {}
    ag = dict(cfg.get("auto_grasp") or {})
    vision = dict(cfg.get("vision") or {})

    place = {j: float(PLACE_POSE[j]) for j in JOINTS}
    place_arm = {n: float(place[n]) for n in JOINT_ORDER}
    g_hold = float(place["gripper"])
    g_open = float(args.gripper_open)

    target_u = float(args.target_u) if args.target_u is not None else float(DEFAULT_GRIPPER_TIP_U)
    target_v = float(args.target_v) if args.target_v is not None else float(DEFAULT_GRIPPER_TIP_V)

    tol_u = float(args.pixel_tol) if args.pixel_tol is not None else float(args.tol_u)
    tol_v = float(args.pixel_tol) if args.pixel_tol is not None else float(args.tol_v)
    dead = ag.get("pixel_deadband") or vision.get("pixel_deadband")
    if args.pixel_tol is None and isinstance(dead, (list, tuple)) and len(dead) >= 2:
        if args.tol_u == 20.0:
            tol_u = float(dead[0])
        if args.tol_v == 25.0:
            tol_v = float(dead[1])

    stable_need = int(args.stable_frames or ag.get("stable_frames", 3))

    try:
        K, dist = load_intrinsics(
            Path(cfg.get("intrinsics", "output/camera_calib/camera_intrinsics.yaml"))
        )
        _ = K, dist  # 水平伺服用外参旋转即可；内参加载确认标定存在
    except Exception as exc:
        print(f"[warn] 内参加载失败（水平伺服仍可用手眼）: {exc}")

    try:
        T_ee_cam = load_handeye(Path(cfg.get("handeye", "output/handeye_ee_cam.yaml")))
    except FileNotFoundError as exc:
        print(f"[warn] {exc}；用手眼单位阵")
        T_ee_cam = np.eye(4)

    camera_cfg = dict(cfg.get("camera") or {})
    if args.camera is not None:
        camera_cfg["index_or_path"] = args.camera
    camera_cfg.setdefault("width", 640)
    camera_cfg.setdefault("height", 480)
    camera_cfg.setdefault("fps", 30)
    camera_cfg["settle_frames"] = 2
    camera_cfg.setdefault("frame_timeout_s", 5.0)

    save_dir = args.save_dir or Path("output/place_apriltag")
    if not save_dir.is_absolute():
        save_dir = (_TTA / save_dir).resolve()

    detector = AprilTagDetector(args.tag_family)
    fk = SO101FK.from_config(cfg) if isinstance(cfg.get("fk"), dict) else SO101FK()
    ik = SO101IK(fk)

    print_pose("放置位姿:", place)
    print(
        f"[info] tip_uv=({target_u:.0f},{target_v:.0f}) tol=({tol_u:.0f},{tol_v:.0f}) "
        f"m_per_px={args.m_per_px} max_step={args.max_step_m*1000:.0f}mm "
        f"approach_sign={args.xy_sign:+.0f} stable={stable_need} timeout={args.timeout_s:.0f}s"
    )

    arm = TicksArm(port, baud, motion)
    camera: Optional[LiveCamera] = None
    try:
        arm.connect()
        camera = LiveCamera(camera_cfg)

        print("\n=== 1) 进入放置位姿 ===")
        go_pose_strict(arm, "place", place)

        print("\n=== 2) AprilTag 水平前移对齐夹爪尖端 ===")
        ok = align_apriltag(
            arm=arm,
            fk=fk,
            ik=ik,
            detector=detector,
            camera=camera,
            T_ee_cam=T_ee_cam,
            limits=limits,
            g_hold=g_hold,
            place_arm=place_arm,
            tag_id=args.tag_id,
            target_u=target_u,
            target_v=target_v,
            tol_u=tol_u,
            tol_v=tol_v,
            m_per_px=float(args.m_per_px),
            max_step_m=float(args.max_step_m),
            stable_need=stable_need,
            timeout_s=float(args.timeout_s),
            settle_s=float(args.settle_s),
            loop_s=float(args.loop_s),
            lose_grace_s=float(args.lose_grace_s),
            step_retries=int(args.step_retries),
            ik_tol=float(args.ik_tol),
            max_horiz_m=float(args.max_horiz_m),
            allow_partial=bool(args.allow_partial),
            xy_sign=float(args.xy_sign),
            save_dir=save_dir,
            no_save=bool(args.no_save_image),
            save_every=int(args.save_every),
        )
        if not ok:
            print("[place] 对齐失败")
            hold_forever(arm, motion)
            return 1

        if not args.no_open:
            print(f"\n=== 3) 开爪放下 -> {int(g_open)} ===")
            set_gripper(arm, g_open, limits, 0.35)
        else:
            print("\n=== 3) --no-open，保持持块 ===")

        hold_forever(arm, motion)
        return 0
    except KeyboardInterrupt:
        print("\n用户中断")
        return 130
    finally:
        if camera is not None:
            camera.close()
        arm.disconnect(release_torque=bool(args.release_on_exit))
        if not args.release_on_exit:
            print("已断开串口；力矩未主动关闭")


if __name__ == "__main__":
    raise SystemExit(main())
