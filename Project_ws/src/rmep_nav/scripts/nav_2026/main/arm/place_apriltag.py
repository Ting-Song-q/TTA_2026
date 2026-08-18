#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 AprilTag 放置（衔接 reset_and_grasp / mobile_grasp 抓取结束状态）。

放置观察位 = place_apriltag.yaml 的 poses.place_observe（仅臂关节）
  + 夹爪保持进入脚本前的实际 ticks（不改写，避免持块过载）
若未配置 place_observe，则回退到 reset_and_grasp.yaml 的 poses.grasp。

流程：
  1. 移到观察位（夹爪不变）
  2. 单次检测 AprilTag → 水平粗接近（锁腕肘、保高度）
  3. 实时视觉伺服：持续取最新帧 → 小步修正（默认开启）
  4. （可选）最终前探
  5. 对准后竖直下降 → 开爪放下
  6. 机械臂回到 poses.initial 后退出（供导航继续任务）

用法:
  python3 reset_and_grasp.py --yes
  python3 place_apriltag.py --yes
  python3 place_apriltag.py --yes --place-descend-m 0.03
  python3 place_apriltag.py --yes --no-live-preview
  python3 place_apriltag.py --yes --no-fine-align

退出码:
  0  成功（含：最终仍无 Tag 时已开爪并回初始位）
  1  一般失败（连接/IK 等）
  2  观察位未检测到 AprilTag 且 --keep-hold-on-no-tag（导航可前移后重试）
  130 用户中断
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
    full_pose_from_ik,
    go_pose_strict,
    ik_route,
    merge_configs,
)
from grasp_from_observe import (  # noqa: E402
    build_forward_route_xyz,
    go_pose_if_changed,
    offset_grasp_left,
    pixel_error_to_base_delta,
)
from og import JOINTS, TicksArm, clamp_pose, pose_i, print_pose  # noqa: E402
from pixel_to_base import (  # noqa: E402
    grasp_z,
    load_handeye,
    load_intrinsics,
    pixel_to_base,
)
from so101_fk import JOINT_ORDER, SO101FK  # noqa: E402
from so101_ik import SO101IK  # noqa: E402
from so101_red_block_camera_test import camera_source  # noqa: E402
from start import DEFAULT_BAUD, move_to  # noqa: E402

# 与 reset_and_grasp 共用串口/抓取参数；放置观察位在 place_apriltag.yaml
DEFAULT_RAG = _HERE / "reset_and_grasp.yaml"
DEFAULT_PLACE = _HERE / "place_apriltag.yaml"

DETECT_MAX_WIDTH = 960


def open_gripper_and_return_initial(
    *,
    arm: TicksArm,
    poses: dict,
    observe_arm: Dict[str, float],
    limits: dict,
    g_open: float,
    g_hold: float,
    grasp_cfg: dict,
    open_gripper: bool = True,
) -> None:
    """开爪（可选）后回到 poses.initial，便于导航继续。"""
    if open_gripper:
        print(f"\n=== 开爪 -> {int(g_open)} ===")
        try:
            set_gripper(
                arm, g_open, limits, float(grasp_cfg.get("close_settle_s", 0.35))
            )
        except Exception as exc:
            print(f"[warn] 开爪写入失败，请手动确认夹爪: {exc}")

    print("\n=== 回到初始位姿 ===")
    initial = dict(poses.get("initial") or {})
    if not initial:
        print("[warn] 配置无 poses.initial，跳过臂复位")
        return
    home = {
        n: float(initial.get(n, observe_arm.get(n, 2048))) for n in JOINT_ORDER
    }
    if open_gripper:
        home["gripper"] = float(g_open)
    else:
        try:
            home["gripper"] = float(arm.read().get("gripper", g_hold))
        except Exception:
            home["gripper"] = float(g_hold)
    try:
        go_pose_strict(arm, "place_return_initial", home)
        print("[place] 已回到 poses.initial")
    except Exception as exc:
        print(f"[warn] 回初始位失败: {exc}")


def cam_forward_xy(T_base_ee: np.ndarray, T_ee_cam: np.ndarray) -> np.ndarray:
    """腕部相机光轴在水平面的投影（朝视野内 Tag 前进用）。"""
    T_base_cam = np.asarray(T_base_ee, dtype=float) @ np.asarray(T_ee_cam, dtype=float)
    cam_z = T_base_cam[:3, 2]
    fwd = np.array([float(cam_z[0]), float(cam_z[1])], dtype=float)
    n = float(np.linalg.norm(fwd))
    if n < 1e-4:
        # 光轴近竖直时，SO-101 常用 -Y 为前方
        return np.array([0.0, -1.0], dtype=float)
    return fwd / n


def _lock_place_joints(
    ticks: Dict[str, float],
    observe_arm: Dict[str, float],
    *,
    lock_wrist: bool,
    lock_elbow: bool,
    max_lift_delta_ticks: float,
) -> Dict[str, float]:
    """锁腕/肘，并限制 shoulder_lift 相对观察位的上探（避免靠下）。"""
    out = {n: float(ticks[n]) for n in JOINT_ORDER}
    obs_lift = float(observe_arm["shoulder_lift"])
    # 本机日志：增大 shoulder_lift → EE 下落；禁止相对观察位大幅抬升 lift
    out["shoulder_lift"] = float(
        np.clip(
            out["shoulder_lift"],
            obs_lift - float(max_lift_delta_ticks),
            obs_lift + float(max_lift_delta_ticks),
        )
    )
    if lock_elbow:
        out["elbow_flex"] = float(observe_arm["elbow_flex"])
    if lock_wrist:
        out["wrist_flex"] = float(observe_arm["wrist_flex"])
        out["wrist_roll"] = float(observe_arm["wrist_roll"])
    return out


def _score_place_sol(
    fk: SO101FK,
    ticks: Dict[str, float],
    target_xyz: np.ndarray,
    z_floor: float,
    observe_arm: Dict[str, float],
) -> Tuple[float, float, float]:
    """返回 (代价越小越好, pos_err, z_fk)。"""
    T = fk.forward_ticks({n: float(ticks[n]) for n in JOINT_ORDER})
    p = T[:3, 3]
    err = float(np.linalg.norm(p - target_xyz))
    z_fk = float(p[2])
    lift_pen = abs(float(ticks["shoulder_lift"]) - float(observe_arm["shoulder_lift"])) / 400.0
    z_pen = max(0.0, z_floor - z_fk) * 40.0
    cost = err + z_pen + 0.01 * lift_pen
    return cost, err, z_fk


def solve_place_waypoint(
    ik: SO101IK,
    fk: SO101FK,
    target_xyz: np.ndarray,
    seed: Dict[str, float],
    observe_arm: Dict[str, float],
    limits: dict,
    ik_tol: float,
    z_floor: float,
    *,
    lock_wrist: bool,
    lock_elbow: bool,
    max_lift_delta_ticks: float,
) -> Tuple[Optional[Dict[str, float]], float, float]:
    """放置单点：多种子 + 锁关节，拒绝明显掉 Z 的解。"""
    raised = dict(observe_arm)
    raised["shoulder_lift"] = float(
        np.clip(float(raised["shoulder_lift"]) - 120.0, 600.0, 3400.0)
    )
    # 轻微收肘抬高，而不是 stretch 下探
    raised["elbow_flex"] = float(
        np.clip(float(raised["elbow_flex"]) + 120.0, 600.0, 3400.0)
    )
    seeds = [seed, observe_arm, raised]
    best: Optional[Dict[str, float]] = None
    best_cost = 1e9
    best_err = 1e9
    best_z = -1e9

    xyz = np.asarray(target_xyz, dtype=float).copy()
    for z_try in (float(xyz[2]), float(xyz[2]), float(max(xyz[2] - 0.005, z_floor))):
        goal = xyz.copy()
        goal[2] = z_try
        for s in seeds:
            ticks0, _info = ik.solve_position(
                goal,
                s,
                joint_limits=limits,
                pos_tol_m=ik_tol,
                max_iters=100,
            )
            ticks = _lock_place_joints(
                ticks0,
                observe_arm,
                lock_wrist=lock_wrist,
                lock_elbow=lock_elbow,
                max_lift_delta_ticks=max_lift_delta_ticks,
            )
            # 锁关节后 XY 可能偏，再以锁后姿态为种子微调 pan（仍锁腕肘）
            ticks1, _ = ik.solve_position(
                goal,
                ticks,
                joint_limits=limits,
                pos_tol_m=ik_tol,
                max_iters=60,
            )
            ticks = _lock_place_joints(
                ticks1,
                observe_arm,
                lock_wrist=lock_wrist,
                lock_elbow=lock_elbow,
                max_lift_delta_ticks=max_lift_delta_ticks,
            )
            cost, err, z_fk = _score_place_sol(fk, ticks, goal, z_floor, observe_arm)
            if z_fk < z_floor - 1e-4:
                continue
            if cost < best_cost:
                best, best_cost, best_err, best_z = ticks, cost, err, z_fk

    if best is None:
        return None, 1e9, -1e9
    return best, best_err, best_z


def ik_route_place(
    ik: SO101IK,
    fk: SO101FK,
    route_xyz: List[np.ndarray],
    seed: Dict[str, float],
    observe_arm: Dict[str, float],
    z_hold: float,
    limits: dict,
    ik_tol: float,
    max_horiz_m: float,
    max_z_drop_m: float = 0.008,
    *,
    lock_wrist: bool = True,
    lock_elbow: bool = True,
    max_lift_delta_ticks: float = 50.0,
    stop_if_bad: bool = True,
) -> List[Dict[str, float]]:
    """放置路点：锁腕肘、限制 lift，掉 Z 则截断（宁短勿低）。"""
    if not route_xyz:
        raise RuntimeError("空放置路径")
    horiz = float(np.hypot(route_xyz[-1][0], route_xyz[-1][1]))
    print(
        f"[route] 放置路点数={len(route_xyz)} 终点水平半径={horiz*1000:.0f} mm "
        f"lock_wrist={lock_wrist} lock_elbow={lock_elbow} "
        f"max_lift_delta={max_lift_delta_ticks:.0f}"
    )
    if horiz > max_horiz_m:
        print("[warn] 终点偏远，可能超出工作空间")

    z_floor = float(z_hold) - float(max_z_drop_m)
    ticks_list: List[Dict[str, float]] = []
    cur_seed = dict(seed)
    accept = max(float(ik_tol), 0.025)

    for i, xyz0 in enumerate(route_xyz):
        xyz = np.asarray(xyz0, dtype=float).copy()
        xyz[2] = float(z_hold)
        ticks, err, z_fk = solve_place_waypoint(
            ik,
            fk,
            xyz,
            cur_seed,
            observe_arm,
            limits,
            ik_tol,
            z_floor,
            lock_wrist=lock_wrist,
            lock_elbow=lock_elbow,
            max_lift_delta_ticks=max_lift_delta_ticks,
        )
        if ticks is None:
            print(
                f"[ik] place_wp{i+1}/{len(route_xyz)} 无满足 Z>={z_floor:.3f} 的解 → 截断"
            )
            break
        print(
            f"[ik] place_wp{i+1}/{len(route_xyz)} "
            f"xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) "
            f"err={err*1000:.1f}mm z_fk={z_fk:.3f} "
            f"lift={ticks['shoulder_lift']:.0f} elbow={ticks['elbow_flex']:.0f}"
        )
        if err > accept and stop_if_bad:
            print(f"[ik] place_wp{i+1} 误差过大({err*1000:.1f}mm) → 截断，避免硬探")
            break
        ticks_list.append(ticks)
        cur_seed = ticks

    if not ticks_list:
        raise RuntimeError(
            "放置路径无可用点：当前观察位向前伸展会掉高度。"
            "请减小 forward_m，或把 place_observe 调得更高/更收一点"
        )
    if len(ticks_list) < len(route_xyz):
        print(
            f"[route] 仅执行 {len(ticks_list)}/{len(route_xyz)} 点 "
            f"（为保高度提前停止）"
        )
    return ticks_list


def plan_forward_distance(
    *,
    forward_m: float,
    horiz_vis: float,
    standoff_m: float,
    max_forward_m: float,
) -> float:
    """前伸距离：不超过 forward_m，且相对视觉 Tag 保留 standoff，避免贴太近。"""
    fwd = float(forward_m)
    if horiz_vis > 1e-3 and standoff_m > 0:
        # 视觉说 Tag 在 30cm 外时，不要按视觉全距去贴桌面
        vis_cap = max(0.0, float(horiz_vis) - float(standoff_m))
        if vis_cap > 1e-4:
            fwd = min(fwd, vis_cap)
    fwd = min(fwd, float(max_forward_m))
    return max(0.0, fwd)


def advance_along_forward_place(
    *,
    arm: TicksArm,
    fk: SO101FK,
    ik: SO101IK,
    limits: dict,
    g_hold: float,
    observe_arm: dict,
    forward_xy: np.ndarray,
    distance_m: float,
    route_points: int,
    ik_tol: float,
    max_horiz_m: float,
    max_z_drop_m: float,
    lock_wrist: bool,
    lock_elbow: bool,
    max_lift_delta_ticks: float,
    label: str,
) -> None:
    """沿记录的水平方向前探；掉 Z 则截断。"""
    distance = float(distance_m)
    direction = np.asarray(forward_xy, dtype=float).reshape(2)
    norm = float(np.linalg.norm(direction))
    if distance <= 0.0 or norm < 1e-6:
        return
    direction /= norm

    ticks = arm.read()
    seed = {n: float(ticks[n]) for n in JOINT_ORDER}
    T_base_ee = fk.forward_ticks(seed)
    cur = T_base_ee[:3, 3].copy()
    z_hold = float(cur[2])
    target = np.array(
        [
            cur[0] + direction[0] * distance,
            cur[1] + direction[1] * distance,
            z_hold,
        ],
        dtype=float,
    )
    print(
        f"[{label}] 沿记录方向前探 {distance * 1000:.0f} mm "
        f"→ ({target[0]:.4f},{target[1]:.4f},{target[2]:.4f})"
    )
    route = build_forward_route_xyz(cur, target, route_points)
    try:
        route_ticks = ik_route_place(
            ik,
            fk,
            route,
            seed,
            observe_arm,
            z_hold,
            limits,
            float(ik_tol),
            float(max_horiz_m),
            float(max_z_drop_m),
            lock_wrist=lock_wrist,
            lock_elbow=lock_elbow,
            max_lift_delta_ticks=max_lift_delta_ticks,
            stop_if_bad=True,
        )
    except RuntimeError as exc:
        print(f"[warn] {label} 跳过: {exc}")
        return
    for index, arm_ticks in enumerate(route_ticks):
        pose = full_pose_from_ik(arm_ticks, g_hold, ticks)
        go_pose_if_changed(arm, f"{label}_wp{index + 1}", pose)
        time.sleep(0.01)


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
        # 尽量减小缓冲，便于实时取最新帧
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        for _ in range(max(1, int(camera_cfg.get("settle_frames", 1)))):
            self.cap.read()

    def grab(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("wrist camera returned empty frame")
        return frame

    def grab_latest(self, discard: int = 2) -> np.ndarray:
        """丢掉缓冲旧帧，返回最新一帧（实时伺服用）。"""
        frame = None
        for _ in range(max(1, int(discard))):
            frame = self.grab()
        assert frame is not None
        return frame

    def close(self) -> None:
        self.cap.release()


def set_gripper(arm: TicksArm, value: float, limits: dict, settle_s: float = 0.25) -> None:
    pose = dict(arm.goal) if arm.goal else arm.read()
    pose["gripper"] = float(value)
    limited = clamp_pose(pose, limits)
    try:
        move_to(arm.bus, pose_i(limited), wait_s=0.0)
        arm.goal = dict(limited)
    except Exception as exc:
        print(f"[warn] 夹爪写入失败 ({value:.0f}): {exc}")
        try:
            actual = arm.read()
            limited["gripper"] = float(actual.get("gripper", value))
            arm.goal = dict(limited)
        except Exception:
            arm.goal = dict(limited)
        raise
    time.sleep(settle_s)


def resolve_hold_gripper(g_now: float, g_close: float, g_open: float) -> float:
    """持块时禁止再往更紧方向硬顶（易 Overload id=6）。

    SO-101：gripper_close < gripper_open（如 800 < 1200）。
    已在闭爪侧则保持当前实际值。
    """
    mid = 0.5 * (float(g_open) + float(g_close))
    g = float(g_now)
    if g <= mid + 80.0:
        return g
    return float(g_close)


def hold_forever_safe(arm: TicksArm, motion: dict) -> None:
    """保持位姿；夹爪过载时停止硬写 Goal，提示断电复位。"""
    print("[hold] 保持设定 ticks；Ctrl+C 退出")
    hz = float(motion.get("hold_hz", 10.0))
    period = 1.0 / max(hz, 0.5)
    overload_strikes = 0
    while True:
        try:
            arm.hold_tick()
            overload_strikes = 0
        except Exception as exc:
            msg = str(exc)
            if "Overload" in msg or "id_=6" in msg or "gripper" in msg.lower():
                overload_strikes += 1
                print(f"[warn] 夹爪/总线过载: {exc}")
                try:
                    actual = arm.read()
                    if arm.goal:
                        # 不再命令更紧的闭爪
                        arm.goal["gripper"] = float(
                            actual.get("gripper", arm.goal.get("gripper", 1000))
                        )
                        print(
                            f"[hold] 改用实际爪值 {arm.goal['gripper']:.0f}，"
                            "停止硬顶 gripper_close"
                        )
                except Exception:
                    pass
                if overload_strikes >= 2:
                    print(
                        "[hold] 夹爪舵机可能已保护掉线 (id=6)。\n"
                        "  请：1) Ctrl+C 退出  2) 给机械臂断电再上电  "
                        "3) 确认 6 号舵机灯正常后重跑"
                    )
                    # 不再高频写总线，只空转等待用户中断
                    while True:
                        time.sleep(1.0)
            else:
                raise
        time.sleep(period)


def snapshot_pose(arm: TicksArm) -> Dict[str, float]:
    src = arm.goal if arm.goal else arm.read()
    return {k: float(v) for k, v in src.items()}


def gripper_tip_uv(
    frame_shape: tuple,
    tip_u: Optional[float],
    tip_v: Optional[float],
) -> Tuple[float, float]:
    """夹爪对准点；未指定时与 grasp_from_observe 一致：0.5w / 0.55h。"""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    tu = float(tip_u) if tip_u is not None else 0.5 * w
    tv = float(tip_v) if tip_v is not None else 0.55 * h
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
    max_z_drop_m: float = 0.008,
    lock_wrist: bool = True,
    lock_elbow: bool = True,
    max_lift_delta_ticks: float = 40.0,
) -> None:
    """水平面移动到 target（保持 Z）；末点用 go_pose_strict 确保真正到位。"""
    ticks_now = arm.read()
    seed = {n: float(ticks_now[n]) for n in JOINT_ORDER}
    cur = fk.forward_ticks(seed)[:3, 3].copy()
    goal = np.asarray(target_xyz, dtype=float).copy()
    goal[2] = float(cur[2])
    horiz = float(np.hypot(goal[0] - cur[0], goal[1] - cur[1]))
    if horiz < 1e-4:
        print(f"[align] skip tiny move {horiz*1000:.2f}mm")
        return

    route = build_forward_route_xyz(cur, goal, max(2, int(n_points)))
    try:
        route_ticks = ik_route_place(
            ik,
            fk,
            route,
            seed,
            place_arm,
            float(cur[2]),
            limits,
            float(ik_tol),
            float(max_horiz_m),
            float(max_z_drop_m),
            lock_wrist=lock_wrist,
            lock_elbow=lock_elbow,
            max_lift_delta_ticks=max_lift_delta_ticks,
            stop_if_bad=True,
        )
    except RuntimeError as exc:
        print(f"[align] 跳过本步水平移动: {exc}")
        return
    del allow_partial
    for j, arm_ticks in enumerate(route_ticks):
        pose = full_pose_from_ik(arm_ticks, g_hold, ticks_now)
        if j < len(route_ticks) - 1:
            limited = clamp_pose(pose, limits)
            move_to(arm.bus, pose_i(limited), wait_s=0.0)
            arm.goal = dict(limited)
            time.sleep(0.03)
        else:
            go_pose_strict(arm, name, pose)


def approach_place_like_grasp(
    *,
    arm: TicksArm,
    fk: SO101FK,
    ik: SO101IK,
    detector: AprilTagDetector,
    camera: LiveCamera,
    T_ee_cam: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    table_z: float,
    table_normal: np.ndarray,
    grasp_z_m: float,
    observe_arm: Dict[str, float],
    limits: dict,
    g_hold: float,
    tag_id: Optional[int],
    range_mode: str,
    forward_m: float,
    left_m: float,
    z_offset: float,
    route_points: int,
    ik_tol: float,
    max_horiz_m: float,
    allow_partial: bool,
    max_z_drop_m: float,
    standoff_m: float,
    lock_wrist: bool,
    lock_elbow: bool,
    max_lift_delta_ticks: float,
    save_dir: Path,
    no_save: bool,
) -> Tuple[str, Optional[np.ndarray]]:
    """单次识别后水平前伸；锁腕肘、限 lift，掉高则缩短/截断（宁短勿低）。

    返回 (status, forward_xy)：
      status = "ok" | "no_tag" | "ik_fail"
    """
    del allow_partial
    det, _, _, _ = detect_and_save(
        detector=detector,
        camera=camera,
        tag_id=tag_id,
        target_u=None,
        target_v=None,
        tol_u=20.0,
        tol_v=25.0,
        save_dir=save_dir,
        no_save=no_save,
        tag="place_det0",
        flush=3,
    )
    if det is None:
        print("[place] 观察位未检测到 AprilTag")
        return "no_tag", None

    cu = float(det["center_u"])
    cv_ = float(det["center_v"])
    print(f"[place] Tag id={det['tag_id']} center=({cu:.1f},{cv_:.1f})")

    ticks_now = arm.read()
    seed = {n: float(ticks_now[n]) for n in JOINT_ORDER}
    T_base_ee = fk.forward_ticks(seed)
    cur_xyz = T_base_ee[:3, 3].copy()
    # 不强行抬高：当前观察位向前时抬高不可达，只会逼 IK 增大 lift 下探
    z_hold = float(cur_xyz[2]) + float(z_offset)
    print(f"[fk] ee_xyz=({cur_xyz[0]:.4f},{cur_xyz[1]:.4f},{cur_xyz[2]:.4f}) m")

    hit = pixel_to_base(
        cu,
        cv_,
        T_base_ee,
        T_ee_cam,
        K,
        dist,
        float(table_z),
        np.asarray(table_normal, dtype=float),
        float(grasp_z_m),
    )
    p_vis = np.asarray(hit["p_grasp"], dtype=float).reshape(3)
    horiz_vis = float(np.hypot(p_vis[0] - cur_xyz[0], p_vis[1] - cur_xyz[1]))
    print(
        f"[3d] 桌面交点 XY=({hit['p_table'][0]:.4f},{hit['p_table'][1]:.4f}) "
        f"水平距={horiz_vis*1000:.0f} mm"
    )

    cam_dir = cam_forward_xy(T_base_ee, T_ee_cam)
    print(f"[3d] 相机光轴水平方向=({cam_dir[0]:+.3f},{cam_dir[1]:+.3f})")

    if horiz_vis >= 0.025:
        d_xy = p_vis[:2] - cur_xyz[:2]
        d_xy = d_xy / float(np.linalg.norm(d_xy))
        dir_src = "vision"
    else:
        d_xy = cam_dir
        dir_src = "camera_axis"
        print("[3d] 视觉水平距过小，改用相机光轴方向靠近 Tag")

    if float(np.dot(d_xy, cam_dir)) < 0.15 and horiz_vis >= 0.025:
        print(
            f"[3d] 视觉方向与光轴夹角过大(dot={float(np.dot(d_xy, cam_dir)):.2f})，"
            "改用相机光轴靠近 Tag"
        )
        d_xy = cam_dir
        dir_src = "camera_axis"

    if range_mode == "forward":
        fwd = plan_forward_distance(
            forward_m=float(forward_m),
            horiz_vis=horiz_vis,
            standoff_m=float(standoff_m),
            max_forward_m=float(forward_m),
        )
    else:
        fwd = plan_forward_distance(
            forward_m=horiz_vis,
            horiz_vis=horiz_vis,
            standoff_m=float(standoff_m),
            max_forward_m=min(float(forward_m) * 2.0, float(max_horiz_m)),
        )
    p = np.array(
        [
            float(cur_xyz[0]) + fwd * float(d_xy[0]),
            float(cur_xyz[1]) + fwd * float(d_xy[1]),
            z_hold,
        ],
        dtype=float,
    )
    print(
        f"[3d] 前方锚定 dir={dir_src} mode={range_mode} "
        f"forward={fwd*1000:.0f} mm (cfg={forward_m*1000:.0f}, "
        f"standoff={standoff_m*1000:.0f}, 视觉 {horiz_vis*1000:.0f}) "
        f"→ XY=({p[0]:.4f},{p[1]:.4f})"
    )

    p_center = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=float)
    p = offset_grasp_left(cur_xyz, p_center, float(left_m))
    if abs(float(left_m)) >= 1e-9:
        print(
            f"[3d] 放置点左偏 {left_m*1000:.1f} mm: "
            f"({p_center[0]:.4f},{p_center[1]:.4f}) → ({p[0]:.4f},{p[1]:.4f})"
        )

    target = np.array([float(p[0]), float(p[1]), z_hold], dtype=float)
    forward_xy = target[:2] - cur_xyz[:2]
    horiz_plan = float(np.linalg.norm(forward_xy))
    print(
        f"[plan] 放置前伸目标 ee=({target[0]:.4f},{target[1]:.4f},{target[2]:.4f}) "
        f"水平={horiz_plan*1000:.0f}mm Z=观察位{float(z_offset):+.3f}m "
        f"route_points={route_points}"
    )

    last_err: Optional[BaseException] = None
    for scale in (1.0, 0.75, 0.5, 0.35):
        tgt = cur_xyz.copy()
        tgt[0] = float(cur_xyz[0]) + float(forward_xy[0]) * scale
        tgt[1] = float(cur_xyz[1]) + float(forward_xy[1]) * scale
        tgt[2] = z_hold
        route_xyz = build_forward_route_xyz(cur_xyz, tgt, int(route_points))
        try:
            route_ticks = ik_route_place(
                ik,
                fk,
                route_xyz,
                seed,
                observe_arm,
                z_hold,
                limits,
                float(ik_tol),
                float(max_horiz_m),
                float(max_z_drop_m),
                lock_wrist=lock_wrist,
                lock_elbow=lock_elbow,
                max_lift_delta_ticks=max_lift_delta_ticks,
                stop_if_bad=True,
            )
        except RuntimeError as exc:
            last_err = exc
            print(f"[plan] scale={scale:.2f} 失败: {exc}")
            continue
        if scale < 0.999:
            print(f"[plan] 采用缩短前伸 scale={scale:.2f} 以保持高度")
        for i, arm_ticks in enumerate(route_ticks):
            pose = full_pose_from_ik(arm_ticks, g_hold, ticks_now)
            go_pose_if_changed(arm, f"place_wp{i+1}", pose)
            time.sleep(0.01)
        print("[place] 粗接近完成（锁腕肘，保高度）")
        return "ok", forward_xy * float(scale)

    print(f"[error] IK 路径失败: {last_err}")
    return "ik_fail", None


# ---------------------------------------------------------------------------
# 视觉伺服对齐：水平前移（参考 grasp_from_observe 闭爪前微调）
# ---------------------------------------------------------------------------

def _gui_available() -> bool:
    """当前 OpenCV 是否支持 imshow（板子上常见无 GUI 的 headless 构建）。"""
    try:
        # 空图探测；失败则说明 highgui 未编译
        cv2.imshow("_place_gui_probe_", np.zeros((8, 8, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyWindow("_place_gui_probe_")
        return True
    except Exception:
        return False


def detect_live(
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
    do_save: bool,
    show: bool,
) -> Tuple[Optional[dict], np.ndarray, float, float, bool]:
    """实时取最新帧并检测；可选存图 / 预览窗口。

    返回 (det, frame, tu, tv, show_ok)；show_ok=False 表示应关闭预览。
    """
    frame = camera.grab_latest(discard=2)
    tu, tv = gripper_tip_uv(frame.shape, target_u, target_v)
    det = detector.detect(frame, tag_id=tag_id)
    if do_save and not no_save:
        save_debug(save_dir, frame, det, tu, tv, tol_u, tol_v, tag)
    show_ok = True
    if show:
        vis = frame.copy()
        cv2.drawMarker(vis, (int(tu), int(tv)), (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.rectangle(
            vis,
            (int(tu - tol_u), int(tv - tol_v)),
            (int(tu + tol_u), int(tv + tol_v)),
            (0, 255, 0),
            2,
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
                f"id={det['tag_id']} tip_err="
                f"({det['center_u']-tu:+.0f},{det['center_v']-tv:+.0f})",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (50, 220, 50),
                2,
            )
        else:
            cv2.putText(
                vis, "no tag", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2
            )
        try:
            cv2.imshow("place_apriltag_live", vis)
            cv2.waitKey(1)
        except Exception as exc:
            print(f"[live] 预览不可用（无 GUI OpenCV），自动关闭窗口: {exc}")
            show_ok = False
    return det, frame, tu, tv, show_ok


def _servo_step_xy(
    *,
    arm: TicksArm,
    fk: SO101FK,
    ik: SO101IK,
    observe_arm: Dict[str, float],
    limits: dict,
    g_hold: float,
    delta_xy: np.ndarray,
    z_hold: float,
    ik_tol: float,
    max_horiz_m: float,
    name: str,
) -> Tuple[bool, float]:
    """实时伺服单步：解一次 IK 后快速写出（不做长平滑）。"""
    ticks = arm.read()
    seed = {n: float(ticks[n]) for n in JOINT_ORDER}
    cur = fk.forward_ticks(seed)[:3, 3].copy()
    target = np.array(
        [cur[0] + float(delta_xy[0]), cur[1] + float(delta_xy[1]), float(z_hold)],
        dtype=float,
    )
    try:
        route_ticks = ik_route(
            ik,
            [target],
            seed,
            observe_arm,
            cur,
            limits,
            float(ik_tol),
            False,
            float(max_horiz_m),
        )
    except RuntimeError as exc:
        print(f"[live] IK 跳过: {exc}")
        return False, 0.0
    pose = full_pose_from_ik(route_ticks[-1], g_hold, ticks)
    limited = clamp_pose(pose, limits)
    move_to(arm.bus, pose_i(limited), wait_s=0.0)
    arm.goal = dict(limited)
    arm.current_name = name
    _, now = ee_xyz_now(arm, fk)
    moved_mm = float(np.hypot(now[0] - cur[0], now[1] - cur[1])) * 1000.0
    return True, moved_mm


def refine_pose_until_tag_aligned(
    *,
    arm: TicksArm,
    fk: SO101FK,
    ik: SO101IK,
    detector: AprilTagDetector,
    camera: LiveCamera,
    T_ee_cam: np.ndarray,
    limits: dict,
    g_hold: float,
    observe_arm: Dict[str, float],
    tag_id: Optional[int],
    target_u: Optional[float],
    target_v: Optional[float],
    tol_u: float,
    tol_v: float,
    m_per_px: float,
    max_step_m: float,
    max_iters: int,
    settle_s: float,
    ik_tol: float,
    max_horiz_m: float,
    max_z_drop_m: float,
    save_dir: Path,
    no_save: bool,
    approach_sign: float = -1.0,
    loop_s: float = 0.05,
    timeout_s: float = 45.0,
    stable_need: int = 3,
    live_preview: bool = True,
    save_every: int = 5,
) -> bool:
    """实时视觉伺服：持续取最新帧 → 算误差 → 小步修正（不再「动完再拍」）。

    settle_s 仅用于到位后的短确认；循环内用 loop_s 控制取帧频率。
    """
    del settle_s  # 实时模式不用长停稳
    _, place_xyz = ee_xyz_now(arm, fk)
    z_hold = float(place_xyz[2])
    sign = float(approach_sign)
    if abs(sign) < 1e-6:
        sign = -1.0
    sign = 1.0 if sign > 0 else -1.0
    flipped = False
    stable = 0
    step_i = 0
    deadline = time.monotonic() + float(timeout_s)
    max_cycles = max(1, int(max_iters) * 20)  # 实时循环远多于旧的「停稳迭代」
    if live_preview and not _gui_available():
        print("[live] 当前 OpenCV 无 GUI，自动 --no-live-preview（仍实时伺服）")
        live_preview = False
    print(
        f"[live] 实时视觉引导 z_hold={z_hold:.4f}m "
        f"tol=({tol_u:.0f},{tol_v:.0f}) max_step={max_step_m*1000:.0f}mm "
        f"loop={loop_s*1000:.0f}ms timeout={timeout_s:.0f}s "
        f"stable_need={stable_need} approach_sign={sign:+.0f} preview={live_preview}"
    )

    try:
        while time.monotonic() < deadline and step_i < max_cycles:
            arm.hold_tick()
            step_i += 1
            do_save = (step_i % max(1, int(save_every)) == 0) or (step_i <= 2)
            det, frame, tu, tv, show_ok = detect_live(
                detector=detector,
                camera=camera,
                tag_id=tag_id,
                target_u=target_u,
                target_v=target_v,
                tol_u=tol_u,
                tol_v=tol_v,
                save_dir=save_dir,
                no_save=no_save,
                tag=f"live{step_i}",
                do_save=do_save,
                show=live_preview,
            )
            if live_preview and not show_ok:
                live_preview = False

            ticks = arm.read()
            seed = {n: float(ticks[n]) for n in JOINT_ORDER}
            T_base_ee = fk.forward_ticks(seed)
            cur = T_base_ee[:3, 3].copy()

            if det is None:
                stable = 0
                if step_i % 10 == 1:
                    print(f"[live {step_i}] 未检测到 Tag，等待...")
                time.sleep(float(loop_s))
                continue

            du = float(det["center_u"]) - tu
            dv = float(det["center_v"]) - tv
            err0 = float(np.hypot(du, dv))
            in_win = abs(du) <= float(tol_u) and abs(dv) <= float(tol_v)
            if step_i % 3 == 1 or in_win:
                print(
                    f"[live {step_i}] tag={det['tag_id']} "
                    f"du={du:+.1f} dv={dv:+.1f} |e|={err0:.1f} "
                    f"{'窗口内' if in_win else '伺服中'} sign={sign:+.0f}"
                )

            if in_win:
                stable += 1
                if stable >= int(stable_need):
                    print(f"[live] 连续 {stable} 帧在窗口内 → 对准完成")
                    return True
                time.sleep(float(loop_s))
                continue
            stable = 0

            T_base_cam = T_base_ee @ T_ee_cam
            delta, lat, fwd = pixel_error_to_approach_xy(
                du,
                dv,
                T_base_cam,
                m_per_px=m_per_px,
                max_step_m=max_step_m,
                approach_sign=sign,
            )
            n = float(np.linalg.norm(delta[:2]))
            min_step = min(float(max_step_m), 0.003)
            if 1e-6 < n < min_step:
                delta[:2] *= min_step / n

            prev_pose = snapshot_pose(arm)
            ok_move, moved_mm = _servo_step_xy(
                arm=arm,
                fk=fk,
                ik=ik,
                observe_arm=observe_arm,
                limits=limits,
                g_hold=g_hold,
                delta_xy=delta,
                z_hold=z_hold,
                ik_tol=ik_tol,
                max_horiz_m=max_horiz_m,
                name=f"live{step_i}",
            )
            if not ok_move:
                time.sleep(float(loop_s))
                continue

            _, now_xyz = ee_xyz_now(arm, fk)
            z_drop = float(z_hold) - float(now_xyz[2])
            if z_drop > float(max_z_drop_m):
                # 放置时水平伺服常伴随自然下探；已下降则保留姿态并进入开爪，不再回退空转
                print(
                    f"[live] 高度已下降 {z_drop*1000:.1f} mm "
                    f"(限 {max_z_drop_m*1000:.0f} mm) → 结束引导，进入下降/开爪"
                )
                return True

            # 立刻再取一帧看误差是否下降（不等长 settle）
            det2, _, tu2, tv2, show_ok2 = detect_live(
                detector=detector,
                camera=camera,
                tag_id=tag_id,
                target_u=target_u,
                target_v=target_v,
                tol_u=tol_u,
                tol_v=tol_v,
                save_dir=save_dir,
                no_save=True,
                tag=f"live{step_i}chk",
                do_save=False,
                show=live_preview,
            )
            if live_preview and not show_ok2:
                live_preview = False
            if det2 is not None:
                du2 = float(det2["center_u"]) - tu2
                dv2 = float(det2["center_v"]) - tv2
                err1 = float(np.hypot(du2, dv2))
                if step_i % 3 == 1:
                    print(
                        f"[live] Δxy=({delta[0]*1000:+.1f},{delta[1]*1000:+.1f})mm "
                        f"moved={moved_mm:.1f}mm |e| {err0:.1f}→{err1:.1f}"
                    )
                if (moved_mm < 1.0 or err1 > err0 * 1.08) and not flipped:
                    sign *= -1.0
                    flipped = True
                    print(f"[live] 误差未降 → approach_sign → {sign:+.0f}")
                    go_pose_strict(arm, f"live{step_i}_rb", prev_pose)
            elif moved_mm < 1.0 and not flipped:
                sign *= -1.0
                flipped = True
                print(f"[live] 丢标/几乎未动 → approach_sign → {sign:+.0f}")

            time.sleep(float(loop_s))
    finally:
        if live_preview:
            try:
                cv2.destroyWindow("place_apriltag_live")
            except Exception:
                pass

    print("[live] 超时/达到最大循环仍未对准")
    return False



def descend_ee_hold_xy(
    *,
    arm: TicksArm,
    fk: SO101FK,
    ik: SO101IK,
    observe_arm: Dict[str, float],
    limits: dict,
    g_hold: float,
    descend_m: float,
    route_points: int,
    ik_tol: float,
    max_horiz_m: float,
    label: str = "place_descend",
) -> bool:
    """对准后竖直下降（保持 XY），再开爪。descend_m>0 表示向下米数。"""
    dist = abs(float(descend_m))
    if dist < 1e-4:
        return True
    ticks = arm.read()
    seed = {n: float(ticks[n]) for n in JOINT_ORDER}
    cur = fk.forward_ticks(seed)[:3, 3].copy()
    target = cur.copy()
    target[2] = float(cur[2]) - dist
    print(
        f"[{label}] 下降 {dist*1000:.0f} mm "
        f"z {cur[2]:.4f} → {target[2]:.4f} (保 XY)"
    )
    n = max(2, int(route_points))
    route: List[np.ndarray] = []
    for i in range(1, n + 1):
        a = i / float(n)
        route.append(cur + a * (target - cur))
    try:
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
    except RuntimeError as exc:
        print(f"[warn] {label} IK 失败: {exc}")
        return False
    for j, arm_ticks in enumerate(route_ticks):
        pose = full_pose_from_ik(arm_ticks, g_hold, ticks)
        go_pose_if_changed(arm, f"{label}_wp{j+1}", pose)
        time.sleep(0.02)
    _, now = ee_xyz_now(arm, fk)
    print(
        f"[{label}] 完成 ee=({now[0]:.4f},{now[1]:.4f},{now[2]:.4f}) "
        f"Δz={(now[2]-cur[2])*1000:+.1f} mm"
    )
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="观察位持块 → 视觉规划前伸放 AprilTag → 开爪"
    )
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--og-config",
        type=Path,
        default=DEFAULT_RAG,
        help="默认 reset_and_grasp.yaml（串口/抓取/场景参数）",
    )
    parser.add_argument(
        "--vision-config",
        type=Path,
        default=DEFAULT_PLACE,
        help="默认 place_apriltag.yaml（poses.place_observe 等放置覆盖项）",
    )
    parser.add_argument("--port", default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--tag-id", type=int, default=None)
    parser.add_argument("--tag-family", default="tag36h11")
    # 与 grasp 相同的路径规划参数（可用 CLI 覆盖 yaml grasp:）
    parser.add_argument("--range-mode", choices=("forward", "vision"), default=None)
    parser.add_argument("--forward-m", type=float, default=None)
    parser.add_argument("--left-m", type=float, default=None)
    parser.add_argument("--z-offset", type=float, default=None)
    parser.add_argument("--route-points", type=int, default=None)
    parser.add_argument("--final-forward-probe-m", type=float, default=None)
    parser.add_argument("--final-forward-route-points", type=int, default=None)
    parser.add_argument(
        "--place-descend-m",
        type=float,
        default=None,
        help="对准完成后竖直下降距离(米)，再开爪；0 关闭",
    )
    parser.add_argument(
        "--place-descend-route-points",
        type=int,
        default=None,
        help="下降路点数",
    )
    parser.add_argument(
        "--skip-observe-move",
        action="store_true",
        help="已在放置观察位时跳过整臂运动（仍同步闭爪）",
    )
    parser.add_argument(
        "--no-fine-align",
        action="store_true",
        help="跳过粗接近后的 preclose 视觉引导（同 grasp_from_observe --no-preclose-check）",
    )
    parser.add_argument(
        "--fine-align",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--preclose-iters", type=int, default=None)
    parser.add_argument("--preclose-tol-u", type=float, default=None)
    parser.add_argument("--preclose-tol-v", type=float, default=None)
    parser.add_argument("--preclose-target-u", type=float, default=None)
    parser.add_argument("--preclose-target-v", type=float, default=None)
    parser.add_argument("--preclose-m-per-px", type=float, default=None)
    parser.add_argument("--preclose-max-step", type=float, default=None)
    parser.add_argument("--preclose-settle-s", type=float, default=None)
    parser.add_argument(
        "--loop-s",
        type=float,
        default=None,
        help="实时伺服取帧周期(秒)，默认 0.05",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="实时视觉引导超时(秒)",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=None,
        help="连续在窗口内多少帧算到位",
    )
    parser.add_argument(
        "--no-live-preview",
        action="store_true",
        help="关闭 OpenCV 实时预览窗口",
    )
    parser.add_argument("--save-every", type=int, default=None, help="每隔多少循环存一张调试图")
    parser.add_argument("--target-u", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target-v", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pixel-tol", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tol-u", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tol-v", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--m-per-px", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-step-m", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--xy-sign",
        type=float,
        default=-1.0,
        help="光轴前移符号：放置默认 -1（Tag 在 tip 下方时前移）；发散会自动翻转",
    )
    parser.add_argument("--settle-s", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lose-grace-s", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--step-retries", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--ik-tol", type=float, default=None)
    parser.add_argument("--max-horiz-m", type=float, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--gripper-open", type=float, default=None)
    parser.add_argument("--gripper-close", type=float, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--no-save-image", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument(
        "--keep-hold-on-no-tag",
        action="store_true",
        help="未检测到 Tag 时保持持块并 exit=2（供导航前移重试）；"
        "不加此参数则开爪、回初始位后 exit=0，便于继续任务",
    )
    parser.add_argument(
        "--hold-after",
        action="store_true",
        help="开爪/复位后保持力矩不退出（调试用；导航任务不要加）",
    )
    parser.add_argument("--release-on-exit", action="store_true")
    args = parser.parse_args()

    if not args.yes:
        print("Refusing motion: pass --yes")
        return 2

    cfg = merge_configs(args.og_config, args.vision_config)
    grasp_cfg = dict(cfg.get("grasp") or {})
    poses = dict(cfg.get("poses") or {})
    # 放置专用观察位（新参数）；未配置时回退 poses.grasp
    place_observe = dict(poses.get("place_observe") or {})
    grasp_observe = dict(poses.get("grasp") or {})
    observe_pose = place_observe or grasp_observe
    observe_pose_src = "poses.place_observe" if place_observe else "poses.grasp"
    if not observe_pose:
        print(
            "[error] 配置缺少 poses.place_observe（place_apriltag.yaml）"
            " 且无 poses.grasp 可回退"
        )
        return 2

    port = args.port or cfg.get("port", "/dev/ttyACM1")
    baud = int(cfg.get("baud") or DEFAULT_BAUD)
    motion = dict(cfg.get("motion") or {})
    limits = motion.get("joint_limits") or {}
    scene = dict(cfg.get("scene") or {})

    # 开爪/闭爪标定值仅用于最终放下与空爪判定；观察位不改写夹爪
    g_close = float(
        args.gripper_close
        if args.gripper_close is not None
        else grasp_cfg.get(
            "gripper_close",
            observe_pose.get("gripper", grasp_observe.get("gripper", 800)),
        )
    )
    g_open = float(
        args.gripper_open
        if args.gripper_open is not None
        else grasp_cfg.get("gripper_open", 1200)
    )
    # place_observe 只取臂关节；忽略 yaml 里的 gripper，避免持块时被改写
    observe_arm = {
        n: float(observe_pose.get(n, grasp_observe.get(n, 2048)))
        for n in JOINT_ORDER
    }
    observe_full = dict(observe_arm)  # 不含 gripper；进入后补实际爪值
    observe_hold = dict(observe_arm)

    range_mode = str(args.range_mode or grasp_cfg.get("range_mode", "forward"))
    forward_m = float(
        args.forward_m if args.forward_m is not None else grasp_cfg.get("forward_m", 0.04)
    )
    left_m = float(args.left_m if args.left_m is not None else grasp_cfg.get("left_m", 0.0))
    z_offset = float(
        args.z_offset if args.z_offset is not None else grasp_cfg.get("z_offset_m", 0.0)
    )
    max_z_drop_m = float(grasp_cfg.get("place_max_z_drop_m", 0.008))
    standoff_m = float(grasp_cfg.get("place_standoff_m", 0.22))
    lock_wrist = bool(grasp_cfg.get("place_lock_wrist", True))
    lock_elbow = bool(grasp_cfg.get("place_lock_elbow", True))
    max_lift_delta_ticks = float(grasp_cfg.get("place_max_lift_delta_ticks", 40))
    route_points = int(
        args.route_points
        if args.route_points is not None
        else grasp_cfg.get("route_points", 3)
    )
    final_probe = float(
        args.final_forward_probe_m
        if args.final_forward_probe_m is not None
        else grasp_cfg.get("final_forward_probe_m", 0.0)
    )
    final_route_pts = int(
        args.final_forward_route_points
        if args.final_forward_route_points is not None
        else grasp_cfg.get("final_forward_route_points", 3)
    )
    place_descend_m = float(
        args.place_descend_m
        if args.place_descend_m is not None
        else grasp_cfg.get("place_descend_m", 0.03)
    )
    place_descend_pts = int(
        args.place_descend_route_points
        if args.place_descend_route_points is not None
        else grasp_cfg.get("place_descend_route_points", 4)
    )
    ik_tol = float(
        args.ik_tol if args.ik_tol is not None else grasp_cfg.get("ik_tol_m", 0.012)
    )
    max_horiz_m = float(
        args.max_horiz_m
        if args.max_horiz_m is not None
        else grasp_cfg.get("max_horiz_m", 0.28)
    )
    allow_partial = bool(args.allow_partial or grasp_cfg.get("allow_partial_ik", False))

    # 视觉引导：与 reset_and_grasp → grasp_from_observe preclose 同款参数
    def _pick(*vals, default=None):
        for v in vals:
            if v is not None:
                return v
        return default

    target_u = _pick(
        args.preclose_target_u,
        args.target_u,
        grasp_cfg.get("preclose_target_u_px"),
        grasp_cfg.get("place_target_u_px"),
    )
    target_v = _pick(
        args.preclose_target_v,
        args.target_v,
        grasp_cfg.get("preclose_target_v_px"),
        grasp_cfg.get("place_target_v_px"),
    )
    if target_u is not None:
        target_u = float(target_u)
    if target_v is not None:
        target_v = float(target_v)
    tol_u = float(
        _pick(
            args.preclose_tol_u,
            args.pixel_tol,
            args.tol_u,
            grasp_cfg.get("preclose_tol_u_px"),
            grasp_cfg.get("place_tol_u_px"),
            default=40.0,
        )
    )
    tol_v = float(
        _pick(
            args.preclose_tol_v,
            args.pixel_tol,
            args.tol_v,
            grasp_cfg.get("preclose_tol_v_px"),
            grasp_cfg.get("place_tol_v_px"),
            default=50.0,
        )
    )
    m_per_px = float(
        _pick(
            args.preclose_m_per_px,
            args.m_per_px,
            grasp_cfg.get("preclose_m_per_px"),
            grasp_cfg.get("place_m_per_px"),
            default=0.00012,
        )
    )
    max_step_m = float(
        _pick(
            args.preclose_max_step,
            args.max_step_m,
            grasp_cfg.get("preclose_max_step_m"),
            grasp_cfg.get("place_max_step_m"),
            default=0.012,
        )
    )
    settle_s = float(
        _pick(
            args.preclose_settle_s,
            args.settle_s,
            grasp_cfg.get("preclose_settle_s"),
            grasp_cfg.get("place_settle_s"),
            default=0.08,
        )
    )
    preclose_iters = int(
        _pick(
            args.preclose_iters,
            args.stable_frames,
            grasp_cfg.get("preclose_iters"),
            grasp_cfg.get("place_stable_frames"),
            default=8,
        )
    )
    loop_s = float(
        _pick(
            args.loop_s,
            grasp_cfg.get("place_live_loop_s"),
            grasp_cfg.get("preclose_loop_s"),
            default=0.05,
        )
    )
    timeout_s = float(
        _pick(
            args.timeout_s,
            grasp_cfg.get("place_live_timeout_s"),
            default=45.0,
        )
    )
    stable_need = int(
        _pick(
            args.stable_frames,
            grasp_cfg.get("place_stable_frames"),
            grasp_cfg.get("preclose_stable_frames"),
            default=3,
        )
    )
    save_every = int(
        _pick(
            args.save_every,
            grasp_cfg.get("place_live_save_every"),
            default=5,
        )
    )
    live_preview = not bool(args.no_live_preview) and bool(
        grasp_cfg.get("place_live_preview", False)
    )
    table_z = float(scene.get("table_z_m", 0.0))
    table_normal = scene.get("table_normal") or [0.0, 0.0, 1.0]
    _, grasp_z_m, grasp_label = grasp_z({**scene, "table_z_m": table_z})

    intr_path = Path(cfg.get("intrinsics", "output/camera_calib/camera_intrinsics.yaml"))
    if not intr_path.is_absolute():
        intr_path = (_TTA / intr_path).resolve()
    he_path = Path(cfg.get("handeye", "output/handeye_ee_cam.yaml"))
    if not he_path.is_absolute():
        he_path = (_TTA / he_path).resolve()

    try:
        K, dist = load_intrinsics(intr_path)
    except Exception as exc:
        print(f"[error] 内参加载失败（放置规划需要）: {exc}")
        return 1

    try:
        T_ee_cam = load_handeye(he_path)
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return 1

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

    print_pose(f"放置观察位臂关节({observe_pose_src})，夹爪将保持进入时实际值:", observe_hold)
    print(
        f"[info] og={args.og_config} vision={args.vision_config} "
        f"observe={observe_pose_src} range_mode={range_mode} "
        f"forward={forward_m*1000:.0f}mm z_offset={z_offset*1000:+.0f}mm "
        f"left={left_m*1000:.1f}mm route={route_points} "
        f"table_z={table_z:.3f} grasp_z={grasp_z_m:.3f}({grasp_label})"
    )
    tip_desc = (
        f"({target_u:.0f},{target_v:.0f})"
        if target_u is not None and target_v is not None
        else "(0.5w,0.55h)"
    )
    print(
        f"[info] live_servo={'off' if args.no_fine_align else 'on'} "
        f"loop={loop_s*1000:.0f}ms timeout={timeout_s:.0f}s stable={stable_need} "
        f"tip={tip_desc} tol=({tol_u:.0f},{tol_v:.0f}) "
        f"m_per_px={m_per_px} max_step={max_step_m*1000:.0f}mm preview={live_preview}"
    )

    arm = TicksArm(port, baud, motion)
    camera: Optional[LiveCamera] = None
    sequence_cfg = dict(cfg.get("sequence") or {})
    max_connect_attempts = max(
        1, int(sequence_cfg.get("connect_max_attempts", 5))
    )
    connect_retry_s = max(0.0, float(sequence_cfg.get("connect_retry_s", 0.8)))
    try:
        try:
            arm.connect_with_retry(
                max_attempts=max_connect_attempts,
                retry_s=connect_retry_s,
            )
        except (ConnectionError, OSError, RuntimeError, TimeoutError) as exc:
            err = str(exc)
            if "Missing motor" in err or "id: model_number" in err:
                print(
                    f"[error] 舵机握手失败（常见于夹爪 id=6 过载保护掉线）:\n{exc}\n"
                    f"已重试 {max_connect_attempts} 次仍失败。\n"
                    "请断电重启机械臂电源，确认 1–6 号舵机都在线后再运行。"
                )
                return 1
            raise
        camera = LiveCamera(camera_cfg)

        print(f"\n=== 1) 放置观察位持块（{observe_pose_src}） ===")
        actual0 = arm.read()
        # 夹爪保持进入本脚本前的实际 ticks，不写 yaml/gripper_close
        g_hold = float(actual0.get("gripper", g_close))
        observe_hold["gripper"] = g_hold
        observe_full["gripper"] = g_hold
        print(
            f"[place] 夹爪保持实际 ticks={g_hold:.0f}（观察位不改写） "
            f"(标定 close={g_close:.0f} open={g_open:.0f})"
        )
        if args.skip_observe_move:
            arm.goal = {
                **{j: float(actual0.get(j, observe_arm.get(j, 2048))) for j in JOINT_ORDER},
                "gripper": g_hold,
            }
            arm.current_name = "observe_hold"
            observe_arm = {n: float(arm.goal[n]) for n in JOINT_ORDER}
        else:
            # 只下发臂关节；目标字典不含 gripper，避免 go_pose_strict 重写夹爪
            move_pose = dict(observe_arm)
            try:
                go_pose_strict(arm, "place_observe_hold", move_pose)
            except Exception as exc:
                if "Overload" in str(exc):
                    print(f"[warn] 回观察位时过载，同步实际爪值后继续: {exc}")
                else:
                    raise
            # go_pose_strict 会把 goal 设成仅有臂关节；补回夹爪实际值
            try:
                g_hold = float(arm.read().get("gripper", g_hold))
            except Exception:
                pass
            if arm.goal is None:
                arm.goal = dict(move_pose)
            arm.goal["gripper"] = g_hold
            observe_arm = {n: float(arm.goal.get(n, move_pose[n])) for n in JOINT_ORDER}
        try:
            g_hold = float(arm.read().get("gripper", g_hold))
        except Exception:
            pass
        if arm.goal:
            arm.goal["gripper"] = g_hold
        print(f"[place] 持块 gripper 保持={g_hold:.0f}")

        print("\n=== 2) 单次识别 Tag → 记录放置位 → 开环前伸 ===")
        approach_status, forward_xy = approach_place_like_grasp(
            arm=arm,
            fk=fk,
            ik=ik,
            detector=detector,
            camera=camera,
            T_ee_cam=T_ee_cam,
            K=K,
            dist=dist,
            table_z=table_z,
            table_normal=np.asarray(table_normal, dtype=float),
            grasp_z_m=float(grasp_z_m),
            observe_arm=observe_arm,
            limits=limits,
            g_hold=g_hold,
            tag_id=args.tag_id,
            range_mode=range_mode,
            forward_m=forward_m,
            left_m=left_m,
            z_offset=z_offset,
            route_points=route_points,
            ik_tol=ik_tol,
            max_horiz_m=max_horiz_m,
            allow_partial=allow_partial,
            max_z_drop_m=max_z_drop_m,
            standoff_m=standoff_m,
            lock_wrist=lock_wrist,
            lock_elbow=lock_elbow,
            max_lift_delta_ticks=max_lift_delta_ticks,
            save_dir=save_dir,
            no_save=bool(args.no_save_image),
        )
        if approach_status == "no_tag":
            if args.keep_hold_on_no_tag:
                # exit 2：供导航前移重试；保持持块
                print("[place] 粗接近失败：未检测到 AprilTag (exit=2, keep hold)")
                return 2
            print(
                "[place] 未检测到 AprilTag：开爪并回初始位后退出，"
                "导航可继续正常运动"
            )
            open_gripper_and_return_initial(
                arm=arm,
                poses=poses,
                observe_arm=observe_arm,
                limits=limits,
                g_open=g_open,
                g_hold=g_hold,
                grasp_cfg=grasp_cfg,
                open_gripper=not bool(args.no_open),
            )
            if args.hold_after:
                hold_forever_safe(arm, motion)
            return 0
        if approach_status != "ok":
            print(f"[place] 粗接近失败：{approach_status} (exit=1)")
            return 1

        # 默认开启：同 reset_and_grasp → grasp_from_observe preclose
        do_fine_align = not bool(args.no_fine_align)
        if do_fine_align:
            print("\n=== 3) 实时视觉引导（边运动边取最新帧） ===")
            ok = refine_pose_until_tag_aligned(
                arm=arm,
                fk=fk,
                ik=ik,
                detector=detector,
                camera=camera,
                T_ee_cam=T_ee_cam,
                limits=limits,
                g_hold=g_hold,
                observe_arm=observe_arm,
                tag_id=args.tag_id,
                target_u=target_u,
                target_v=target_v,
                tol_u=tol_u,
                tol_v=tol_v,
                m_per_px=m_per_px,
                max_step_m=max_step_m,
                max_iters=preclose_iters,
                settle_s=settle_s,
                ik_tol=ik_tol,
                max_horiz_m=max_horiz_m,
                max_z_drop_m=max_z_drop_m,
                save_dir=save_dir,
                no_save=bool(args.no_save_image),
                approach_sign=float(args.xy_sign),
                loop_s=loop_s,
                timeout_s=timeout_s,
                stable_need=stable_need,
                live_preview=live_preview,
                save_every=save_every,
            )
            if not ok:
                print(
                    "[place] 实时视觉引导未完全对准，仍继续下降并开爪放置 "
                    "（需要严格对准可改代码或加逻辑中止）"
                )
        else:
            print("\n=== 3) --no-fine-align，跳过实时视觉引导 ===")

        if final_probe > 1e-4 and forward_xy is not None:
            print(f"\n=== 4) 最终前探 {final_probe*1000:.0f} mm（开环，沿记录方向） ===")
            try:
                advance_along_forward_place(
                    arm=arm,
                    fk=fk,
                    ik=ik,
                    limits=limits,
                    g_hold=g_hold,
                    observe_arm=observe_arm,
                    forward_xy=forward_xy,
                    distance_m=final_probe,
                    route_points=final_route_pts,
                    ik_tol=ik_tol,
                    max_horiz_m=max_horiz_m,
                    max_z_drop_m=max_z_drop_m,
                    lock_wrist=lock_wrist,
                    lock_elbow=lock_elbow,
                    max_lift_delta_ticks=max_lift_delta_ticks,
                    label="place_final_probe",
                )
            except RuntimeError as exc:
                print(f"[warn] 最终前探失败，按当前位置继续: {exc}")

        if place_descend_m > 1e-4:
            print(f"\n=== 5) 对准后下降 {place_descend_m*1000:.0f} mm ===")
            ok_down = descend_ee_hold_xy(
                arm=arm,
                fk=fk,
                ik=ik,
                observe_arm=observe_arm,
                limits=limits,
                g_hold=g_hold,
                descend_m=place_descend_m,
                route_points=place_descend_pts,
                ik_tol=ik_tol,
                max_horiz_m=max_horiz_m,
                label="place_descend",
            )
            if not ok_down:
                print("[warn] 下降失败，仍尝试开爪")
        else:
            print("\n=== 5) place_descend_m=0，跳过下降 ===")

        # 开爪前保存当前画面（无论是否真的开爪，都在这一步拍）
        if camera is not None and not bool(args.no_save_image):
            print("\n=== 6a) 开爪前拍照 ===")
            try:
                detect_and_save(
                    detector=detector,
                    camera=camera,
                    tag_id=args.tag_id,
                    target_u=target_u,
                    target_v=target_v,
                    tol_u=tol_u,
                    tol_v=tol_v,
                    save_dir=save_dir,
                    no_save=False,
                    tag="before_open",
                    flush=3,
                )
            except Exception as exc:
                print(f"[warn] 开爪前拍照失败: {exc}")

        if not args.no_open:
            print(f"\n=== 6) 开爪放下 -> {int(g_open)} ===")
            try:
                set_gripper(arm, g_open, limits, float(grasp_cfg.get("close_settle_s", 0.35)))
            except Exception as exc:
                print(f"[warn] 开爪写入失败，请手动确认夹爪: {exc}")
        else:
            print("\n=== 6) --no-open，保持持块 ===")

        open_gripper_and_return_initial(
            arm=arm,
            poses=poses,
            observe_arm=observe_arm,
            limits=limits,
            g_open=g_open,
            g_hold=g_hold,
            grasp_cfg=grasp_cfg,
            open_gripper=False,  # 上面已开爪；此处只回初始位
        )

        if args.hold_after:
            hold_forever_safe(arm, motion)
            return 0

        print("[place] 放置完成，退出（导航可继续）")
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
