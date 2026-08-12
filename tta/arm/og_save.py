#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 视觉指导抓取（原始 ticks）。

参考:
  - grasp_test/horizontal_red_grasp_test.py (factory_ik 跟踪→锁定→下降)
  - armpi_pro intelligent_grasp_node.py (PID 死区锁、稳定后下降)
  - arm/grasp.py (pan/reach ticks 伺服 + 保留 pan 抓取)

核心差异（相对旧开环 insert）:
  视觉积分出 pan/reach → 稳定后带着锁定量 approach→descend→close
  而不是对准后盲目叠加固定 insert_delta。

用法:
  python3 og.py --yes
  # initial | grasp | place | auto | detect | q
  python3 og.py --yes --auto-grasp
  python3 og.py --yes --sequence
"""

from __future__ import annotations

import argparse
import copy
import select
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from start import (  # noqa: E402
    DEFAULT_BAUD,
    connect_bus,
    disable_torque_safe,
    enable_torque_safe,
    move_to,
    read_positions,
)
from so101_red_block_camera_test import (  # noqa: E402
    RedBlockDetector,
    camera_source,
)

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
POSE_NAMES = ("initial", "grasp", "place")
DEFAULT_CONFIG_PATH = _HERE / "og.yaml"

CFG = {
    "port": "/dev/ttyACM1",
    "baud": DEFAULT_BAUD,
    "motion": {
        "step_ticks": 40,
        "step_s": 0.06,
        "settle_s": 0.4,
        "hold_hz": 10.0,
        "arrive_tol_ticks": 12,
        "joint_limits": {
            "shoulder_pan": [800, 3300],
            "shoulder_lift": [800, 3300],
            "elbow_flex": [800, 3300],
            "wrist_flex": [800, 3300],
            "wrist_roll": [800, 3300],
            "gripper": [700, 2800],
        },
    },
    "camera": {
        "index_or_path": "/dev/video11",
        "width": 640,
        "height": 480,
        "fps": 30,
        "settle_frames": 8,
        "frame_timeout_s": 5.0,
    },
    "vision": {
        "enabled": True,
        "detect_on_arrive": True,
        "desired_center": [320, 395],
        "output_directory": "output/og_grasp_vision",
    },
    "detector": {
        "roi": [0.08, 0.15, 0.92, 0.92],
        "red_hsv": {
            "lower1": [0, 75, 45],
            "upper1": [18, 255, 255],
            "lower2": [165, 75, 45],
            "upper2": [180, 255, 255],
        },
        "red_rgb": {
            "min_r": 95,
            "min_r_minus_g": 28,
            "min_r_minus_b": 22,
            "max_g": 205,
        },
        "min_area": 500,
        "min_rect_fill": 0.40,
        "aspect_ratio_range": [0.35, 2.80],
        "open_kernel": 3,
        "close_kernel": 5,
    },
    "auto_grasp": {
        "enabled": True,
        "run_after_align": True,
        "go_place_after": True,
        "return_initial": True,
        "gripper_open": 1200,
        "gripper_close": 2100,
        "x_pid_p": 0.35,
        "y_pid_p": 0.25,
        "pid_lock_px": 10,
        "stable_err_u": 22,
        "stable_err_v": 28,
        "stable_frames": 3,
        "track_timeout_s": 14.0,
        "continue_on_timeout": True,
        "track_settle_first_s": 0.25,
        "track_settle_s": 0.08,
        "position_settle_px": 3,
        "position_settle_frames": 5,
        "pan_limits": [1600, 2500],
        "reach_limits": [-250, 250],
        "reach_elbow_scale": 0.8,
        "reach_lift_scale": -0.35,
        "grasp_reach_extra": 25,
        "approach_delta": {"shoulder_lift": -25, "elbow_flex": 30, "wrist_flex": 10},
        "descend_delta": {"shoulder_lift": -55, "elbow_flex": 55, "wrist_flex": 20},
        "lift_delta": {"shoulder_lift": 80, "elbow_flex": -45, "wrist_flex": -15},
        "approach_settle_s": 0.8,
        "descend_settle_s": 1.0,
        "close_settle_s": 0.85,
        "lift_settle_s": 0.8,
        "close_require_visible": False,
        "close_max_err": [50, 60],
        "close_min_area": 0,
    },
    "poses": {
        "initial": {
            "shoulder_pan": 2048,
            "shoulder_lift": 1900,
            "elbow_flex": 2300,
            "wrist_flex": 2048,
            "wrist_roll": 2048,
            "gripper": 1200,
        },
        "grasp": {
            "shoulder_pan": 2048,
            "shoulder_lift": 1800,
            "elbow_flex": 2400,
            "wrist_flex": 2100,
            "wrist_roll": 2048,
            "gripper": 1200,
        },
        "place": {
            "shoulder_pan": 2300,
            "shoulder_lift": 1800,
            "elbow_flex": 2400,
            "wrist_flex": 2100,
            "wrist_roll": 2048,
            "gripper": 1200,
        },
    },
}


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


def deep_update(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict:
    cfg = copy.deepcopy(CFG)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config root must be a mapping: {path}")
        cfg = deep_update(cfg, loaded)
        print(f"[config] loaded {path}")
    else:
        print(f"[warn] 配置不存在: {path}，使用内置默认")
    return cfg


def pose_i(pose: Dict[str, float]) -> Dict[str, int]:
    return {k: int(round(float(v))) for k, v in pose.items()}


def clamp(value: float, limits: Optional[List[float]]) -> float:
    if not limits:
        return float(value)
    return max(float(limits[0]), min(float(limits[1]), float(value)))


def clamp_pose(pose: Dict[str, float], limits: dict) -> Dict[str, float]:
    return {n: clamp(float(v), limits.get(n)) for n, v in pose.items()}


def print_pose(title: str, pose: Dict[str, float]) -> None:
    print(title)
    for name in JOINTS:
        if name in pose:
            print(f"  {name}: {int(round(float(pose[name])))}")


def add_delta(pose: Dict[str, float], delta: Dict[str, float], limits: dict) -> Dict[str, float]:
    out = dict(pose)
    for k, dv in (delta or {}).items():
        if k in out:
            out[k] = clamp(float(out[k]) + float(dv), limits.get(k))
    return out


def _stdin_ready(timeout_s: float) -> bool:
    if not sys.stdin.isatty():
        time.sleep(timeout_s)
        return False
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        return bool(ready)
    except (OSError, ValueError):
        time.sleep(timeout_s)
        return False


class SimplePID:
    """P-only PID，配合死区锁定 setpoint（同 grasp.py / intelligent_grasp）。"""

    def __init__(self, p: float):
        self.p = float(p)
        self.setpoint = 0.0
        self.output = 0.0

    def clear(self) -> None:
        self.output = 0.0

    def update(self, measurement: float) -> float:
        self.output = self.p * (self.setpoint - float(measurement))
        return self.output


@dataclass
class TrackLock:
    """视觉跟踪锁定结果：后续下降必须带着 pan/reach。"""

    pan: float
    reach: float
    base_lift: float
    base_elbow: float
    base_wrist: float
    base_roll: float
    pose: Dict[str, float]
    stable: bool
    ever_seen: bool


# ---------------------------------------------------------------------------
# arm
# ---------------------------------------------------------------------------


class TicksArm:
    def __init__(self, port: str, baud: int, motion_cfg: dict):
        self.port = port
        self.baud = baud
        self.motion = motion_cfg
        self.bus = None
        self.goal: Dict[str, float] = {}
        self.current_name: Optional[str] = None

    def connect(self) -> None:
        self.bus = connect_bus(self.port, baud=self.baud, configure=False)
        enable_torque_safe(self.bus)
        time.sleep(0.15)
        print("[torque] ON")

    def disconnect(self, release_torque: bool = False) -> None:
        if self.bus is None:
            return
        if release_torque:
            print("[torque] OFF")
            disable_torque_safe(self.bus)
        try:
            self.bus.disconnect(disable_torque=False)
        except Exception:
            pass
        self.bus = None

    def read(self) -> Dict[str, float]:
        assert self.bus is not None
        return {k: float(v) for k, v in read_positions(self.bus).items()}

    def _write_goal(self, pose: Dict[str, float]) -> None:
        assert self.bus is not None
        limited = clamp_pose(pose, self.motion.get("joint_limits") or {})
        move_to(self.bus, pose_i(limited), wait_s=0.0)
        self.goal = dict(limited)

    def hold_tick(self) -> None:
        if self.goal:
            self._write_goal(self.goal)

    def send_partial(self, pose: Dict[str, float], settle_s: float = 0.0) -> None:
        merged = dict(self.goal)
        merged.update({k: float(v) for k, v in pose.items()})
        self._write_goal(merged)
        if settle_s > 0:
            time.sleep(settle_s)

    def write_pose(self, pose: Dict[str, float], settle_s: float, name: str = "") -> None:
        """跟踪环用：直接写目标 + settle（不做长距离插值）。"""
        self._write_goal(pose)
        if name:
            self.current_name = name
        if settle_s > 0:
            time.sleep(settle_s)

    def go_pose(self, name: str, target: Dict[str, float]) -> Dict[str, float]:
        assert self.bus is not None
        limits = self.motion.get("joint_limits") or {}
        target = clamp_pose({k: float(v) for k, v in target.items()}, limits)
        step = float(self.motion["step_ticks"])
        step_s = float(self.motion["step_s"])
        tol = float(self.motion["arrive_tol_ticks"])

        print(f"\n>>> 平滑切换: {self.current_name or 'current'} -> {name}")
        print_pose("目标 ticks:", target)

        current = self.read()
        max_delta = max(abs(float(target[n]) - float(current.get(n, target[n]))) for n in target)
        steps = max(1, int(np.ceil(max_delta / max(step, 1.0))))

        for i in range(steps):
            actual = self.read()
            command: Dict[str, float] = {}
            done = True
            for joint, goal in target.items():
                cur = float(actual.get(joint, current.get(joint, goal)))
                err = float(goal) - cur
                if abs(err) > tol:
                    done = False
                command[joint] = cur + max(-step, min(step, err))
            self._write_goal(command)
            if done:
                break
            time.sleep(step_s)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  step {i + 1}/{steps}")

        self._write_goal(target)
        time.sleep(float(self.motion["settle_s"]))
        final = self.read()
        print_pose("实际到达:", final)
        self.current_name = name
        self.goal = dict(target)
        print(f"[holding] 位姿={name}，力矩保持中")
        return final

    def hold_until_command(self) -> str:
        period = 1.0 / max(float(self.motion["hold_hz"]), 0.5)
        label = self.current_name or "custom"
        print(f"[holding {label}] 命令: initial | grasp | place | auto | detect | q")
        while True:
            self.hold_tick()
            if _stdin_ready(period):
                line = sys.stdin.readline()
                if not line:
                    continue
                cmd = line.strip().lower()
                if cmd:
                    return cmd


# ---------------------------------------------------------------------------
# vision
# ---------------------------------------------------------------------------


class GraspVision:
    """持久打开相机，减少跟踪环每帧 reopen 的延迟。"""

    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("vision", {}).get("enabled", True))
        self.vision = cfg.get("vision", {})
        self.camera_cfg = copy.deepcopy(cfg.get("camera", CFG["camera"]))
        self.detector = RedBlockDetector({"detector": cfg["detector"]})
        out = Path(self.vision.get("output_directory", "output/og_grasp_vision"))
        if not out.is_absolute():
            out = _HERE.parent / out
        self.output_dir = out
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._cap: Optional[cv2.VideoCapture] = None

    def desired_center(self) -> Tuple[float, float]:
        c = self.vision.get("desired_center", [320, 395])
        return float(c[0]), float(c[1])

    def open(self) -> bool:
        if not self.enabled:
            return False
        if self._cap is not None and self._cap.isOpened():
            return True
        source = camera_source(self.camera_cfg["index_or_path"])
        if isinstance(source, int):
            self._cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(source)
        else:
            self._cap = cv2.VideoCapture(str(source))
        if not self._cap.isOpened():
            print(f"[vision] 无法打开相机: {source!r}")
            self._cap = None
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.camera_cfg["width"]))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.camera_cfg["height"]))
        self._cap.set(cv2.CAP_PROP_FPS, int(self.camera_cfg["fps"]))
        for _ in range(max(1, int(self.camera_cfg.get("settle_frames", 8)))):
            self._cap.read()
        print(f"[vision] camera open: {source}")
        return True

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_frame(self, discard: int = 2) -> Optional[np.ndarray]:
        if not self.open() or self._cap is None:
            return None
        frame = None
        for _ in range(max(1, discard)):
            ok, frame = self._cap.read()
            if not ok:
                return None
        return frame

    def detect_frame(self, frame: np.ndarray, tag: str = "track") -> Optional[object]:
        detection = self.detector.detect(frame)
        annotated = self.detector.draw(frame, detection)
        du, dv = self.desired_center()
        cv2.drawMarker(annotated, (int(round(du)), int(round(dv))), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
        if detection is not None:
            cv2.putText(
                annotated,
                f"err=({detection.center_u - du:+.0f},{detection.center_v - dv:+.0f})",
                (20, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (50, 220, 50),
                2,
            )
        path = self.output_dir / f"{tag}_{datetime.now():%H%M%S}_{self._seq:03d}.jpg"
        self._seq += 1
        cv2.imwrite(str(path), annotated)
        if detection is None:
            print(f"[vision] 未检测到红块 -> {path.name}")
        else:
            print(
                f"[vision] center=({detection.center_u:.1f},{detection.center_v:.1f}) "
                f"area={detection.area:.0f} -> {path.name}"
            )
        return detection


# ---------------------------------------------------------------------------
# vision-guided track + grasp
# ---------------------------------------------------------------------------


def _track_pose_from_lock(
    lock: TrackLock,
    ag: dict,
    joint_limits: dict,
    *,
    reach_extra: float = 0.0,
    gripper: float,
    delta: Optional[dict] = None,
) -> Dict[str, float]:
    """用锁定的 pan/reach 生成位姿；可选叠加竖直 delta。shoulder_pan 始终锁定。"""
    reach = float(lock.reach) + float(reach_extra)
    pose = {
        "shoulder_pan": float(lock.pan),
        "shoulder_lift": float(lock.base_lift) + reach * float(ag["reach_lift_scale"]),
        "elbow_flex": float(lock.base_elbow) + reach * float(ag["reach_elbow_scale"]),
        "wrist_flex": float(lock.base_wrist),
        "wrist_roll": float(lock.base_roll),
        "gripper": float(gripper),
    }
    if delta:
        pose = add_delta(pose, delta, joint_limits)
        pose["shoulder_pan"] = float(lock.pan)  # 强制保留视觉左右
    return clamp_pose(pose, joint_limits)


def visual_track_lock(arm: TicksArm, vision: GraspVision, ag: dict, joint_limits: dict) -> Optional[TrackLock]:
    """
    PID 跟踪红块，积分 pan/reach；稳定或超时后返回锁定量。
    参考: grasp.py visual_servo_align / factory_ik / intelligent_grasp run()
    """
    if not vision.enabled:
        print("[track] vision disabled")
        return None

    desired_u, desired_v = vision.desired_center()
    lock_px = float(ag.get("pid_lock_px", 10))
    stable_u = float(ag.get("stable_err_u", 22))
    stable_v = float(ag.get("stable_err_v", 28))
    stable_need = int(ag.get("stable_frames", 3))
    timeout = float(ag.get("track_timeout_s", 14.0))
    continue_on_timeout = bool(ag.get("continue_on_timeout", True))
    settle0 = float(ag.get("track_settle_first_s", 0.25))
    settle = float(ag.get("track_settle_s", 0.08))
    pos_px = float(ag.get("position_settle_px", 3))
    pos_need = int(ag.get("position_settle_frames", 5))
    pan_limits = ag.get("pan_limits", [1600, 2500])
    reach_limits = ag.get("reach_limits", [-250, 250])
    open_g = float(ag["gripper_open"])

    base = dict(arm.goal) if arm.goal else arm.read()
    base["gripper"] = open_g
    arm.send_partial({"gripper": open_g}, settle_s=0.35)

    pan = float(base["shoulder_pan"])
    reach = 0.0
    base_lift = float(base["shoulder_lift"])
    base_elbow = float(base["elbow_flex"])
    base_wrist = float(base.get("wrist_flex", 2048))
    base_roll = float(base.get("wrist_roll", 2048))

    x_pid = SimplePID(float(ag["x_pid_p"]))
    y_pid = SimplePID(float(ag["y_pid_p"]))
    x_pid.clear()
    y_pid.clear()

    stable_count = 0
    pos_count = 0
    position_en = pos_need <= 0
    last_center: Optional[Tuple[float, float]] = None
    ever_seen = False
    first = True
    deadline = time.monotonic() + timeout

    print("[track] 开始视觉跟踪（PID + 死区锁定）...")
    while time.monotonic() < deadline:
        arm.hold_tick()
        frame = vision.get_frame(discard=2)
        if frame is None:
            time.sleep(0.05)
            continue
        det = vision.detect_frame(frame, tag="track")
        if det is None:
            stable_count = 0
            pos_count = 0
            time.sleep(0.05)
            continue

        ever_seen = True
        cu, cv_ = float(det.center_u), float(det.center_v)
        err_u = cu - desired_u
        err_v = cv_ - desired_v

        # 目标放稳门控（intelligent_grasp）
        if not position_en:
            if last_center is not None:
                du = abs(cu - last_center[0])
                dv = abs(cv_ - last_center[1])
                if du < pos_px and dv < pos_px:
                    pos_count += 1
                else:
                    pos_count = 0
                if pos_count >= pos_need:
                    position_en = True
                    print("[track] 目标已放稳，开始跟随")
            last_center = (cu, cv_)
            if not position_en:
                time.sleep(0.05)
                continue
        last_center = (cu, cv_)

        # PID 死区锁定
        x_pid.setpoint = cu if abs(err_u) < lock_px else desired_u
        y_pid.setpoint = cv_ if abs(err_v) < lock_px else desired_v
        x_pid.update(cu)
        y_pid.update(cv_)

        pan = clamp(pan + x_pid.output, pan_limits)
        reach = clamp(reach + y_pid.output, reach_limits)

        lock = TrackLock(
            pan=pan,
            reach=reach,
            base_lift=base_lift,
            base_elbow=base_elbow,
            base_wrist=base_wrist,
            base_roll=base_roll,
            pose={},
            stable=False,
            ever_seen=True,
        )
        pose = _track_pose_from_lock(lock, ag, joint_limits, gripper=open_g)
        lock.pose = pose
        arm.write_pose(pose, settle0 if first else settle, name="grasp")
        first = False

        print(
            f"[track] err=({err_u:+.1f},{err_v:+.1f}) "
            f"pan={pan:.0f} reach={reach:+.0f} pid=({x_pid.output:+.2f},{y_pid.output:+.2f})"
        )

        if abs(err_u) <= stable_u and abs(err_v) <= stable_v:
            stable_count += 1
            if stable_count >= stable_need:
                lock.stable = True
                print(f"[track] 对准稳定 stable={stable_count} pan={pan:.0f} reach={reach:+.0f}")
                return lock
        else:
            stable_count = 0

    if not ever_seen:
        print("[track] 超时且从未检测到红块")
        return None
    if not continue_on_timeout:
        print("[track] 超时且 continue_on_timeout=false，取消")
        return None

    print(f"[track] 超时，使用最后 pan={pan:.0f} reach={reach:+.0f} 继续抓取")
    lock = TrackLock(
        pan=pan,
        reach=reach,
        base_lift=base_lift,
        base_elbow=base_elbow,
        base_wrist=base_wrist,
        base_roll=base_roll,
        pose={},
        stable=False,
        ever_seen=True,
    )
    lock.pose = _track_pose_from_lock(lock, ag, joint_limits, gripper=open_g)
    return lock


def execute_vision_grasp(
    arm: TicksArm,
    lock: TrackLock,
    vision: GraspVision,
    ag: dict,
    joint_limits: dict,
) -> Dict[str, float]:
    """
    带着锁定的 pan/reach：开爪悬停 → 下降 → (可选二次检测) → 闭爪 → 抬起。
    参考 intelligent_grasp move() / grasp.py collision_safe_grasp。
    """
    open_g = float(ag["gripper_open"])
    close_g = float(ag["gripper_close"])
    extra = float(ag.get("grasp_reach_extra", 0))

    print("[grasp] approach hover（保留视觉 pan/reach）")
    approach = _track_pose_from_lock(
        lock,
        ag,
        joint_limits,
        reach_extra=0.0,
        gripper=open_g,
        delta=ag.get("approach_delta"),
    )
    arm.go_pose("approach", approach)
    time.sleep(float(ag.get("approach_settle_s", 0.8)))

    print(f"[grasp] descend（reach_extra={extra:+.0f}）")
    descend = _track_pose_from_lock(
        lock,
        ag,
        joint_limits,
        reach_extra=extra,
        gripper=open_g,
        delta=ag.get("descend_delta"),
    )
    arm.go_pose("descend", descend)
    time.sleep(float(ag.get("descend_settle_s", 1.0)))

    if ag.get("close_require_visible", False) and vision.enabled:
        frame = vision.get_frame(discard=2)
        if frame is not None:
            det = vision.detect_frame(frame, tag="pre_close")
            du, dv = vision.desired_center()
            max_u, max_v = map(float, ag.get("close_max_err", [50, 60]))
            min_area = float(ag.get("close_min_area", 0))
            if det is None:
                print("[grasp] 闭爪前看不到红块，仍闭爪（可改 close_require_visible）")
            else:
                eu, ev = det.center_u - du, det.center_v - dv
                if abs(eu) > max_u or abs(ev) > max_v or det.area < min_area:
                    print(f"[grasp] 闭爪前门控未过 err=({eu:+.0f},{ev:+.0f}) area={det.area:.0f}")

    print(f"[grasp] close gripper -> {int(close_g)}")
    arm.send_partial({"gripper": close_g}, settle_s=float(ag.get("close_settle_s", 0.85)))

    print("[grasp] lift（夹爪保持闭合）")
    lift = _track_pose_from_lock(
        lock,
        ag,
        joint_limits,
        reach_extra=0.0,
        gripper=close_g,
        delta=ag.get("lift_delta"),
    )
    arm.go_pose("lift", lift)
    time.sleep(float(ag.get("lift_settle_s", 0.8)))
    print("[grasp] 视觉指导抓取完成")
    return lift


def run_auto_grasp(
    arm: TicksArm,
    poses: dict,
    vision: GraspVision,
    cfg: dict,
    *,
    force: bool = False,
) -> bool:
    ag = cfg.get("auto_grasp") or {}
    if not ag.get("enabled", True) and not force:
        print("[auto] auto_grasp.enabled=false")
        return False
    if not vision.enabled:
        print("[auto] vision disabled")
        return False

    joint_limits = cfg["motion"].get("joint_limits") or {}
    open_g = float(ag.get("gripper_open", 1200))

    if arm.current_name != "grasp":
        gpose = dict(poses["grasp"])
        gpose["gripper"] = open_g
        arm.go_pose("grasp", gpose)

    if not vision.open():
        return False

    if vision.vision.get("detect_on_arrive", True):
        frame = vision.get_frame(discard=3)
        if frame is not None:
            vision.detect_frame(frame, tag="arrive")

    lock = visual_track_lock(arm, vision, ag, joint_limits)
    if lock is None:
        print("[auto] 跟踪失败，取消抓取")
        return False

    if not (ag.get("run_after_align", True) or force):
        print("[auto] 已锁定；run_after_align=false，跳过下降抓取")
        arm.goal = lock.pose
        arm.current_name = "grasp"
        return True

    execute_vision_grasp(arm, lock, vision, ag, joint_limits)

    if ag.get("go_place_after", True):
        place = dict(poses["place"])
        place_hold = dict(place)
        place_hold["gripper"] = float(ag.get("gripper_close", 2100))
        arm.go_pose("place", place_hold)
        arm.send_partial(
            {"gripper": float(ag.get("gripper_open", place.get("gripper", open_g)))},
            settle_s=0.6,
        )
        arm.current_name = "place"

    if ag.get("return_initial", True):
        arm.go_pose("initial", poses["initial"])

    return True


def resolve_poses(cfg: dict) -> Dict[str, Dict[str, float]]:
    return {name: {j: float(cfg["poses"][name][j]) for j in JOINTS} for name in POSE_NAMES}


def cmd_read(port: str, baud: int, motion_cfg: dict) -> int:
    arm = TicksArm(port, baud, motion_cfg)
    try:
        arm.connect()
        print_pose("当前舵机原始 ticks:", arm.read())
        return 0
    finally:
        arm.disconnect(release_torque=False)


def run_interactive(arm: TicksArm, poses: dict, vision: GraspVision, cfg: dict, auto_on_grasp: bool) -> int:
    arm.go_pose("initial", poses["initial"])
    ag = cfg.get("auto_grasp") or {}
    while True:
        cmd = arm.hold_until_command()
        if cmd in {"q", "quit", "exit"}:
            return 0
        if cmd in {"i", "init", "initial", "home"}:
            arm.go_pose("initial", poses["initial"])
            continue
        if cmd in {"g", "grasp"}:
            gpose = dict(poses["grasp"])
            gpose["gripper"] = float(ag.get("gripper_open", gpose.get("gripper", 1200)))
            arm.go_pose("grasp", gpose)
            if vision.enabled:
                vision.open()
                frame = vision.get_frame(discard=3)
                if frame is not None:
                    vision.detect_frame(frame, tag="grasp_cmd")
            if auto_on_grasp:
                run_auto_grasp(arm, poses, vision, cfg, force=True)
            continue
        if cmd in {"p", "place"}:
            arm.go_pose("place", poses["place"])
            continue
        if cmd in {"d", "detect"}:
            if arm.current_name != "grasp":
                print("[vision] 请先到 grasp")
            else:
                vision.open()
                frame = vision.get_frame(discard=2)
                if frame is not None:
                    vision.detect_frame(frame, tag="manual")
            continue
        if cmd in {"a", "auto", "grab"}:
            run_auto_grasp(arm, poses, vision, cfg, force=True)
            continue
        print(f"未知命令: {cmd!r}")


def run_sequence(arm: TicksArm, poses: dict, vision: GraspVision, cfg: dict, hold_s: float) -> int:
    arm.go_pose("initial", poses["initial"])
    if hold_s > 0:
        time.sleep(hold_s)
    ok = run_auto_grasp(arm, poses, vision, cfg, force=True)
    print("sequence done" if ok else "sequence failed")
    try:
        while True:
            arm.hold_tick()
            time.sleep(1.0 / max(float(arm.motion["hold_hz"]), 0.5))
    except KeyboardInterrupt:
        print("\n用户中断")
        return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="SO-101 视觉指导抓取（ticks）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--read", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--auto-grasp", action="store_true")
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--release-on-exit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--step-ticks", type=float, default=None)
    parser.add_argument("--step-s", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.step_ticks is not None:
        cfg["motion"]["step_ticks"] = float(args.step_ticks)
    if args.step_s is not None:
        cfg["motion"]["step_s"] = float(args.step_s)
    if args.camera is not None:
        cfg["camera"]["index_or_path"] = camera_source(args.camera)
    if args.no_vision:
        cfg["vision"]["enabled"] = False

    port = args.port or cfg["port"]
    baud = int(args.baud or cfg["baud"])
    poses = resolve_poses(cfg)
    vision = GraspVision(cfg)

    for name in POSE_NAMES:
        print_pose(f"位姿[{name}]:", poses[name])
    ag = cfg.get("auto_grasp") or {}
    print(f"视觉指导: desired_center={cfg['vision'].get('desired_center')}")
    print(
        f"PID: x={ag.get('x_pid_p')} y={ag.get('y_pid_p')} "
        f"timeout={ag.get('track_timeout_s')}s reach_extra={ag.get('grasp_reach_extra')}"
    )

    if args.dry_run:
        return 0
    if args.read:
        return cmd_read(port, baud, cfg["motion"])
    if not args.yes:
        print("Refusing motion: pass --yes")
        return 2

    arm = TicksArm(port, baud, cfg["motion"])
    try:
        arm.connect()
        if args.sequence:
            return run_sequence(arm, poses, vision, cfg, float(args.hold_s))
        return run_interactive(arm, poses, vision, cfg, auto_on_grasp=bool(args.auto_grasp))
    except KeyboardInterrupt:
        print("\n用户中断")
        return 130
    finally:
        vision.close()
        arm.disconnect(release_torque=bool(args.release_on_exit))
        if not args.release_on_exit:
            print("已断开串口；力矩未主动关闭（可用 --release-on-exit）")


if __name__ == "__main__":
    raise SystemExit(main())
