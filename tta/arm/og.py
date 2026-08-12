#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 三位姿（ticks）+ 抓取位红块视觉对准与自动抓取。

位姿:
  initial  初始
  grasp    观察/对准（视觉伺服）
  place    放置

流水线（grasp）:
  到 grasp → 开爪 → 红块视觉伺服对准 → pregrasp → insert → 闭爪 → retreat
  →（可选）place →（可选）initial

参考:
  - so101_red_block_camera_test.py（检测）
  - grasp.py / so101_horizontal_red_grasp.py（像素→关节、抓取序列）
  - start.py（Feetech 原始 ticks）

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
    capture_frame,
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
        "detect_hz": 0.0,
        "desired_center": [320, 360],
        "output_directory": "output/og_grasp_vision",
    },
    "detector": {
        "roi": [0.06, 0.10, 0.94, 0.94],
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
        "min_area": 180,
        "min_rect_fill": 0.45,
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
        "pan_ticks_per_pixel": 0.35,
        "reach_ticks_per_pixel": 0.25,
        "max_joint_step_ticks": 40,
        "pixel_deadband": [22, 28],
        "max_align_iters": 30,
        "stable_frames": 3,
        "loop_hz": 5.0,
        "align_step_s": 0.08,
        "reach_elbow_scale": 0.8,
        "reach_lift_scale": -0.35,
        "pan_limits": [1600, 2500],
        "reach_limits": [-250, 250],
        "pregrasp_delta": {
            "shoulder_pan": 0,
            "shoulder_lift": -30,
            "elbow_flex": 40,
            "wrist_flex": 0,
            "wrist_roll": 0,
        },
        "insert_delta": {
            "shoulder_pan": 0,
            "shoulder_lift": -50,
            "elbow_flex": 70,
            "wrist_flex": 20,
            "wrist_roll": 0,
        },
        "retreat_delta": {
            "shoulder_pan": 0,
            "shoulder_lift": 90,
            "elbow_flex": -60,
            "wrist_flex": -20,
            "wrist_roll": 0,
        },
        "pregrasp_settle_s": 0.5,
        "insert_settle_s": 0.8,
        "close_settle_s": 0.85,
        "retreat_settle_s": 0.8,
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
            "shoulder_lift": 1650,
            "elbow_flex": 2550,
            "wrist_flex": 2150,
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
    if not path.exists():
        print(f"[warn] 配置不存在: {path}，使用内置默认")
        return cfg
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    cfg = deep_update(cfg, loaded)
    print(f"[config] loaded {path}")
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
    for k, dv in delta.items():
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
        """合并到当前 goal 后写出（用于只改夹爪等）。"""
        merged = dict(self.goal)
        merged.update({k: float(v) for k, v in pose.items()})
        self._write_goal(merged)
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


class GraspVision:
    def __init__(self, cfg: dict):
        self.cfg = cfg
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

    def desired_center(self) -> Tuple[float, float]:
        c = self.vision.get("desired_center", [320, 360])
        return float(c[0]), float(c[1])

    def capture_detect(self, tag: str = "grasp") -> Tuple[Optional[object], Optional[np.ndarray]]:
        if not self.enabled:
            return None, None
        try:
            frame = capture_frame(self.camera_cfg)
        except RuntimeError as exc:
            print(f"[vision] camera error: {exc}")
            return None, None
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
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{tag}_{stamp}_{self._seq:03d}.jpg"
        self._seq += 1
        cv2.imwrite(str(path), annotated)
        if detection is None:
            print(f"[vision] 未检测到红块 -> {path}")
        else:
            print(
                f"[vision] center=({detection.center_u:.1f},{detection.center_v:.1f}) "
                f"area={detection.area:.0f} -> {path}"
            )
        return detection, annotated


def visual_servo_align(arm: TicksArm, vision: GraspVision, ag: dict, joint_limits: dict) -> Optional[Dict[str, float]]:
    """像素误差 → ticks 视觉伺服，返回对准后的位姿；失败返回 None。"""
    if not vision.enabled:
        print("[align] vision disabled")
        return dict(arm.goal)

    desired_u, desired_v = vision.desired_center()
    dead_u, dead_v = map(float, ag.get("pixel_deadband", [22, 28]))
    max_iters = int(ag.get("max_align_iters", 30))
    stable_need = int(ag.get("stable_frames", 3))
    loop_hz = float(ag.get("loop_hz", 5.0))
    step_s = float(ag.get("align_step_s", 0.08))
    step_limit = float(ag.get("max_joint_step_ticks", 40))
    pan_gain = float(ag.get("pan_ticks_per_pixel", 0.35))
    reach_gain = float(ag.get("reach_ticks_per_pixel", 0.25))
    elbow_scale = float(ag.get("reach_elbow_scale", 0.8))
    lift_scale = float(ag.get("reach_lift_scale", -0.35))
    pan_limits = ag.get("pan_limits", [1600, 2500])
    reach_limits = ag.get("reach_limits", [-250, 250])

    base = dict(arm.goal) if arm.goal else arm.read()
    # 观察位先张开
    open_g = float(ag.get("gripper_open", base.get("gripper", 1200)))
    base["gripper"] = open_g
    arm.send_partial({"gripper": open_g}, settle_s=0.4)

    pan0 = float(base["shoulder_pan"])
    lift0 = float(base["shoulder_lift"])
    elbow0 = float(base["elbow_flex"])
    pan = pan0
    reach = 0.0
    pose = dict(base)
    stable = 0

    print("[align] 开始视觉伺服对准红块 ...")
    for it in range(max_iters):
        arm.hold_tick()
        detection, _ = vision.capture_detect(tag=f"align_{it:03d}")
        if detection is None:
            stable = 0
            print(f"[align {it}] no red block")
            time.sleep(1.0 / max(loop_hz, 0.5))
            continue

        err_u = float(detection.center_u) - desired_u
        err_v = float(detection.center_v) - desired_v
        print(f"[align {it}] err=({err_u:+.1f},{err_v:+.1f}) pan={pan:.0f} reach={reach:.0f}")

        if abs(err_u) <= dead_u and abs(err_v) <= dead_v:
            stable += 1
            if stable >= stable_need:
                print("[align] 对准稳定")
                return pose
        else:
            stable = 0
            d_pan = max(-step_limit, min(step_limit, -err_u * pan_gain))
            d_reach = max(-step_limit, min(step_limit, -err_v * reach_gain))
            pan = clamp(pan + d_pan, pan_limits)
            reach = clamp(reach + d_reach, reach_limits)
            pose = dict(base)
            pose["shoulder_pan"] = pan
            pose["shoulder_lift"] = clamp(lift0 + reach * lift_scale, joint_limits.get("shoulder_lift"))
            pose["elbow_flex"] = clamp(elbow0 + reach * elbow_scale, joint_limits.get("elbow_flex"))
            pose["gripper"] = open_g
            # 伺服环内用小步直写，避免每次完整 go_pose 过慢
            arm._write_goal(pose)
            time.sleep(max(step_s, 1.0 / max(loop_hz, 0.5)))
            arm.current_name = "grasp"

    print("[align] 超时，未对准")
    return None


def execute_grasp_sequence(
    arm: TicksArm,
    aligned: Dict[str, float],
    ag: dict,
    joint_limits: dict,
) -> Dict[str, float]:
    """对准后: pregrasp → insert → close → retreat。"""
    open_g = float(ag["gripper_open"])
    close_g = float(ag["gripper_close"])

    pose = dict(aligned)
    pose["gripper"] = open_g
    print("[grasp] pregrasp")
    pose = add_delta(pose, ag.get("pregrasp_delta") or {}, joint_limits)
    arm.go_pose("pregrasp", pose)
    time.sleep(float(ag.get("pregrasp_settle_s", 0.5)))

    print("[grasp] insert")
    pose = add_delta(pose, ag.get("insert_delta") or {}, joint_limits)
    pose["gripper"] = open_g
    arm.go_pose("insert", pose)
    time.sleep(float(ag.get("insert_settle_s", 0.8)))

    print(f"[grasp] close gripper -> {int(close_g)}")
    pose["gripper"] = close_g
    arm.send_partial({"gripper": close_g}, settle_s=float(ag.get("close_settle_s", 0.85)))

    print("[grasp] retreat")
    pose = add_delta(pose, ag.get("retreat_delta") or {}, joint_limits)
    pose["gripper"] = close_g
    arm.go_pose("retreat", pose)
    time.sleep(float(ag.get("retreat_settle_s", 0.8)))
    print("[grasp] 抓取序列完成（夹爪保持闭合）")
    return pose


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
        print("[auto] auto_grasp.enabled=false；用命令 auto 或 --auto-grasp 强制")
        return False
    if not vision.enabled:
        print("[auto] vision disabled，无法自动抓取")
        return False

    joint_limits = cfg["motion"].get("joint_limits") or {}

    # 确保在 grasp 观察位
    if arm.current_name != "grasp":
        gpose = dict(poses["grasp"])
        gpose["gripper"] = float(ag.get("gripper_open", gpose.get("gripper", 1200)))
        arm.go_pose("grasp", gpose)

    if vision.vision.get("detect_on_arrive", True):
        vision.capture_detect(tag="grasp_before_align")

    aligned = visual_servo_align(arm, vision, ag, joint_limits)
    if aligned is None:
        print("[auto] 对准失败，取消抓取")
        return False

    if not (ag.get("run_after_align", True) or force):
        print("[auto] 已对准；run_after_align=false，跳过抓取动作")
        arm.goal = aligned
        arm.current_name = "grasp"
        return True

    execute_grasp_sequence(arm, aligned, ag, joint_limits)

    if ag.get("go_place_after", True):
        place = dict(poses["place"])
        # 放置时张开
        place["gripper"] = float(ag.get("gripper_open", place.get("gripper", 1200)))
        # 先带着闭合姿态移到放置上方再张开更稳：先用闭爪到 place 关节，再开爪
        place_hold = dict(place)
        place_hold["gripper"] = float(ag.get("gripper_close", 2100))
        arm.go_pose("place", place_hold)
        arm.send_partial({"gripper": float(place["gripper"])}, settle_s=0.6)
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
        print("复制到 og.yaml 的 poses.*")
        return 0
    finally:
        arm.disconnect(release_torque=False)


def run_interactive(arm: TicksArm, poses: dict, vision: GraspVision, cfg: dict, auto_on_grasp: bool) -> int:
    arm.go_pose("initial", poses["initial"])
    while True:
        cmd = arm.hold_until_command()
        if cmd in {"q", "quit", "exit"}:
            print("退出")
            return 0
        if cmd in {"i", "init", "initial", "home"}:
            arm.go_pose("initial", poses["initial"])
            continue
        if cmd in {"g", "grasp"}:
            gpose = dict(poses["grasp"])
            ag = cfg.get("auto_grasp") or {}
            gpose["gripper"] = float(ag.get("gripper_open", gpose.get("gripper", 1200)))
            arm.go_pose("grasp", gpose)
            if vision.enabled and vision.vision.get("detect_on_arrive", True):
                vision.capture_detect(tag="grasp_arrive")
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
                vision.capture_detect(tag="grasp_manual")
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
    print("保持当前位姿；Ctrl+C 退出")
    try:
        while True:
            arm.hold_tick()
            time.sleep(1.0 / max(float(arm.motion["hold_hz"]), 0.5))
    except KeyboardInterrupt:
        print("\n用户中断")
        return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="SO-101 三位姿 + 红块自动抓取")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--read", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--sequence", action="store_true", help="initial→自动抓取→place→initial")
    parser.add_argument("--auto-grasp", action="store_true", help="交互模式下输入 grasp 后自动抓取")
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
    print(
        f"平滑: step_ticks={cfg['motion']['step_ticks']}, step_s={cfg['motion']['step_s']}"
    )
    print(
        f"视觉: enabled={vision.enabled}, camera={cfg['camera']['index_or_path']}"
    )
    print(
        f"自动抓取: enabled={ag.get('enabled')}, open={ag.get('gripper_open')}, "
        f"close={ag.get('gripper_close')}, go_place={ag.get('go_place_after')}"
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
        arm.disconnect(release_torque=bool(args.release_on_exit))
        if not args.release_on_exit:
            print("已断开串口；力矩未主动关闭（可用 --release-on-exit）")


if __name__ == "__main__":
    raise SystemExit(main())
