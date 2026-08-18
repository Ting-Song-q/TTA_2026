#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 三位姿切换（Feetech 原始 ticks）。

位姿:
  initial  初始
  grasp    抓取/观察
  place    放置

用法:
  python3 og.py --yes
  # initial | grasp | place | q
  python3 og.py --yes --sequence
  python3 og.py --read
"""

from __future__ import annotations

import argparse
import copy
import select
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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
            "gripper": [700, 3300],
        },
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

    def connect(self, motor_names=None) -> None:
        self.bus = connect_bus(
            self.port,
            baud=self.baud,
            configure=False,
            motor_names=motor_names,
        )
        # 只给实际握手到的舵机上力矩
        present = None
        try:
            present = list(self.bus.motors.keys())
        except Exception:
            present = motor_names
        enable_torque_safe(self.bus, present)
        time.sleep(0.15)
        print("[torque] ON")

    def connect_with_retry(
        self,
        max_attempts: int = 5,
        retry_s: float = 0.8,
        motor_names=None,
        allow_missing_gripper: bool = False,
    ) -> None:
        """串口/舵机握手失败时多次重试（常见于夹爪 id=6 过载掉线后恢复）。

        allow_missing_gripper=True 时：全量握手失败且仅缺 id=6，则降级只连 1–5。
        """
        max_attempts = max(1, int(max_attempts))
        retry_s = max(0.0, float(retry_s))
        last_exc: Optional[BaseException] = None
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"[connect] 尝试 {attempt}/{max_attempts} …")
                self.connect(motor_names=motor_names)
                if attempt > 1:
                    print(f"[connect] 第 {attempt} 次握手成功")
                return
            except KeyboardInterrupt:
                self.disconnect(release_torque=False)
                raise
            except (ConnectionError, OSError, RuntimeError, TimeoutError) as exc:
                last_exc = exc
                self.disconnect(release_torque=False)
                err = str(exc)
                only_gripper_missing = (
                    "Missing motor" in err and "- 6" in err
                )
                if (
                    allow_missing_gripper
                    and only_gripper_missing
                    and motor_names is None
                ):
                    print(
                        "[connect] 夹爪 id=6 掉线；降级握手舵机 1–5（保持持块力矩）",
                        flush=True,
                    )
                    arm_only = [
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow_flex",
                        "wrist_flex",
                        "wrist_roll",
                    ]
                    try:
                        self.connect(motor_names=arm_only)
                        print("[connect] 降级握手成功（无夹爪）", flush=True)
                        return
                    except KeyboardInterrupt:
                        self.disconnect(release_torque=False)
                        raise
                    except (ConnectionError, OSError, RuntimeError, TimeoutError) as exc2:
                        last_exc = exc2
                        self.disconnect(release_torque=False)
                if attempt >= max_attempts:
                    break
                print(
                    f"[connect] 第 {attempt}/{max_attempts} 次握手失败: {exc}; "
                    f"{retry_s:.1f}s 后重试（掉线舵机若已上电复位可自动恢复）"
                )
                try:
                    time.sleep(retry_s)
                except KeyboardInterrupt:
                    raise        print(f"[connect] {max_attempts} 次连接均失败，停止重试")
        assert last_exc is not None
        raise last_exc

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
        max_track_err = max(
            abs(float(final.get(j, g)) - float(g)) for j, g in target.items()
        )
        if max_track_err > max(tol * 4.0, 80.0):
            print(f"[warn] 到位偏差过大 ({max_track_err:.0f} ticks)，仍持设定目标")
        self.goal = dict(target)
        self.current_name = name
        print(f"[holding] 位姿={name}，力矩保持中")
        return final

    def hold_until_command(self) -> str:
        period = 1.0 / max(float(self.motion["hold_hz"]), 0.5)
        label = self.current_name or "custom"
        print(f"[holding {label}] 命令: initial | grasp | place | q")
        while True:
            self.hold_tick()
            if _stdin_ready(period):
                line = sys.stdin.readline()
                if not line:
                    continue
                cmd = line.strip().lower()
                if cmd:
                    return cmd


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


def run_interactive(arm: TicksArm, poses: dict) -> int:
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
            arm.go_pose("grasp", poses["grasp"])
            continue
        if cmd in {"p", "place"}:
            arm.go_pose("place", poses["place"])
            continue
        print(f"未知命令: {cmd!r}")


def run_sequence(arm: TicksArm, poses: dict, hold_s: float) -> int:
    """按 initial → grasp → place → initial 走一遍，便于标定检查。"""
    for name in ("initial", "grasp", "place", "initial"):
        arm.go_pose(name, poses[name])
        if hold_s > 0:
            time.sleep(hold_s)
    print("sequence done；保持当前位姿；Ctrl+C 退出")
    try:
        while True:
            arm.hold_tick()
            time.sleep(1.0 / max(float(arm.motion["hold_hz"]), 0.5))
    except KeyboardInterrupt:
        print("\n用户中断")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SO-101 三位姿切换（ticks）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--read", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--sequence", action="store_true", help="initial→grasp→place→initial")
    parser.add_argument("--hold-s", type=float, default=1.0)
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

    port = args.port or cfg["port"]
    baud = int(args.baud or cfg["baud"])
    poses = resolve_poses(cfg)

    for name in POSE_NAMES:
        print_pose(f"位姿[{name}]:", poses[name])
    print(
        f"平滑: step_ticks={cfg['motion']['step_ticks']}, step_s={cfg['motion']['step_s']}"
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
            return run_sequence(arm, poses, float(args.hold_s))
        return run_interactive(arm, poses)
    except KeyboardInterrupt:
        print("\n用户中断")
        return 130
    finally:
        arm.disconnect(release_torque=bool(args.release_on_exit))
        if not args.release_on_exit:
            print("已断开串口；力矩未主动关闭（可用 --release-on-exit）")


if __name__ == "__main__":
    raise SystemExit(main())
