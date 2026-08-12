#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101：运行后进入初始观察位姿，并保持该位姿直到切换到下一位姿。

参考 so101_horizontal_red_grasp(1).py 的 LeRobot 适配与配置加载方式。
默认从 horizontal_red_grasp_config.yaml 的 grasp.observe_pose 读取初始位姿。

用法（EVA）:
  python3 arm.py --port /dev/ttyACM1 --read          # 只读取当前位姿
  python3 arm.py --port /dev/ttyACM1 --yes
  # 到达后持续维持初始位；终端输入 next 进入下一位姿，输入 q 退出

注意:
  默认 connect(calibrate=False)，不会再进入标定向导。
  若尚未标定，先跑 lerobot-calibrate，并用相同的 --robot-id。
"""

from __future__ import annotations

import argparse
import select
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from so101_red_block_camera_test import DEFAULT_CONFIG_PATH, load_config


JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

DEFAULT_INITIAL_POSE: Dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 50.0,
}

DEFAULT_GRIPPER_OPEN = 50.0
DEFAULT_SETTLE_S = 1.2
DEFAULT_HOLD_HZ = 10.0


class SO101Adapter:
    """Small compatibility wrapper around LeRobot's SO-101 follower API."""

    def __init__(self, port: str, robot_id: Optional[str]):
        try:
            from lerobot.robots.so_follower import SOFollower as SO101Follower
            from lerobot.robots.so_follower import SOFollowerRobotConfig as SO101FollowerConfig
        except ImportError:
            try:
                from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
            except ImportError as exc:
                raise RuntimeError(
                    "Cannot import SO-101 LeRobot driver. Install the hardware extras "
                    "and run this script inside the LeRobot environment."
                ) from exc
        cfg_kwargs = {
            "port": port,
            "disable_torque_on_disconnect": False,
        }
        if robot_id:
            cfg_kwargs["id"] = robot_id
        self._robot = SO101Follower(SO101FollowerConfig(**cfg_kwargs))
        self._pose: Dict[str, float] = {}

    def connect(self, calibrate: bool = False) -> None:
        self._robot.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        self._robot.disconnect()

    def send(self, pose: Dict[str, float]) -> None:
        self._pose = {name: float(value) for name, value in pose.items()}
        self._robot.send_action(
            {f"{name}.pos": float(value) for name, value in self._pose.items()}
        )

    def read_pose(self) -> Dict[str, float]:
        """读取当前关节位姿（标定后的角度 / 夹爪量）。"""
        obs = self._robot.get_observation()
        pose: Dict[str, float] = {}
        for name in JOINTS:
            key = f"{name}.pos"
            if key in obs:
                pose[name] = float(obs[key])
        return pose

    def hold_tick(self) -> None:
        if self._pose:
            self.send(self._pose)


def resolve_pose_dict(raw: dict, fallback: Dict[str, float]) -> Dict[str, float]:
    return {name: float(raw.get(name, fallback[name])) for name in JOINTS}


def resolve_poses(config: dict) -> tuple[Dict[str, float], Dict[str, float], float]:
    grasp = config.get("grasp") if isinstance(config.get("grasp"), dict) else {}
    initial = resolve_pose_dict(grasp.get("observe_pose") or DEFAULT_INITIAL_POSE, DEFAULT_INITIAL_POSE)

    if isinstance(grasp.get("next_pose"), dict):
        nxt = resolve_pose_dict(grasp["next_pose"], initial)
    elif isinstance(grasp.get("pregrasp_delta"), dict):
        nxt = dict(initial)
        for name, delta in grasp["pregrasp_delta"].items():
            if name in nxt:
                nxt[name] = float(nxt[name]) + float(delta)
    else:
        nxt = dict(initial)

    gripper_open = float(grasp.get("gripper_open", initial.get("gripper", DEFAULT_GRIPPER_OPEN)))
    initial["gripper"] = gripper_open
    return initial, nxt, gripper_open


def print_pose(pose: Dict[str, float], title: str = "当前位姿") -> None:
    print(title)
    for name in JOINTS:
        if name in pose:
            print(f"  {name}: {pose[name]:.3f}")
    print("可粘贴到 YAML 的 observe_pose:")
    print("  observe_pose:")
    for name in JOINTS:
        if name in pose:
            print(f"    {name}: {pose[name]:.3f}")


def _stdin_line_ready(timeout_s: float) -> bool:
    if not sys.stdin.isatty():
        time.sleep(timeout_s)
        return False
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        return bool(ready)
    except (OSError, ValueError):
        time.sleep(timeout_s)
        return False


def hold_until_command(
    arm: SO101Adapter,
    pose: Dict[str, float],
    hold_hz: float,
    prompt: str,
) -> str:
    period = 1.0 / max(hold_hz, 0.1)
    print(prompt)
    while True:
        arm.hold_tick()
        if _stdin_line_ready(period):
            line = sys.stdin.readline()
            if line == "":
                continue
            cmd = line.strip().lower()
            if cmd:
                return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="SO-101 进入并保持初始位姿 / 读取当前位姿")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--port", required=True, help="从臂串口，例如 /dev/ttyACM1 或 COM10")
    parser.add_argument("--robot-id", default=None, help="与 lerobot-calibrate 时相同的 id")
    parser.add_argument("--calibrate", action="store_true", help="连接时允许进入标定（默认跳过）")
    parser.add_argument("--read", action="store_true", help="只读取并打印当前位姿，不发送运动指令")
    parser.add_argument("--yes", action="store_true", help="确认真实运动（必须带上才会动臂）")
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S)
    parser.add_argument("--hold-hz", type=float, default=DEFAULT_HOLD_HZ)
    parser.add_argument("--no-open-gripper", action="store_true")
    parser.add_argument("--exit-after-settle", action="store_true")
    parser.add_argument("--auto-next", action="store_true")
    parser.add_argument("--auto-next-after", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.read:
        arm = SO101Adapter(args.port, args.robot_id)
        try:
            print(f"连接 {args.port} (calibrate={args.calibrate}) ...")
            arm.connect(calibrate=args.calibrate)
            pose = arm.read_pose()
            if not pose:
                print("未读到关节位置（检查标定 / 串口 / 舵机供电）")
                return 1
            print_pose(pose)
            return 0
        finally:
            try:
                arm.disconnect()
            except Exception:
                pass
            print("已断开连接")

    config = load_config(args.config)
    initial_pose, next_pose, gripper_open = resolve_poses(config)
    if args.no_open_gripper:
        grasp = config.get("grasp") if isinstance(config.get("grasp"), dict) else {}
        raw = grasp.get("observe_pose") or DEFAULT_INITIAL_POSE
        initial_pose["gripper"] = float(raw.get("gripper", DEFAULT_GRIPPER_OPEN))

    print("初始位姿 (observe_pose):")
    for name in JOINTS:
        print(f"  {name}: {initial_pose[name]}")
    print("下一位姿 (next_pose / observe+pregrasp_delta):")
    for name in JOINTS:
        print(f"  {name}: {next_pose[name]}")

    if args.dry_run:
        print("dry-run：未连接机械臂")
        return 0

    if not args.yes:
        print("Refusing motion: pass --yes to confirm real movement.")
        print("示例: python3 arm.py --port /dev/ttyACM1 --yes")
        print("只读位姿: python3 arm.py --port /dev/ttyACM1 --read")
        return 2

    arm = SO101Adapter(args.port, args.robot_id)
    try:
        print(f"连接 {args.port} (calibrate={args.calibrate}) ...")
        arm.connect(calibrate=args.calibrate)

        print("发送初始位姿 ...")
        arm.send(initial_pose)
        time.sleep(max(0.0, float(args.settle)))
        print("已到达初始位姿，开始保持（力矩维持中）")
        print_pose(arm.read_pose(), "到达后实测位姿")

        if args.exit_after_settle:
            print("已按 --exit-after-settle 退出（不再保持）")
            return 0

        if args.auto_next:
            deadline = time.monotonic() + max(0.0, float(args.auto_next_after))
            period = 1.0 / max(float(args.hold_hz), 0.1)
            print(f"将在 {args.auto_next_after:.1f}s 后自动进入下一位姿")
            while time.monotonic() < deadline:
                arm.hold_tick()
                time.sleep(period)
            print("进入下一位姿 ...")
            arm.send(next_pose)
            time.sleep(max(0.0, float(args.settle)))
            print("已到达下一位姿，继续保持；输入 q 退出，next 可再次下发下一位姿")
        else:
            print("保持初始位姿中。命令: next / hold / read / q")

        current = "initial"
        while True:
            holding = initial_pose if current == "initial" else next_pose
            prompt = f"[holding {current}] 输入 next / hold / read / q :"
            arm.send(holding)
            cmd = hold_until_command(arm, holding, float(args.hold_hz), prompt)
            if cmd in {"q", "quit", "exit"}:
                print("退出保持")
                break
            if cmd in {"read", "r", "pos"}:
                print_pose(arm.read_pose())
                continue
            if cmd in {"next", "n"}:
                print("切换到下一位姿并保持 ...")
                arm.send(next_pose)
                time.sleep(max(0.0, float(args.settle)))
                current = "next"
                continue
            if cmd in {"hold", "initial", "home", "h"}:
                print("回到初始位姿并保持 ...")
                arm.send(initial_pose)
                time.sleep(max(0.0, float(args.settle)))
                current = "initial"
                continue
            print(f"未知命令: {cmd!r}（可用 next / hold / read / q）")

        return 0
    except KeyboardInterrupt:
        print("\n用户中断（Ctrl+C）")
        return 130
    finally:
        try:
            arm.disconnect()
        except Exception:
            pass
        print("已断开连接（disable_torque_on_disconnect=False，舵机可能仍保持力矩）")


if __name__ == "__main__":
    raise SystemExit(main())
