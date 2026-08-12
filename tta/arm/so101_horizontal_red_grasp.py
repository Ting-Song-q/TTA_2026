#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""First SO-101 horizontal red-block grasp prototype.

Default mode is dry-run and never connects to the arm. The controller uses a
fixed observe pose, wrist-camera red detection, and bounded joint increments.
This is deliberately a calibration scaffold, not a production collision-safe
planner.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from so101_red_block_camera_test import (
    DEFAULT_CONFIG_PATH,
    RedBlockDetector,
    camera_source,
    capture_frame,
    load_config,
)


JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class SO101Adapter:
    """Small compatibility wrapper around LeRobot's SO-101 follower API."""

    def __init__(self, port: str, robot_id: str, max_relative_target_deg):
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
        kwargs = {"port": port, "id": robot_id}
        if max_relative_target_deg is not None:
            kwargs["max_relative_target"] = float(max_relative_target_deg)
        self._robot = SO101Follower(SO101FollowerConfig(**kwargs))

    def connect(self) -> None:
        # Calibration is a separate one-time procedure. Never trigger the
        # interactive range sweep from a grasp run.
        try:
            self._robot.connect(calibrate=False)
        except TypeError:
            self._robot.connect()

    def disconnect(self) -> None:
        self._robot.disconnect()

    def send(self, pose: Dict[str, float]) -> None:
        self._robot.send_action({f"{name}.pos": float(value) for name, value in pose.items()})

    def read(self) -> Dict[str, float]:
        raw = self._robot.get_observation()
        return {
            name: float(raw[f"{name}.pos"])
            for name in JOINTS
            if f"{name}.pos" in raw
        }


def bounded(value: float, limits) -> float:
    if limits is None:
        return float(value)
    return max(float(limits[0]), min(float(limits[1]), float(value)))


def apply_delta(pose: Dict[str, float], delta: Dict[str, float], cfg: dict) -> Dict[str, float]:
    result = dict(pose)
    for joint, amount in delta.items():
        if joint not in result:
            continue
        step_limit = float(cfg.get("max_joint_step_deg", 3.0))
        amount = max(-step_limit, min(step_limit, float(amount)))
        limits = cfg.get("joint_limits")
        result[joint] = bounded(amount + result[joint], limits.get(joint, [-180, 180]) if limits else None)
    return result


def add_pose_delta(pose: Dict[str, float], delta: Dict[str, float], cfg: dict) -> Dict[str, float]:
    result = dict(pose)
    for joint, amount in delta.items():
        if joint in result:
            result[joint] += float(amount)
            limits = cfg.get("joint_limits")
            result[joint] = bounded(result[joint], limits.get(joint, [-180, 180]) if limits else None)
    return result


def move_in_small_steps(
    arm: SO101Adapter,
    current: Dict[str, float],
    target: Dict[str, float],
    step_deg: float,
    step_s: float,
) -> Dict[str, float]:
    """Move toward a taught pose without issuing a large first command."""
    pose = dict(current)
    max_delta = max(abs(float(target[name]) - pose.get(name, float(target[name]))) for name in target)
    steps = max(1, int(np.ceil(max_delta / max(float(step_deg), 0.1))))
    for _ in range(steps):
        # Read back the real motor positions. LeRobot may clip a command, so
        # the next increment must be based on where the arm actually arrived.
        actual = arm.read()
        command = {}
        done = True
        for name, value in target.items():
            actual_value = float(actual.get(name, pose.get(name, float(value))))
            error = float(value) - actual_value
            if abs(error) > 0.2:
                done = False
            command[name] = actual_value + max(-float(step_deg), min(float(step_deg), error))
        if done:
            break
        arm.send(command)
        time.sleep(float(step_s))
    final = arm.read()
    print("actual pose after step sequence:")
    for name in target:
        print(f"  {name}: {final.get(name, float('nan')):.3f} (target {float(target[name]):.3f})")
    return {name: float(final.get(name, target[name])) for name in target}


def main() -> int:
    parser = argparse.ArgumentParser(description="SO-101 horizontal red-block grasp prototype")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--camera", help="Override camera source")
    parser.add_argument("--port", help="Override robot.port from YAML")
    parser.add_argument("--robot-id", help="Override robot.id from YAML")
    parser.add_argument("--observe-only", action="store_true", help="Move to observe pose, open gripper, then stop")
    parser.add_argument("--execute", action="store_true", help="Enable arm motion")
    parser.add_argument("--yes", action="store_true", help="Confirm real motion")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.camera is not None:
        config["camera"]["index_or_path"] = camera_source(args.camera)
    grasp = config["grasp"]
    robot_cfg = config["robot"]
    port = args.port or robot_cfg["port"]
    robot_id = args.robot_id or robot_cfg["id"]
    detector = RedBlockDetector(config)
    camera_cfg = config["camera"]
    observe_pose = {name: float(value) for name, value in grasp["observe_pose"].items()}
    reset_pose = {name: float(value) for name, value in grasp["reset_pose"].items()}
    pose = dict(observe_pose)
    arm = None
    if args.observe_only or args.execute:
        if not grasp.get("motion_enabled", False):
            print("Refusing motion: set grasp.motion_enabled=true after teaching safe poses.")
            return 2
        if not port or not args.yes:
            print("Refusing motion: set robot.port in YAML or pass --port, then include --yes.")
            return 2
        arm = SO101Adapter(port, robot_id, robot_cfg["max_relative_target_deg"])
        arm.connect()
        current_pose = arm.read()
        print("moving to reset_pose in small steps")
        reset_actual = move_in_small_steps(
            arm,
            current_pose,
            reset_pose,
            robot_cfg["reset_step_deg"],
            robot_cfg["reset_step_s"],
        )
        print("moving from reset_pose to observe_pose in small steps")
        pose = move_in_small_steps(
            arm,
            reset_actual,
            observe_pose,
            robot_cfg["reset_step_deg"],
            robot_cfg["reset_step_s"],
        )
        arm.send({"gripper": float(grasp["gripper_open"])})
        if args.observe_only:
            frame = capture_frame(camera_cfg)
            detection = detector.detect(frame)
            annotated = detector.draw(frame, detection)
            output_dir = Path(config["output"]["directory"])
            if not output_dir.is_absolute():
                output_dir = Path(__file__).resolve().parent.parent / output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "observe_pose.jpg"
            cv2.imwrite(str(path), annotated)
            print(f"observe pose reached; gripper opened; image saved to {path}")
            if detection:
                print(f"red block detected at ({detection.center_u:.1f}, {detection.center_v:.1f})")
            else:
                print("red block not detected in observe image")
            arm.disconnect()
            arm = None
            return 0

    output_dir = Path(config["output"]["directory"])
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    desired_u, desired_v = map(float, grasp["desired_center"])
    dead_u, dead_v = map(float, grasp["pixel_deadband"])
    stable = 0
    try:
        for iteration in range(int(grasp["max_align_iters"])):
            frame = capture_frame(camera_cfg)
            detection = detector.detect(frame)
            annotated = detector.draw(frame, detection)
            cv2.drawMarker(annotated, (round(desired_u), round(desired_v)), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
            cv2.imwrite(str(output_dir / f"align_{iteration:03d}.jpg"), annotated)
            if detection is None:
                stable = 0
                print(f"[{iteration}] red block not detected")
                time.sleep(1.0 / float(grasp["loop_hz"]))
                continue

            err_u = detection.center_u - desired_u
            err_v = detection.center_v - desired_v
            print(f"[{iteration}] center=({detection.center_u:.1f},{detection.center_v:.1f}) error=({err_u:+.1f},{err_v:+.1f})")
            if abs(err_u) <= dead_u and abs(err_v) <= dead_v:
                stable += 1
            else:
                stable = 0
                delta = {
                    "shoulder_pan": -err_u * float(grasp["pan_deg_per_pixel"]),
                    "shoulder_lift": -err_v * float(grasp["reach_deg_per_pixel"]) * 0.5,
                    "elbow_flex": err_v * float(grasp["reach_deg_per_pixel"]),
                }
                pose = apply_delta(pose, delta, grasp)
                if arm:
                    arm.send(pose)
            if stable >= int(grasp["stable_frames"]):
                print("alignment stable")
                break
            time.sleep(1.0 / float(grasp["loop_hz"]))
        else:
            print("alignment timeout; refusing grasp")
            return 1

        pose = add_pose_delta(pose, grasp["pregrasp_delta"], grasp)
        if arm:
            arm.send(pose)
            time.sleep(0.4)
        pose = add_pose_delta(pose, grasp["insert_delta"], grasp)
        if arm:
            arm.send(pose)
            time.sleep(float(grasp["insert_settle_s"]))
            pose["gripper"] = float(grasp["gripper_close"])
            arm.send({"gripper": pose["gripper"]})
            time.sleep(0.8)
        else:
            print("dry-run insert pose:", pose)
            print("dry-run close gripper")
        pose = add_pose_delta(pose, grasp["retreat_delta"], grasp)
        if arm:
            arm.send(pose)
            time.sleep(float(grasp["retreat_settle_s"]))
        print("grasp sequence finished" if arm else "dry-run sequence finished")
        return 0
    finally:
        if arm:
            arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
