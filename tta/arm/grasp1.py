#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""grasp1：红色物块视觉抓取。

- 视觉识别 / 抓取闭环：同 reset_and_grasp → grasp_from_observe.py（红块检测）
- 观察位与抓取行程初始参数：取自 place_apriltag（见 grasp1.yaml）

流程：
  1. 到 poses.grasp（= place_apriltag 的 place_observe 臂姿，开爪）
  2. 调用 grasp_from_observe.py（同 reset_and_grasp）：
     红块检测 → 前伸 → 对准；对准失败则前探重试，
     最后一次仍失败则强制闭爪

用法:
  python3 grasp1.py --yes
  python3 grasp1.py --yes --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from og import TicksArm, load_config, resolve_poses  # noqa: E402

DEFAULT_CONFIG = _HERE / "grasp1.yaml"
DEFAULT_VISION_CONFIG = _HERE / "og.yaml"


def require_mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def add_value(command: list[str], flag: str, value: Any) -> None:
    command.extend((flag, str(value)))


def build_grasp_command(config_path: Path, cfg: dict) -> list[str]:
    """组装 grasp_from_observe.py（红块视觉抓取），配置同 reset_and_grasp。"""
    grasp = require_mapping(cfg.get("grasp"), "grasp")
    command = [
        sys.executable,
        str(_HERE / "grasp_from_observe.py"),
        "--yes",
        "--og-config",
        str(config_path),
        "--vision-config",
        str(DEFAULT_VISION_CONFIG),
    ]

    scalar_flags = {
        "ik_tol_m": "--ik-tol",
        "max_horiz_m": "--max-horiz-m",
        "route_points": "--route-points",
        "range_mode": "--range-mode",
        "forward_m": "--forward-m",
        "z_offset_m": "--z-offset",
        "left_m": "--left-m",
        "retry_forward_compensation_m": "--retry-forward-compensation-m",
        "retry_forward_route_points": "--retry-forward-route-points",
        "final_forward_probe_m": "--final-forward-probe-m",
        "final_forward_route_points": "--final-forward-route-points",
        "gripper_open": "--gripper-open",
        "gripper_close": "--gripper-close",
        "save_dir": "--save-dir",
        "preclose_iters": "--preclose-iters",
        "preclose_tol_u_px": "--preclose-tol-u",
        "preclose_tol_v_px": "--preclose-tol-v",
        "preclose_target_u_px": "--preclose-target-u",
        "preclose_target_v_px": "--preclose-target-v",
        "preclose_m_per_px": "--preclose-m-per-px",
        "preclose_max_step_m": "--preclose-max-step",
        "preclose_min_area_px": "--preclose-min-area",
        "preclose_max_area_px": "--preclose-max-area",
        "retry_open_settle_s": "--retry-open-settle-s",
        "preclose_settle_s": "--preclose-settle-s",
        "close_settle_s": "--close-settle-s",
        "miss_tol_ticks": "--grasp-miss-tol",
        "retries": "--grasp-retries",
    }
    for key, flag in scalar_flags.items():
        value = grasp.get(key)
        if value is not None:
            add_value(command, flag, value)

    child_source = (_HERE / "grasp_from_observe.py").read_text(encoding="utf-8")
    fail_forward = grasp.get("preclose_fail_forward_m")
    if fail_forward is not None:
        if "--preclose-fail-forward" in child_source:
            add_value(command, "--preclose-fail-forward", fail_forward)
        else:
            print("[warn] grasp_from_observe.py 未支持 --preclose-fail-forward")

    bool_flags = {
        "allow_partial_ik": "--allow-partial",
        "open_gripper_before_plan": "--open-gripper",
        "save_images": "--no-save-image",
        "preclose_check": "--no-preclose-check",
        "close_even_if_ungraspable": "--close-even-if-ungraspable",
        "return_to_observe_after_grasp": "--return-to-observe",
        "release_torque_on_exit": "--release-on-exit",
    }
    for key, flag in bool_flags.items():
        value = grasp.get(key)
        if value is None:
            continue
        if key in {"save_images", "preclose_check"}:
            if not bool(value):
                command.append(flag)
        elif bool(value):
            command.append(flag)
    return command


def move_to_observe(config_path: Path, cfg: dict, dry_run: bool) -> None:
    sequence = require_mapping(cfg.get("sequence"), "sequence")
    og_cfg = load_config(config_path)
    motion = require_mapping(og_cfg.get("motion"), "motion")
    poses = resolve_poses(og_cfg)

    if dry_run:
        print("[dry-run] 观察位运动跳过")
        return

    if "grasp" not in poses:
        raise KeyError("grasp1.yaml 缺少 poses.grasp（观察位）")

    arm = TicksArm(str(og_cfg["port"]), int(og_cfg["baud"]), motion)
    max_attempts = max(1, int(sequence.get("connect_max_attempts", 5)))
    retry_s = max(0.0, float(sequence.get("connect_retry_s", 0.8)))
    try:
        for attempt in range(1, max_attempts + 1):
            try:
                arm.connect()
                break
            except (ConnectionError, OSError, RuntimeError) as exc:
                arm.disconnect(release_torque=False)
                if attempt >= max_attempts:
                    print(f"[connect] {max_attempts} 次连接均失败，停止重试")
                    raise
                print(
                    f"[connect] 第 {attempt}/{max_attempts} 次握手失败: {exc}; "
                    f"{retry_s:.1f}s 后重试"
                )
                time.sleep(retry_s)

        joint_order = sequence.get("joint_order")
        if isinstance(joint_order, list):
            ordered_joints = [str(item) for item in joint_order]
            sequential = True
        elif joint_order in ("ascending", "id_ascending"):
            ordered_joints = list(
                og_cfg.get("joint_order")
                or (
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                )
            )
            sequential = True
        else:
            sequential = False

        if sequential:

            def move_ordered(name: str) -> None:
                target = poses[name]
                for joint in ordered_joints:
                    if joint not in target:
                        continue
                    current = arm.read()
                    one_joint_target = {
                        key: float(current.get(key, target[key])) for key in target
                    }
                    one_joint_target[joint] = float(target[joint])
                    arm.go_pose(f"{name}:{joint}", one_joint_target)

        else:

            def move_ordered(name: str) -> None:
                arm.go_pose(name, poses[name])

        if bool(sequence.get("move_to_initial", False)):
            move_ordered("initial")
            time.sleep(float(sequence.get("initial_hold_s", 0.0)))
        if bool(sequence.get("move_to_observe", True)):
            print("[grasp1] 前往观察位 poses.grasp（place_apriltag.place_observe 臂姿）")
            move_ordered("grasp")
            time.sleep(float(sequence.get("observe_hold_s", 0.0)))
    finally:
        arm.disconnect(release_torque=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="grasp1：观察位(place_observe) → 红块视觉抓取"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--yes", action="store_true", help="确认允许机械臂运动")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不运动")
    args = parser.parse_args()

    if not args.yes and not args.dry_run:
        print("Refusing motion: pass --yes")
        return 2

    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = require_mapping(yaml.safe_load(f) or {}, "config root")

    vision_cfg = load_config(DEFAULT_VISION_CONFIG)
    required = {
        "detector": cfg.get("detector") or vision_cfg.get("detector"),
        "camera": cfg.get("camera") or vision_cfg.get("camera"),
        "intrinsics": cfg.get("intrinsics") or vision_cfg.get("intrinsics"),
        "handeye": cfg.get("handeye") or vision_cfg.get("handeye"),
    }
    missing = [name for name, value in required.items() if value in (None, "")]
    if missing:
        print(
            "[config] 缺少抓取视觉配置: "
            + ", ".join(missing)
            + "; 请确认 grasp1.yaml 含 detector/camera/intrinsics/handeye"
        )
        return 2

    print(f"[config] {config_path}")
    move_to_observe(config_path, cfg, args.dry_run)

    command = build_grasp_command(config_path, cfg)
    print("[grasp1] launch:", " ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False, cwd=_HERE.parent).returncode


if __name__ == "__main__":
    raise SystemExit(main())
