#!/usr/bin/env python3
"""Mobile-base alignment followed by SO-101 red-block grasp."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARM_DIR = ROOT / "arm"
for path in (HERE, ARM_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mobile_base import TimedMobileBase
from debug_images import DebugImages


def log(message: str) -> None:
    print(f"[mobile-grasp] {message}", flush=True)


def save_debug_camera_stage(debug: DebugImages, cfg: dict, tag: str, status: str) -> None:
    """Camera diagnostics must never interrupt the safety cleanup path."""
    try:
        from red_block_detector import RedBlockDetector, capture_frame

        detector = RedBlockDetector(cfg)
        frame = capture_frame(cfg["camera"])
        debug.save(tag, frame, detector, detector.detect(frame), status=status)
    except Exception as exc:
        log(f"debug image skipped at {tag}: {exc}")


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a YAML mapping")
    for key in ("intrinsics", "handeye"):
        value = cfg.get(key)
        if not value:
            continue
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            local = path.parent / candidate
            project = ROOT / candidate
            candidate = local if local.is_file() else project
        cfg[key] = str(candidate.resolve())
    return cfg


def align_base(base: TimedMobileBase, cfg: dict, debug: DebugImages) -> bool:
    from red_block_detector import RedBlockDetector, capture_frame

    align = cfg["base_alignment"]
    detector = RedBlockDetector(cfg)
    target_ratio = float(align["target_u_ratio"])
    previous_error_px: float | None = None
    no_response_count = 0
    saw_target_on_first_frame = False
    for index in range(1, int(align["max_iters"]) + 1):
        log(f"lateral alignment {index}/{align['max_iters']}: capture frame")
        frame = capture_frame(cfg["camera"])
        target_u = frame.shape[1] * target_ratio
        candidates = detector.detect_candidates(frame)
        det = detector.select_closest_to_u(candidates, target_u)
        if det is None:
            if index == 1:
                log("initial alignment frame has no red block; refuse to enter grasp stage")
                debug.save(f"align_{index:02d}_not_found", frame, detector, None,
                           target_u=target_u, tolerance_px=float(align["tolerance_px"]),
                           status="INITIAL RED BLOCK NOT FOUND", candidates=candidates)
                return False
            debug.save(f"align_{index:02d}_not_found", frame, detector, None,
                       target_u=target_u, tolerance_px=float(align["tolerance_px"]),
                       status="RED BLOCK NOT FOUND", candidates=candidates)
            print(f"[align] {index}: red block not found")
            continue
        saw_target_on_first_frame = saw_target_on_first_frame or index == 1
        error_px = float(det.center_u) - target_u
        within_tolerance = abs(error_px) <= float(align["tolerance_px"])
        debug.save(
            f"align_{index:02d}_{'aligned' if within_tolerance else 'before_strafe'}",
            frame, detector, det, target_u=target_u, tolerance_px=float(align["tolerance_px"]),
            status=f"candidates={len(candidates)}  error={error_px:+.1f}px  {'ALIGNED' if within_tolerance else 'MOVE BASE'}",
            candidates=candidates,
        )
        print(f"[align] {index}: candidates={len(candidates)}, selected_u={det.center_u:.1f}, "
              f"target={target_u:.1f}, error={error_px:+.1f}px")
        if within_tolerance:
            return True
        if previous_error_px is not None:
            changed_px = abs(error_px - previous_error_px)
            same_side = error_px * previous_error_px > 0.0
            if same_side and changed_px < float(align.get("min_response_px", 3.0)):
                no_response_count += 1
                print(f"[align] no visible lateral response: {changed_px:.1f}px "
                      f"({no_response_count}/{align.get('max_no_response_iters', 2)})")
                if no_response_count >= int(align.get("max_no_response_iters", 2)):
                    print("[align] lateral response is small; continue forward with the detected target")
                    return True
            else:
                no_response_count = 0
        # Keep the lateral speed fixed and scale only the pulse duration.
        # The command is blocking, so the next image is captured only after
        # this pulse has stopped and the configured settle time has elapsed.
        tolerance_px = float(align["tolerance_px"])
        max_error_px = max(float(align.get("max_error_px", 180.0)), tolerance_px)
        lateral_speed = float(align["lateral_speed_mps"])
        if lateral_speed <= 0.0:
            raise ValueError("base_alignment.lateral_speed_mps must be positive")
        min_duration_s = float(align["min_lateral_duration_s"])
        max_duration_s = float(align["max_lateral_duration_s"])
        if min_duration_s <= 0.0 or max_duration_s <= 0.0:
            raise ValueError("base_alignment lateral durations must be positive")
        if max_duration_s < min_duration_s:
            min_duration_s, max_duration_s = max_duration_s, min_duration_s
        if max_error_px == tolerance_px:
            error_ratio = 1.0
        else:
            error_ratio = min(1.0, (abs(error_px) - tolerance_px) / (max_error_px - tolerance_px))
        duration_s = min_duration_s + error_ratio * (max_duration_s - min_duration_s)
        direction_sign = float(align.get("lateral_direction_sign", 1.0))
        print(f"[align] pulse: speed={lateral_speed:.3f}m/s duration={duration_s:.3f}s")
        base.strafe_for_duration(error_px * direction_sign, lateral_speed, duration_s)
        base.pause(float(align["settle_s"]))
        previous_error_px = error_px
    log("lateral alignment reached iteration limit; continue with the detected target")
    return saw_target_on_first_frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Mobile alignment plus SO-101 grasp prototype")
    parser.add_argument("--config", type=Path, default=HERE / "configs" / "mobile_grasp.yaml")
    parser.add_argument("--yes", action="store_true", help="allow vehicle and arm motion")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    log(f"load config: {args.config.resolve()}")
    cfg = load_cfg(args.config.resolve())
    debug_cfg = cfg.get("debug") or {}
    debug_dir = ROOT / str(debug_cfg.get("directory", "grasp/debug"))
    debug = DebugImages(debug_dir, bool(debug_cfg.get("enabled", True)))
    debug.reset()
    if not args.yes and not args.dry_run:
        print("Refusing motion: pass --yes")
        return 2
    if args.dry_run:
        print("[dry-run] arm observe -> base +10cm -> lateral align -> base +10cm -> grasp -> base -20cm -> arm final")
        return 0
    log("use system Python ROS helper for cmd_vel")
    log("import arm and vision modules")
    from arm_grasp import move_to_pose, move_to_pose_no_read, run_visual_grasp
    from og import TicksArm

    base_cfg = cfg["mobile_base"]
    helper = ROOT / str(base_cfg.get("ros_cmd_helper", "grasp/ros_cmd_vel.py"))
    log(f"create cmd_vel helper: {helper}")
    base = TimedMobileBase(
        str(base_cfg["cmd_vel_topic"]), int(base_cfg["control_rate_hz"]),
        str(base_cfg.get("ros_python", "/usr/bin/python3")), helper,
    )
    log(f"create arm controller: port={cfg['port']} baud={cfg['baud']}")
    arm = TicksArm(str(cfg["port"]), int(cfg["baud"]), cfg["motion"])
    entered_m = 0.0
    grasped = False
    try:
        max_attempts = max(1, int(cfg["sequence"].get("connect_max_attempts", 5)))
        for attempt in range(1, max_attempts + 1):
            try:
                log(f"connect arm {attempt}/{max_attempts}")
                arm.connect()
                log("arm connected")
                break
            except (ConnectionError, OSError, RuntimeError) as exc:
                log(f"arm connect failed: {exc}")
                arm.disconnect(release_torque=False)
                if attempt == max_attempts:
                    raise
                time.sleep(float(cfg["sequence"].get("connect_retry_s", 0.8)))
        if bool(cfg["sequence"].get("move_to_initial", False)):
            log("move arm to initial")
            move_to_pose(arm, "initial", cfg)
        log("move arm to grasp observation pose")
        move_to_pose(arm, "grasp", cfg, float(cfg["grasp"]["gripper_open"]))
        log("base forward before lateral alignment")
        base.forward_m(float(base_cfg["approach_before_align_m"]), float(base_cfg["forward_speed_mps"]))
        entered_m += float(base_cfg["approach_before_align_m"])
        aligned = align_base(base, cfg, debug)
        log(f"lateral alignment result={aligned}")
        if not aligned and bool(cfg["base_alignment"].get("require_success", True)):
            raise RuntimeError("base lateral alignment failed; refuse to enter divider")
        log("base forward into divider")
        base.forward_m(float(base_cfg["approach_after_align_m"]), float(base_cfg["forward_speed_mps"]))
        entered_m += float(base_cfg["approach_after_align_m"])
        log("start arm visual grasp")
        grasped = run_visual_grasp(arm, cfg, ROOT, debug)
        try:
            g_ticks = float(arm.read().get("gripper", float("nan")))
            log(
                f"arm visual grasp completed: success={grasped} "
                f"gripper_ticks={g_ticks:.0f}"
            )
        except Exception as exc:
            log(
                f"arm visual grasp completed: success={grasped}; "
                f"read gripper failed: {exc}"
            )
        return 0 if grasped else 1
    finally:
        log("cleanup: stop base")
        base.stop()
        if entered_m > 0:
            log(f"cleanup: base retreat {entered_m:.3f}m")
            base.backward_m(entered_m, float(base_cfg["backward_speed_mps"]))
        if arm.bus is not None:
            save_debug_camera_stage(debug, cfg, "after_base_retreat_before_final",
                                    "BASE RETREATED; ARM FINAL NEXT")
            log("cleanup: move arm to final pose")
            # Never read or rewrite the gripper during cleanup: a held block
            # can make its Present_Position and Goal_Position report overload.
            move_to_pose_no_read(arm, "final", cfg)
            save_debug_camera_stage(debug, cfg, "after_arm_final", "ARM AT FINAL")
            arm.disconnect(release_torque=bool(cfg.get("release_torque_on_exit", False)))
            log("cleanup: arm disconnected")


if __name__ == "__main__":
    raise SystemExit(main())
