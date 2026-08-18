"""Arm-only visual grasp stage for the mobile grasp prototype."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

GRASP_DIR = Path(__file__).resolve().parent
ARM_DIR = GRASP_DIR.parent / "arm"
for path in (GRASP_DIR, ARM_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grasp_3d import full_pose_from_ik, go_pose_strict, ik_route
from grasp_from_observe import (
    advance_along_forward,
    build_forward_route_xyz,
    go_pose_if_changed,
    offset_grasp_left,
    refine_pose_until_graspable,
    save_detection_image,
    set_gripper,
)
from og import JOINTS, clamp_pose, pose_i
from start import move_to
from pixel_to_base import anchor_grasp_forward, grasp_z, load_handeye, load_intrinsics, pixel_to_base
from so101_fk import JOINT_ORDER, SO101FK
from so101_ik import SO101IK
from red_block_detector import RedBlockDetector, capture_frame


def _gripper_holding(
    actual_g: float, g_close: float, g_open: float, miss_tol: float
) -> bool:
    """未抓到：贴近闭爪标定(800±tol)或开爪标定(1200±tol)；其余视为已抓到。"""
    actual = float(actual_g)
    tol = float(miss_tol)
    if abs(actual - float(g_close)) <= tol:
        return False
    if abs(actual - float(g_open)) <= tol:
        return False
    return True


def _log_hold(
    actual_g: float, g_close: float, g_open: float, miss_tol: float, tag: str
) -> bool:
    holding = _gripper_holding(actual_g, g_close, g_open, miss_tol)
    print(
        f"[grasp] {tag}: gripper={actual_g:.0f} "
        f"(empty if in close {g_close:.0f}±{miss_tol:.0f} or "
        f"open {g_open:.0f}±{miss_tol:.0f}) "
        f"→ {'HOLDING' if holding else 'EMPTY'}",
        flush=True,
    )
    return holding


def _print_gripper_ticks(arm, label: str, fallback: float | None = None) -> float:
    """读并打印当前夹爪 Present_Position ticks。"""
    try:
        actual = float(arm.read().get("gripper", fallback if fallback is not None else 0.0))
    except Exception as exc:
        print(f"[grasp] {label}: 读取夹爪 ticks 失败: {exc}", flush=True)
        return float(fallback if fallback is not None else float("nan"))
    print(f"[grasp] {label}: gripper_ticks={actual:.0f}", flush=True)
    return actual


def _sync_hold_gripper_goal(arm, actual_g: float, limits: dict) -> float:
    """成功持块后把 Goal_Position 同步到当前实际 ticks，避免继续硬顶 g_close。"""
    g = float(actual_g)
    if g != g:  # NaN
        return g
    try:
        limited = clamp_pose({"gripper": g}, limits)
        g = float(limited["gripper"])
        move_to(arm.bus, pose_i({"gripper": g}), wait_s=0.0)
        if arm.goal is not None:
            arm.goal["gripper"] = g
        print(f"[grasp] sync hold goal: gripper -> {g:.0f}", flush=True)
    except Exception as exc:
        print(f"[grasp] sync hold goal write failed; software goal only: {exc}", flush=True)
        if arm.goal is not None:
            arm.goal["gripper"] = g
    return g


def run_visual_grasp(arm, cfg: dict, project_root: Path, debug=None) -> bool:
    """Run the existing forward-only grasp behavior and return after closing."""
    print("[grasp] load calibration and initialize FK/IK", flush=True)
    grasp = cfg["grasp"]
    motion = cfg["motion"]
    limits = motion["joint_limits"]
    camera_cfg = dict(cfg["camera"])
    poses = cfg["poses"]
    scene = cfg["scene"]
    g_open = float(grasp["gripper_open"])
    g_close = float(grasp["gripper_close"])
    miss_tol = float(grasp.get("miss_tol_ticks", 50))
    _, grasp_z_m, _ = grasp_z(scene)
    intrinsics = project_root / cfg["intrinsics"]
    handeye = project_root / cfg["handeye"]
    K, dist = load_intrinsics(intrinsics)
    T_ee_cam = load_handeye(handeye)
    detector = RedBlockDetector(cfg)
    fk = SO101FK.from_config(cfg) if isinstance(cfg.get("fk"), dict) else SO101FK()
    ik = SO101IK(fk)
    save_dir = project_root / grasp["save_dir"]
    observe_arm = {joint: float(poses["grasp"][joint]) for joint in JOINT_ORDER}
    retries = max(1, int(grasp["retries"]))
    print(f"[grasp] ready: retries={retries} camera={camera_cfg['index_or_path']}", flush=True)

    for attempt in range(1, retries + 1):
        try:
            pre = float(arm.read().get("gripper", g_open))
        except Exception:
            pre = g_open
        if _log_hold(pre, g_close, g_open, miss_tol, f"attempt{attempt}_precheck"):
            print("[grasp] already holding before open; treat as success", flush=True)
            _sync_hold_gripper_goal(arm, pre, limits)
            return True

        print(f"[grasp] attempt {attempt}/{retries}: open gripper and capture", flush=True)
        set_gripper(arm, g_open, limits, float(grasp["retry_open_settle_s"]))
        ticks = arm.read()
        seed = {joint: float(ticks[joint]) for joint in JOINT_ORDER}
        T_base_ee = fk.forward_ticks(seed)
        current_xyz = T_base_ee[:3, 3].copy()
        frame = capture_frame(camera_cfg)
        det = detector.detect(frame)
        if debug is not None:
            debug.save(f"grasp_{attempt}_initial_detection", frame, detector, det,
                       status="GRASP INITIAL DETECTION")
        if debug is None and bool(grasp.get("save_images", True)):
            save_detection_image(save_dir, frame, detector, det, tag=f"grasp_{attempt}")
        if det is None:
            print("[grasp] red block not found", flush=True)
            if attempt == retries:
                print("[grasp] final no-detect: close and check gripper", flush=True)
                set_gripper(arm, g_close, limits, float(grasp["close_settle_s"]))
                actual = _print_gripper_ticks(arm, "after_close(final_no_detect)", g_close)
                if _log_hold(actual, g_close, g_open, miss_tol, "final_no_detect_close"):
                    print(
                        f"[grasp] object detected in gripper (no vision); "
                        f"gripper_ticks={actual:.0f}",
                        flush=True,
                    )
                    _sync_hold_gripper_goal(arm, actual, limits)
                    return True
            continue

        vision = pixel_to_base(float(det.center_u), float(det.center_v), T_base_ee, T_ee_cam, K, dist, float(scene["table_z_m"]), np.asarray(scene["table_normal"], dtype=float), float(grasp_z_m))
        point = vision["p_grasp"]
        if grasp["range_mode"] == "forward":
            point = anchor_grasp_forward(current_xyz, point, float(grasp["forward_m"]), float(grasp_z_m))["p_grasp"]
        point = offset_grasp_left(current_xyz, point, float(grasp["left_m"]))
        target = np.array([point[0], point[1], current_xyz[2] + float(grasp["z_offset_m"])])
        forward_xy = target[:2] - current_xyz[:2]
        route = build_forward_route_xyz(current_xyz, target, int(grasp["route_points"]))
        print(f"[grasp] plan {len(route)} IK waypoints", flush=True)
        try:
            route_ticks = ik_route(
                ik, route, seed, observe_arm, current_xyz, limits,
                float(grasp["ik_tol_m"]), bool(grasp["allow_partial_ik"]),
                float(grasp["max_horiz_m"]),
            )
        except RuntimeError as exc:
            print(f"[grasp] route failed: {exc}")
            if attempt < retries:
                continue
            set_gripper(arm, g_close, limits, float(grasp["close_settle_s"]))
            actual = _print_gripper_ticks(arm, "after_close(route_fail)", g_close)
            if _log_hold(actual, g_close, g_open, miss_tol, "route_fail_close"):
                print(
                    f"[grasp] object detected after route fail; "
                    f"gripper_ticks={actual:.0f}",
                    flush=True,
                )
                _sync_hold_gripper_goal(arm, actual, limits)
                return True
            return False
        for index, arm_ticks in enumerate(route_ticks):
            print(f"[grasp] execute waypoint {index + 1}/{len(route_ticks)}", flush=True)
            go_pose_if_changed(arm, f"grasp_{attempt}_wp{index + 1}", full_pose_from_ik(arm_ticks, g_open, ticks))
            time.sleep(0.01)
        frame = capture_frame(camera_cfg)
        det = detector.detect(frame)
        if debug is not None:
            debug.save(f"grasp_{attempt}_at_planned_pose", frame, detector, det,
                       status="AT PLANNED GRASP POSE")

        aligned = True
        if bool(grasp.get("preclose_check", True)):
            print("[grasp] start preclose visual refinement", flush=True)
            aligned = refine_pose_until_graspable(
                arm=arm, fk=fk, ik=ik, detector=detector, camera_cfg=camera_cfg, T_ee_cam=T_ee_cam,
                limits=limits, g_open=g_open, observe_arm=observe_arm, save_dir=save_dir,
                no_save=debug is not None or not bool(grasp.get("save_images", True)),
                ik_tol=float(grasp["ik_tol_m"]), max_horiz_m=float(grasp["max_horiz_m"]),
                z_hold=float(fk.forward_ticks({j: float(arm.read()[j]) for j in JOINT_ORDER})[2, 3]),
                target_u=grasp.get("preclose_target_u_px"), target_v=grasp.get("preclose_target_v_px"),
                tol_u=float(grasp["preclose_tol_u_px"]), tol_v=float(grasp["preclose_tol_v_px"]),
                min_area=float(grasp["preclose_min_area_px"]), max_area=float(grasp["preclose_max_area_px"]),
                m_per_px=float(grasp["preclose_m_per_px"]), max_step_m=float(grasp["preclose_max_step_m"]),
                max_iters=int(grasp["preclose_iters"]), settle_s=float(grasp["preclose_settle_s"]),
            )
        frame = capture_frame(camera_cfg)
        det = detector.detect(frame)
        if debug is not None:
            debug.save(f"grasp_{attempt}_after_preclose", frame, detector, det,
                       status=f"PRECLOSE {'ALIGNED' if aligned else 'NOT ALIGNED'}")
        if not aligned and not bool(grasp.get("close_even_if_ungraspable", False)):
            try:
                advance_along_forward(
                    arm=arm, fk=fk, ik=ik, limits=limits, g_open=g_open,
                    observe_arm=observe_arm, forward_xy=forward_xy,
                    distance_m=float(grasp["preclose_fail_forward_m"]), route_points=2,
                    ik_tol=float(grasp["ik_tol_m"]), max_horiz_m=float(grasp["max_horiz_m"]),
                    label=f"grasp_{attempt}_recover",
                )
            except RuntimeError as exc:
                print(f"[grasp] preclose recovery failed: {exc}")
            if attempt < retries:
                # 未对准也先闭爪检查，避免已扫到块却因丢检重试开爪丢物
                set_gripper(arm, g_close, limits, float(grasp["close_settle_s"]))
                actual = _print_gripper_ticks(
                    arm, f"after_close(attempt{attempt}_recover)", g_close
                )
                if _log_hold(actual, g_close, g_open, miss_tol, f"attempt{attempt}_recover_close"):
                    print(
                        f"[grasp] object detected after recover close; "
                        f"gripper_ticks={actual:.0f}",
                        flush=True,
                    )
                    _sync_hold_gripper_goal(arm, actual, limits)
                    return True
                print("[grasp] recover close empty; reopen and retry", flush=True)
                set_gripper(arm, g_open, limits, float(grasp["retry_open_settle_s"]))
                _print_gripper_ticks(arm, f"after_reopen(attempt{attempt})", g_open)
                continue
            print("[grasp] final attempt: force close")
        elif not aligned:
            print("[grasp] unaligned: close_even_if_ungraspable enabled")

        try:
            advance_along_forward(
                arm=arm, fk=fk, ik=ik, limits=limits, g_open=g_open,
                observe_arm=observe_arm, forward_xy=forward_xy,
                distance_m=float(grasp["final_forward_probe_m"]),
                route_points=int(grasp["final_forward_route_points"]),
                ik_tol=float(grasp["ik_tol_m"]), max_horiz_m=float(grasp["max_horiz_m"]),
                label=f"grasp_{attempt}_probe",
            )
        except RuntimeError as exc:
            print(f"[grasp] final probe failed; close at current pose: {exc}")
        set_gripper(arm, g_close, limits, float(grasp["close_settle_s"]))
        frame = capture_frame(camera_cfg)
        det = detector.detect(frame)
        if debug is not None:
            debug.save(f"grasp_{attempt}_after_close", frame, detector, det,
                       status="GRIPPER CLOSED")
        print("[grasp] gripper close command sent", flush=True)
        actual = _print_gripper_ticks(arm, f"after_close(attempt{attempt})", g_close)
        if _log_hold(actual, g_close, g_open, miss_tol, f"attempt{attempt}_close"):
            print(
                f"[grasp] object detected in gripper; gripper_ticks={actual:.0f}",
                flush=True,
            )
            _sync_hold_gripper_goal(arm, actual, limits)
            return True
        print(f"[grasp] empty close; gripper_ticks={actual:.0f}", flush=True)
        set_gripper(arm, g_open, limits, float(grasp["retry_open_settle_s"]))
        _print_gripper_ticks(arm, f"after_reopen(attempt{attempt})", g_open)
        try:
            advance_along_forward(
                arm=arm, fk=fk, ik=ik, limits=limits, g_open=g_open, observe_arm=observe_arm,
                forward_xy=forward_xy, distance_m=float(grasp["retry_forward_compensation_m"]),
                route_points=int(grasp["retry_forward_route_points"]), ik_tol=float(grasp["ik_tol_m"]),
                max_horiz_m=float(grasp["max_horiz_m"]), label=f"grasp_{attempt}_retry",
            )
        except RuntimeError as exc:
            print(f"[grasp] retry compensation failed: {exc}")

    try:
        final_g = _print_gripper_ticks(arm, "after_grasp_exit", g_close)
    except Exception:
        final_g = g_close
    if _log_hold(final_g, g_close, g_open, miss_tol, "final_exit"):
        print(
            f"[grasp] holding at exit; gripper_ticks={final_g:.0f}",
            flush=True,
        )
        _sync_hold_gripper_goal(arm, final_g, limits)
        return True
    print(f"[grasp] grasp failed; final gripper_ticks={final_g:.0f}", flush=True)
    return False


def move_to_pose(arm, name: str, cfg: dict, gripper: float | None = None) -> None:
    target = {joint: float(cfg["poses"][name][joint]) for joint in JOINTS}
    if gripper is not None:
        target["gripper"] = float(gripper)
    go_pose_strict(arm, name, target)


def move_to_pose_no_read(arm, name: str, cfg: dict, gripper: float | None = None) -> None:
    """Move without reading the servo; omit gripper to preserve a held object."""
    assert arm.bus is not None
    target = {
        joint: float(cfg["poses"][name][joint])
        for joint in JOINTS
        if gripper is not None or joint != "gripper"
    }
    if gripper is not None:
        target["gripper"] = float(gripper)
    target = clamp_pose(target, cfg["motion"].get("joint_limits") or {})
    move_to(arm.bus, pose_i(target), wait_s=0.0)
    arm.goal = dict(target)
    arm.current_name = name
    time.sleep(float(cfg["motion"].get("settle_s", 0.12)))
