#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO101 wrist-camera grasp -> place controller.

Only four taught poses are required. Target-dependent motion is planned at
runtime from the camera pixel, hand-eye calibration and SO101 IK:
grasp_observe -> automatic grasp -> grasp_reset; place_observe -> automatic
place -> safe.
"""
from __future__ import annotations
import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from og import JOINTS, TicksArm, clamp_pose  # noqa: E402

try:
    from grasp_3d import full_pose_from_ik, go_pose_strict, ik_route  # noqa: E402
    from pixel_to_base import grasp_z, load_handeye, load_intrinsics, pixel_to_base  # noqa: E402
    from so101_fk import JOINT_ORDER, SO101FK  # noqa: E402
    from so101_ik import SO101IK  # noqa: E402
except ImportError as exc:
    PLANNER_IMPORT_ERROR = exc
    full_pose_from_ik = go_pose_strict = ik_route = None
else:
    PLANNER_IMPORT_ERROR = None

DEFAULT_CONFIG = HERE / "grasp_and_put.yaml"

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a YAML mapping")
    return cfg

def capture_frame(camera: dict) -> np.ndarray:
    source = camera.get("index_or_path", 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera.get("width", 640)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera.get("height", 480)))
    for _ in range(max(0, int(camera.get("settle_frames", 4)))):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("camera returned no frame")
    return frame

@dataclass
class Detection:
    center_u: float
    center_v: float
    area: float
    contour: np.ndarray
    ring: bool

class RedDetector:
    def __init__(self, cfg: dict, ring: bool):
        self.cfg, self.ring = cfg, ring

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo1 = np.asarray(self.cfg.get("lower1", [0, 70, 45]), np.uint8)
        hi1 = np.asarray(self.cfg.get("upper1", [18, 255, 255]), np.uint8)
        lo2 = np.asarray(self.cfg.get("lower2", [165, 70, 45]), np.uint8)
        hi2 = np.asarray(self.cfg.get("upper2", [180, 255, 255]), np.uint8)
        mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)
        k = max(1, int(self.cfg.get("kernel", 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return None
        best = None
        for i, contour in enumerate(contours):
            area = float(cv2.contourArea(contour))
            if area < float(self.cfg.get("min_area", 500)):
                continue
            child, hole = int(hierarchy[0][i][2]), 0.0
            while child >= 0:
                hole = max(hole, float(cv2.contourArea(contours[child])))
                child = int(hierarchy[0][child][0])
            is_ring = hole / max(area, 1.0) >= float(self.cfg.get("min_hole_ratio", 0.10))
            if is_ring != self.ring:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            candidate = Detection(x + w / 2.0, y + h / 2.0, area, contour, is_ring)
            if best is None or candidate.area > best.area:
                best = candidate
        return best

    def draw(self, frame: np.ndarray, det: Optional[Detection]) -> np.ndarray:
        out = frame.copy()
        if det is not None:
            cv2.drawContours(out, [det.contour], -1, (0, 255, 0), 2)
            cv2.drawMarker(out, (round(det.center_u), round(det.center_v)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        return out

def save_debug(out_dir: Path, tag: str, image: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cv2.imwrite(str(out_dir / f"{tag}_{stamp}.jpg"), image)

def reset_output(out_dir: Path) -> None:
    """Clear only this task's output directory at the beginning of a run."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

def pose(cfg: dict, name: str, gripper: float) -> dict:
    item = (cfg.get("poses") or {}).get(name)
    if not isinstance(item, dict):
        raise ValueError(f"missing poses.{name}")
    missing = [j for j in JOINTS if j not in item]
    if missing:
        raise ValueError(f"poses.{name} missing: {missing}")
    result = {j: float(item[j]) for j in JOINTS}
    result["gripper"] = float(gripper)
    return result

def move_anchor(arm: TicksArm, cfg: dict, name: str, gripper: float, limits: dict) -> None:
    arm.go_pose(name, clamp_pose(pose(cfg, name, gripper), limits))

def plan_xyz(arm: TicksArm, fk, ik, xyz: np.ndarray, limits: dict, cfg: dict,
             gripper: float, label: str) -> None:
    ticks = arm.read()
    current = {n: float(ticks[n]) for n in JOINT_ORDER}
    cur_xyz = fk.forward_ticks(current)[:3, 3].copy()
    motion = cfg.get("motion") or {}
    n = max(2, int(motion.get("route_points", 5)))
    route = [cur_xyz + i / n * (np.asarray(xyz) - cur_xyz) for i in range(1, n + 1)]
    planned = ik_route(ik, route, current, current, cur_xyz, limits,
                       float(motion.get("ik_tol_m", 0.012)), False,
                       float(motion.get("max_horiz_m", 0.28)))
    for i, joint_ticks in enumerate(planned):
        go_pose_strict(arm, f"{label}_{i + 1}", full_pose_from_ik(joint_ticks, gripper, ticks))
        time.sleep(float(motion.get("waypoint_s", 0.05)))

def detect_once(camera: dict, detector: RedDetector, out: Path, tag: str) -> Detection:
    frame = capture_frame(camera)
    det = detector.detect(frame)
    save_debug(out, tag, detector.draw(frame, det))
    if det is None:
        raise RuntimeError(f"{tag}: target not detected")
    return det

def snapshot(camera: dict, detector: RedDetector, out: Path, tag: str) -> Optional[Detection]:
    """Save a stage frame with the current detector result overlaid."""
    frame = capture_frame(camera)
    det = detector.detect(frame)
    save_debug(out, tag, detector.draw(frame, det))
    return det

def main() -> int:
    ap = argparse.ArgumentParser(description="SO101 automatic red-block grasp and placement")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config.resolve())
    output_path = Path(cfg.get("output_dir", "output/grasp_put"))
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    reset_output(output_path)
    if args.dry_run:
        print("[dry-run] no arm connection; full automatic planning needs the hardware environment")
        return 0
    if not args.yes or not bool(cfg.get("motion_enabled", False)):
        print("Refusing motion: set motion_enabled: true and pass --yes")
        return 2
    if PLANNER_IMPORT_ERROR is not None:
        raise RuntimeError("SO101 automatic planner modules are unavailable") from PLANNER_IMPORT_ERROR
    limits = (cfg.get("motion") or {}).get("joint_limits") or {}
    g_open, g_close = float(cfg.get("gripper_open", 1200)), float(cfg.get("gripper_close", 800))
    camera = cfg.get("camera") or {}
    out = Path(cfg.get("output_dir", "output/grasp_put"))
    if not out.is_absolute():
        out = ROOT / out
    arm = TicksArm(str(cfg["port"]), int(cfg["baud"]), cfg["motion"])
    fk = SO101FK.from_config(cfg) if isinstance(cfg.get("fk"), dict) else SO101FK()
    ik = SO101IK(fk)
    intrinsics = Path(cfg["intrinsics"])
    handeye = Path(cfg["handeye"])
    K, dist = load_intrinsics(intrinsics if intrinsics.is_absolute() else ROOT / intrinsics)
    T_ee_cam = load_handeye(handeye if handeye.is_absolute() else ROOT / handeye)
    try:
        arm.connect()
        move_anchor(arm, cfg, "grasp_observe", g_open, limits)
        block_detector = RedDetector(cfg["grasp"]["detector"], False)
        block = detect_once(camera, block_detector, out, "grasp_observe")
        current = {n: float(arm.read()[n]) for n in JOINT_ORDER}
        T_ee = fk.forward_ticks(current)
        scene = cfg["scene"]
        _, grasp_z_m, _ = grasp_z(scene)
        result = pixel_to_base(block.center_u, block.center_v, T_ee, T_ee_cam, K, dist,
                               float(scene["table_z_m"]), np.asarray(scene.get("table_normal", [0, 0, 1])), grasp_z_m)
        target = np.asarray(result["p_grasp"], dtype=float)
        clearance = float(cfg["motion"].get("approach_clearance_m", 0.06))
        plan_xyz(arm, fk, ik, target + [0, 0, clearance], limits, cfg, g_open, "grasp_approach")
        snapshot(camera, block_detector, out, "grasp_approach_reached")
        plan_xyz(arm, fk, ik, target, limits, cfg, g_open, "grasp_down")
        snapshot(camera, block_detector, out, "grasp_down_before_close")
        arm.go_pose("close", clamp_pose({**arm.read(), "gripper": g_close}, limits))
        time.sleep(float(cfg["motion"].get("close_settle_s", 0.8)))
        snapshot(camera, block_detector, out, "grasp_closed")
        move_anchor(arm, cfg, "grasp_reset", g_close, limits)
        move_anchor(arm, cfg, "place_observe", g_close, limits)
        frame_detector = RedDetector(cfg["place"]["detector"], True)
        frame = detect_once(camera, frame_detector, out, "place_observe")
        current = {n: float(arm.read()[n]) for n in JOINT_ORDER}
        T_ee = fk.forward_ticks(current)
        place_z = float(scene["place_z_m"])
        result = pixel_to_base(frame.center_u, frame.center_v, T_ee, T_ee_cam, K, dist,
                               place_z, np.asarray([0, 0, 1.0]), place_z)
        target = np.asarray(result["p_grasp"], dtype=float)
        plan_xyz(arm, fk, ik, target + [0, 0, clearance], limits, cfg, g_close, "place_approach")
        snapshot(camera, frame_detector, out, "place_approach_reached")
        plan_xyz(arm, fk, ik, target, limits, cfg, g_close, "place_down")
        snapshot(camera, frame_detector, out, "place_down_before_release")
        arm.go_pose("release", clamp_pose({**arm.read(), "gripper": g_open}, limits))
        time.sleep(float(cfg["motion"].get("open_settle_s", 0.8)))
        snapshot(camera, frame_detector, out, "place_released")
        move_anchor(arm, cfg, "safe", g_open, limits)
        return 0
    finally:
        arm.disconnect(release_torque=bool(cfg.get("release_torque_on_exit", False)))

if __name__ == "__main__":
    raise SystemExit(main())
