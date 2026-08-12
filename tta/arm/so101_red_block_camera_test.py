#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 wrist-camera red-block detection test.

This script intentionally does not connect to, calibrate, or command the arm.
It uses the same OpenCV camera settings that LeRobot's ``so101_follower``
configuration uses, so it is safe to run before adding visual-servo control.
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "horizontal_red_grasp_config.yaml"

DEFAULT_CONFIG = {
    "camera": {
        # Match LeRobot's OpenCV camera configuration. On Linux this can also
        # be a stable path such as /dev/video_wrist.
        "index_or_path": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "settle_frames": 12,
        "frame_timeout_s": 5.0,
    },
    "output": {
        "directory": "output/so101_red_block_camera",
        "save_mask": True,
    },
    "detector": {
        # Exclude image regions occupied by the gripper or camera mount.
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
        # A square viewed from the wrist camera can be perspective-distorted.
        "aspect_ratio_range": [0.35, 2.80],
        "open_kernel": 3,
        "close_kernel": 5,
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
    if not path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return deep_update(DEFAULT_CONFIG, loaded)


@dataclass
class RedBlockDetection:
    center_u: float
    center_v: float
    area: float
    bbox: Tuple[int, int, int, int]
    rect_fill: float
    aspect_ratio: float
    contour: np.ndarray
    roi: Tuple[int, int, int, int]
    mask: np.ndarray


class RedBlockDetector:
    def __init__(self, config: dict):
        cfg = config["detector"]
        self.roi_norm = cfg["roi"]
        hsv = cfg["red_hsv"]
        self.lower1 = np.array(hsv["lower1"], dtype=np.uint8)
        self.upper1 = np.array(hsv["upper1"], dtype=np.uint8)
        self.lower2 = np.array(hsv["lower2"], dtype=np.uint8)
        self.upper2 = np.array(hsv["upper2"], dtype=np.uint8)
        rgb = cfg["red_rgb"]
        self.min_r = int(rgb["min_r"])
        self.min_r_minus_g = int(rgb["min_r_minus_g"])
        self.min_r_minus_b = int(rgb["min_r_minus_b"])
        self.max_g = int(rgb["max_g"])
        self.min_area = float(cfg["min_area"])
        self.min_rect_fill = float(cfg["min_rect_fill"])
        self.aspect_min, self.aspect_max = map(float, cfg["aspect_ratio_range"])
        self.open_kernel = int(cfg["open_kernel"])
        self.close_kernel = int(cfg["close_kernel"])

    def roi_rect(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        x1n, y1n, x2n, y2n = self.roi_norm
        x1 = max(0, min(width - 1, int(width * x1n)))
        y1 = max(0, min(height - 1, int(height * y1n)))
        x2 = max(x1 + 1, min(width, int(width * x2n)))
        y2 = max(y1 + 1, min(height, int(height * y2n)))
        return x1, y1, x2, y2

    def detect(self, frame: np.ndarray) -> Optional[RedBlockDetection]:
        x1, y1, x2, y2 = self.roi_rect(frame)
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, self.lower1, self.upper1) | cv2.inRange(
            hsv, self.lower2, self.upper2
        )

        blue, green, red = cv2.split(crop)
        blue = blue.astype(np.int16)
        green = green.astype(np.int16)
        red = red.astype(np.int16)
        rgb_mask = (
            (red >= self.min_r)
            & ((red - green) >= self.min_r_minus_g)
            & ((red - blue) >= self.min_r_minus_b)
            & (green <= self.max_g)
        ).astype(np.uint8) * 255
        mask = cv2.bitwise_and(hsv_mask, rgb_mask)

        if self.open_kernel > 1:
            kernel = np.ones((self.open_kernel, self.open_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if self.close_kernel > 1:
            kernel = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[RedBlockDetection] = None
        best_score = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            rect_fill = area / float(max(width * height, 1))
            aspect_ratio = width / float(max(height, 1))
            if rect_fill < self.min_rect_fill or not self.aspect_min <= aspect_ratio <= self.aspect_max:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            score = area * rect_fill
            if score > best_score:
                best_score = score
                best = RedBlockDetection(
                    center_u=moments["m10"] / moments["m00"] + x1,
                    center_v=moments["m01"] / moments["m00"] + y1,
                    area=area,
                    bbox=(x + x1, y + y1, x + x1 + width, y + y1 + height),
                    rect_fill=rect_fill,
                    aspect_ratio=aspect_ratio,
                    contour=contour.copy(),
                    roi=(x1, y1, x2, y2),
                    mask=mask,
                )
        return best

    def draw(self, frame: np.ndarray, detection: Optional[RedBlockDetection]) -> np.ndarray:
        vis = frame.copy()
        x1, y1, x2, y2 = detection.roi if detection else self.roi_rect(frame)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if detection is None:
            cv2.putText(vis, "no red block", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            return vis
        bx1, by1, bx2, by2 = detection.bbox
        contour = detection.contour + np.array([[[x1, y1]]], dtype=np.int32)
        cv2.drawContours(vis, [contour], -1, (0, 255, 255), 2)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        cv2.drawMarker(vis, (round(detection.center_u), round(detection.center_v)), (255, 0, 0), cv2.MARKER_CROSS, 18, 2)
        label = "u={:.1f} v={:.1f} area={:.0f} fill={:.2f}".format(
            detection.center_u, detection.center_v, detection.area, detection.rect_fill
        )
        cv2.putText(vis, label, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (50, 220, 50), 2)
        return vis


def camera_source(value: object) -> object:
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


def capture_frame(camera_cfg: dict) -> np.ndarray:
    source = camera_source(camera_cfg["index_or_path"])
    if isinstance(source, int) and os.name == "nt":
        # DirectShow avoids long startup delays on many Windows USB cameras.
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    elif isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open wrist camera: {source!r}")
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_cfg["width"]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_cfg["height"]))
        cap.set(cv2.CAP_PROP_FPS, int(camera_cfg["fps"]))
        deadline = time.monotonic() + float(camera_cfg["frame_timeout_s"])
        frame = None
        for _ in range(max(1, int(camera_cfg["settle_frames"]))):
            ok, frame = cap.read()
            if not ok and time.monotonic() >= deadline:
                raise RuntimeError("wrist camera did not provide a frame before timeout")
        if frame is None:
            raise RuntimeError("wrist camera returned an empty frame")
        return frame
    finally:
        cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one SO-101 wrist-camera image and detect a red block.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--camera", help="Override camera.index_or_path, for example 0 or /dev/video_wrist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.camera is not None:
        config["camera"]["index_or_path"] = camera_source(args.camera)
    detector = RedBlockDetector(config)
    try:
        frame = capture_frame(config["camera"])
    except RuntimeError as exc:
        print(f"camera error: {exc}")
        return 2

    detection = detector.detect(frame)
    annotated = detector.draw(frame, detection)
    output_dir = Path(config["output"]["directory"])
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = output_dir / f"{stamp}_raw.jpg"
    annotated_path = output_dir / f"{stamp}_detected.jpg"
    cv2.imwrite(str(raw_path), frame)
    cv2.imwrite(str(annotated_path), annotated)
    if detection is not None and config["output"].get("save_mask", True):
        cv2.imwrite(str(output_dir / f"{stamp}_mask.png"), detection.mask)

    print(f"raw image: {raw_path}")
    print(f"annotated image: {annotated_path}")
    if detection is None:
        print("red block: not detected")
        return 1
    print(
        "red block: center=({:.1f}, {:.1f}) bbox={} area={:.0f} fill={:.2f} aspect={:.2f}".format(
            detection.center_u,
            detection.center_v,
            detection.bbox,
            detection.area,
            detection.rect_fill,
            detection.aspect_ratio,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
