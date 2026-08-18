"""Self-contained wrist-camera red-block detector for the mobile grasp flow."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


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

    def detect_candidates(self, frame: np.ndarray) -> List[RedBlockDetection]:
        x1, y1, x2, y2 = self.roi_rect(frame)
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower1, self.upper1) | cv2.inRange(hsv, self.lower2, self.upper2)
        blue, green, red = [channel.astype(np.int16) for channel in cv2.split(crop)]
        rgb_mask = ((red >= self.min_r) & ((red - green) >= self.min_r_minus_g) &
                    ((red - blue) >= self.min_r_minus_b) & (green <= self.max_g)).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, rgb_mask)
        if self.open_kernel > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((self.open_kernel, self.open_kernel), np.uint8))
        if self.close_kernel > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((self.close_kernel, self.close_kernel), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[RedBlockDetection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, width, height = cv2.boundingRect(contour)
            fill = area / float(max(width * height, 1))
            aspect = width / float(max(height, 1))
            if area < self.min_area or fill < self.min_rect_fill or not self.aspect_min <= aspect <= self.aspect_max:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            candidates.append(RedBlockDetection(
                moments["m10"] / moments["m00"] + x1, moments["m01"] / moments["m00"] + y1,
                area, (x + x1, y + y1, x + x1 + width, y + y1 + height), fill, aspect,
                contour.copy(), (x1, y1, x2, y2), mask))
        return candidates

    @staticmethod
    def select_closest_to_u(candidates: List[RedBlockDetection], target_u: float) -> Optional[RedBlockDetection]:
        return min(candidates, key=lambda item: abs(item.center_u - target_u), default=None)

    def detect(self, frame: np.ndarray) -> Optional[RedBlockDetection]:
        candidates = self.detect_candidates(frame)
        return max(candidates, key=lambda item: item.area * item.rect_fill, default=None)

    def draw(self, frame: np.ndarray, detection: Optional[RedBlockDetection], candidates=None) -> np.ndarray:
        vis = frame.copy()
        roi = detection.roi if detection is not None else self.roi_rect(frame)
        cv2.rectangle(vis, roi[:2], roi[2:], (0, 255, 255), 2)
        for index, candidate in enumerate(candidates or [], start=1):
            cv2.rectangle(vis, candidate.bbox[:2], candidate.bbox[2:], (255, 255, 0), 1)
            cv2.putText(vis, f"candidate {index}", (candidate.bbox[0], max(16, candidate.bbox[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1)
        if detection is None:
            cv2.putText(vis, "no red block", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            return vis
        cv2.rectangle(vis, detection.bbox[:2], detection.bbox[2:], (0, 0, 255), 2)
        contour = detection.contour + np.array([[[detection.roi[0], detection.roi[1]]]], dtype=np.int32)
        cv2.drawContours(vis, [contour], -1, (0, 255, 255), 2)
        cv2.drawMarker(vis, (round(detection.center_u), round(detection.center_v)), (255, 0, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(vis, f"selected u={detection.center_u:.1f} v={detection.center_v:.1f} area={detection.area:.0f}",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (50, 220, 50), 2)
        return vis


def camera_source(value: object) -> object:
    text = str(value)
    return int(text) if text.isdigit() else text


def capture_frame(camera_cfg: dict) -> np.ndarray:
    source = camera_source(camera_cfg["index_or_path"])
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2) if isinstance(source, int) and os.name != "nt" else cv2.VideoCapture(source)
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
