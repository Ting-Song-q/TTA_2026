#!/usr/bin/python3
# coding=UTF-8

# BEGIN added: yellow zone-boundary vision detector module
"""Detect and validate the yellow boundary of a vehicle zone."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rospy
import yaml


def _order_points_clockwise(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 4:
        return points
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    start = np.argmin(points[:, 0] + points[:, 1])
    return np.roll(points, -start, axis=0)


def _size_pair(value, default):
    if isinstance(value, dict):
        return float(value.get("width", default[0])), float(
            value.get("height", default[1])
        )
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return float(default[0]), float(default[1])


@dataclass
class ZoneBoundaryObservation:
    zone_name: str
    confidence: float
    polygon_image: list
    polygon_base: Optional[list]
    bbox: tuple
    geometry_valid: bool = False
    reason: str = ""
    width_base: Optional[float] = None
    height_base: Optional[float] = None


class ZoneBoundaryDetector:
    """Detect a yellow rectangle and project it onto the base_link ground plane."""

    def __init__(self, config=None):
        zone_cfg = (config or {}).get("zone_entry", {})
        hsv = zone_cfg.get("yellow_hsv", {})
        geometry = zone_cfg.get("geometry_validation", {})
        self.enabled = zone_cfg.get("enabled", True)
        self.fallback_to_map = zone_cfg.get("fallback_to_map", True)
        self.vision_confidence = float(zone_cfg.get("vision_confidence", 0.65))
        self.min_area = float(zone_cfg.get("min_area", 600.0))
        self.hsv_lower = np.array(hsv.get("lower", [15, 60, 80]), dtype=np.uint8)
        self.hsv_upper = np.array(hsv.get("upper", [40, 255, 255]), dtype=np.uint8)
        self.geometry_enabled = geometry.get("enabled", True)
        self.expected_size = _size_pair(geometry.get("expected_size"), (0.8, 0.8))
        self.size_tolerance = _size_pair(
            geometry.get("size_tolerance"), (0.20, 0.20)
        )
        self.min_rectangularity = float(geometry.get("min_rectangularity", 0.55))
        self.min_projected_area = float(geometry.get("min_projected_area", 0.25))
        self._homography = None
        self.homography_reason = "homography_disabled"
        self._load_homography(zone_cfg.get("homography", {}))

    @property
    def homography_available(self):
        return self._homography is not None

    def _load_homography_file(self, cfg):
        filename = cfg.get("file")
        if not filename:
            return cfg
        path = Path(filename)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        if not path.exists():
            self.homography_reason = "homography_file_missing:%s" % path
            return cfg
        try:
            with open(path, "r", encoding="utf-8") as stream:
                loaded = yaml.safe_load(stream) or {}
        except (OSError, yaml.YAMLError) as exc:
            self.homography_reason = "homography_file_invalid:%s" % exc
            return cfg
        if "zone_entry" in loaded:
            loaded = loaded.get("zone_entry", {}).get("homography", {})
        elif "homography" in loaded:
            loaded = loaded.get("homography", {})
        merged = dict(cfg)
        merged.update(loaded)
        return merged

    def _load_homography(self, cfg):
        if not cfg or not cfg.get("enabled", False):
            return
        cfg = self._load_homography_file(cfg)
        matrix = cfg.get("matrix")
        if matrix is not None:
            candidate = np.asarray(matrix, dtype=np.float64)
            if candidate.shape == (3, 3) and np.isfinite(candidate).all():
                self._homography = candidate
                self.homography_reason = "homography_loaded"
                return

        image_points = cfg.get("image_points", [])
        base_points = cfg.get("base_points", [])
        if len(image_points) < 4 or len(base_points) != len(image_points):
            self.homography_reason = "homography_points_missing"
            return
        image = np.asarray(image_points, dtype=np.float32)
        base = np.asarray(base_points, dtype=np.float32)
        try:
            self._homography, _ = cv2.findHomography(image, base, method=0)
        except cv2.error as exc:
            self.homography_reason = "homography_compute_failed:%s" % exc
            self._homography = None
            return
        if self._homography is None:
            self.homography_reason = "homography_compute_failed"
            return
        self.homography_reason = "homography_loaded"

    def _detect_contour(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            rect = cv2.minAreaRect(contour)
            rect_area = max(float(rect[1][0] * rect[1][1]), 1.0)
            rectangularity = min(1.0, area / rect_area)
            vertex_score = 1.0 if len(approx) == 4 else min(0.8, 4.0 / max(len(approx), 4))
            score = area * (0.65 * rectangularity + 0.35 * vertex_score)
            if score <= best_score:
                continue
            if len(approx) == 4:
                polygon = approx.reshape(-1, 2)
            else:
                polygon = cv2.boxPoints(rect)
            best_score = score
            best = {
                "contour": contour,
                "polygon": _order_points_clockwise(polygon),
                "area": area,
                "rectangularity": rectangularity,
                "score": score,
            }
        return best

    def _project_to_base(self, image_polygon):
        if self._homography is None:
            return None
        points = np.asarray(image_polygon, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(points, self._homography).reshape(-1, 2)
        return [(float(x), float(y)) for x, y in _order_points_clockwise(projected)]

    def _validate_geometry(self, polygon_base, rectangularity):
        if polygon_base is None:
            return False, self.homography_reason, None, None
        rect = cv2.minAreaRect(np.asarray(polygon_base, dtype=np.float32))
        width, height = sorted((float(rect[1][0]), float(rect[1][1])))
        expected_width, expected_height = sorted(self.expected_size)
        tolerance_width, tolerance_height = sorted(self.size_tolerance)
        projected_area = width * height
        if not self.geometry_enabled:
            return True, "geometry_check_disabled", width, height
        if rectangularity < self.min_rectangularity:
            return False, "low_rectangularity", width, height
        if projected_area < self.min_projected_area:
            return False, "projected_area_too_small", width, height
        if abs(width - expected_width) > tolerance_width:
            return False, "zone_width_mismatch", width, height
        if abs(height - expected_height) > tolerance_height:
            return False, "zone_height_mismatch", width, height
        return True, "geometry_valid", width, height

    def detect(self, frame, zone_name) -> Optional[ZoneBoundaryObservation]:
        if frame is None or not self.enabled:
            return None
        best = self._detect_contour(frame)
        if best is None:
            return None

        image_polygon = [(float(x), float(y)) for x, y in best["polygon"]]
        polygon_base = self._project_to_base(image_polygon)
        geometry_valid, reason, width, height = self._validate_geometry(
            polygon_base, best["rectangularity"]
        )
        frame_area = max(float(frame.shape[0] * frame.shape[1]), 1.0)
        confidence = min(
            1.0,
            0.35
            + 0.30 * min(1.0, best["area"] / (frame_area * 0.10))
            + 0.25 * best["rectangularity"]
            + (0.10 if geometry_valid else 0.0),
        )
        return ZoneBoundaryObservation(
            zone_name=zone_name,
            confidence=confidence,
            polygon_image=image_polygon,
            polygon_base=polygon_base,
            bbox=cv2.boundingRect(best["contour"]),
            geometry_valid=geometry_valid,
            reason=reason,
            width_base=width,
            height_base=height,
        )

    def draw_debug(self, frame, observation=None, wheel_points=None):
        visual = frame.copy()
        if observation is not None and observation.polygon_image:
            points = np.asarray(observation.polygon_image, dtype=np.int32).reshape(
                -1, 1, 2
            )
            color = (0, 200, 0) if observation.geometry_valid else (0, 0, 255)
            cv2.polylines(visual, [points], True, color, 2)
            x, y, width, height = observation.bbox
            cv2.rectangle(visual, (x, y), (x + width, y + height), color, 1)
            size_text = ""
            if observation.width_base is not None:
                size_text = " %.2fx%.2fm" % (
                    observation.width_base,
                    observation.height_base,
                )
            cv2.putText(
                visual,
                "%s %.2f%s %s"
                % (
                    observation.zone_name,
                    observation.confidence,
                    size_text,
                    observation.reason,
                ),
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        if wheel_points:
            for _, (wheel_x, wheel_y) in wheel_points.items():
                cv2.circle(visual, (int(wheel_x), int(wheel_y)), 5, (255, 0, 0), -1)
        return visual
# END added: yellow zone-boundary vision detector module
