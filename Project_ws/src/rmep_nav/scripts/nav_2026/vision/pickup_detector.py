#!/usr/bin/python3
# coding=UTF-8

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rospy


@dataclass
class PickupTarget:
    cell_index: int
    row: int
    col: int
    pixel_u: float
    pixel_v: float
    confidence: float
    bbox: Tuple[int, int, int, int]


class PickupDetector:
    """
    识别取货区货架上的红色圆柱物资，映射到 3x3 格子。
    支持 yolo / hsv 两种模式，默认 hsv（无需模型文件）。
    """

    GRID_SIZE = 3

    def __init__(self, config=None):
        cfg = (config or {}).get("vision", {}).get("detector", {})
        shelf = (config or {}).get("shelf", {})

        self.mode = cfg.get("mode", "hsv")
        self.yolo_model_path = cfg.get("yolo_model", "")
        self.min_area = cfg.get("min_area", 150)
        self.min_circularity = cfg.get("min_circularity", 0.45)
        self.shelf_roi = cfg.get("shelf_roi", [0.15, 0.1, 0.85, 0.9])

        # 规则：中心 + 四边沿，线性索引 0-8
        # 0 1 2
        # 3 4 5
        # 6 7 8
        default_cells = shelf.get("center_cells", [1, 3, 4, 5, 7])
        self.valid_cells = set(default_cells)

        hsv = cfg.get("red_hsv", {})
        self.lower1 = np.array(hsv.get("lower1", [0, 80, 60]), dtype=np.uint8)
        self.upper1 = np.array(hsv.get("upper1", [10, 255, 255]), dtype=np.uint8)
        self.lower2 = np.array(hsv.get("lower2", [170, 80, 60]), dtype=np.uint8)
        self.upper2 = np.array(hsv.get("upper2", [180, 255, 255]), dtype=np.uint8)

        self._yolo = None
        if self.mode == "yolo" and self.yolo_model_path:
            self._init_yolo()

    def _init_yolo(self):
        try:
            from ultralytics import YOLO

            self._yolo = YOLO(self.yolo_model_path)
            rospy.loginfo("pickup detector loaded yolo: %s", self.yolo_model_path)
        except Exception as exc:
            rospy.logwarn("yolo load failed, fallback to hsv: %s", exc)
            self.mode = "hsv"

    @staticmethod
    def _index_to_rc(index: int) -> Tuple[int, int]:
        return index // 3, index % 3

    @staticmethod
    def _rc_to_index(row: int, col: int) -> int:
        return row * 3 + col

    def _roi_rect(self, frame) -> Tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        x1n, y1n, x2n, y2n = self.shelf_roi
        return (
            int(x1n * w),
            int(y1n * h),
            int(x2n * w),
            int(y2n * h),
        )

    def _cell_from_point(self, u: float, v: float, roi) -> int:
        x1, y1, x2, y2 = roi
        if not (x1 <= u < x2 and y1 <= v < y2):
            return -1
        cell_w = (x2 - x1) / self.GRID_SIZE
        cell_h = (y2 - y1) / self.GRID_SIZE
        col = min(int((u - x1) / cell_w), self.GRID_SIZE - 1)
        row = min(int((v - y1) / cell_h), self.GRID_SIZE - 1)
        return self._rc_to_index(row, col)

    def _detect_hsv(self, frame) -> Optional[PickupTarget]:
        roi = self._roi_rect(frame)
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower1, self.upper1) | cv2.inRange(
            hsv, self.lower2, self.upper2
        )
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cu = m["m10"] / m["m00"] + x1
            cv_ = m["m01"] / m["m00"] + y1
            cell = self._cell_from_point(cu, cv_, roi)
            if cell not in self.valid_cells:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            score = area * circularity
            if score > best_score:
                best_score = score
                conf = min(1.0, circularity)
                row, col = self._index_to_rc(cell)
                best = PickupTarget(
                    cell_index=cell,
                    row=row,
                    col=col,
                    pixel_u=cu,
                    pixel_v=cv_,
                    confidence=conf,
                    bbox=(x + x1, y + y1, x + x1 + w, y + y1 + h),
                )
        return best

    def _detect_yolo(self, frame) -> Optional[PickupTarget]:
        if self._yolo is None:
            return self._detect_hsv(frame)

        results = self._yolo(frame, verbose=False)
        if not results:
            return None

        roi = self._roi_rect(frame)
        best = None
        best_conf = 0.0
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                if conf < 0.5 or conf <= best_conf:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cu = (x1 + x2) / 2.0
                cv_ = (y1 + y2) / 2.0
                cell = self._cell_from_point(cu, cv_, roi)
                if cell not in self.valid_cells:
                    continue
                row, col = self._index_to_rc(cell)
                best_conf = conf
                best = PickupTarget(
                    cell_index=cell,
                    row=row,
                    col=col,
                    pixel_u=cu,
                    pixel_v=cv_,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                )
        return best

    def detect(self, frame) -> Optional[PickupTarget]:
        if frame is None:
            return None
        if self.mode == "yolo":
            return self._detect_yolo(frame)
        return self._detect_hsv(frame)

    def draw_debug(self, frame, target: Optional[PickupTarget] = None):
        vis = frame.copy()
        x1, y1, x2, y2 = self._roi_rect(frame)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cell_w = (x2 - x1) / self.GRID_SIZE
        cell_h = (y2 - y1) / self.GRID_SIZE
        for idx in range(self.GRID_SIZE * self.GRID_SIZE):
            row, col = self._index_to_rc(idx)
            cx1 = int(x1 + col * cell_w)
            cy1 = int(y1 + row * cell_h)
            cx2 = int(cx1 + cell_w)
            cy2 = int(cy1 + cell_h)
            color = (0, 180, 0) if idx in self.valid_cells else (80, 80, 80)
            cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), color, 1)

        if target is not None:
            bx1, by1, bx2, by2 = target.bbox
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            cv2.circle(vis, (int(target.pixel_u), int(target.pixel_v)), 6, (255, 0, 0), -1)
            label = f"cell={target.cell_index} conf={target.confidence:.2f}"
            cv2.putText(
                vis,
                label,
                (bx1, max(by1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        return vis
