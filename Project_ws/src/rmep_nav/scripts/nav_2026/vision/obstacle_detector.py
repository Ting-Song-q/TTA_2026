#!/usr/bin/python3
# coding=UTF-8
"""单目视觉障碍风险检测：输出前、左、右方向风险，不伪造米制深度。"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VisualSectorObservation:
    """单个图像方向的障碍占用和风险结果。"""

    risk: float
    occupied_ratio: float
    nearest_row_ratio: float


@dataclass
class VisualObstacleObservation:
    """一帧图像的三方向障碍风险结果。"""

    valid: bool
    confidence: float
    front: VisualSectorObservation
    left: VisualSectorObservation
    right: VisualSectorObservation
    reason: str = "ok"

    def risks(self):
        """返回与避障融合接口一致的方向风险字典。"""
        return {
            "front": self.front.risk,
            "left": self.left.risk,
            "right": self.right.risk,
        }


def _empty_sector():
    """创建无障碍方向占位结果。"""
    return VisualSectorObservation(0.0, 0.0, 0.0)


class VisualObstacleDetector:
    """基于地面外观差异和连通区域的单目视觉障碍检测器。"""

    def __init__(self, config=None):
        cfg = (config or {}).get("visual_avoidance", config or {})
        self.cfg = cfg
        self.roi = cfg.get("roi", [0.05, 0.35, 0.95, 0.98])
        self.floor_reference_height = cfg.get(
            "floor_reference_height_ratio", 0.10
        )
        self.color_difference_threshold = cfg.get(
            "color_difference_threshold", 0.24
        )
        self.canny_low = int(cfg.get("canny_low", 50))
        self.canny_high = int(cfg.get("canny_high", 140))
        self.min_contour_area_ratio = cfg.get(
            "min_contour_area_ratio", 0.002
        )
        self.risk_occupancy_ratio = cfg.get("risk_occupancy_ratio", 0.10)
        self.near_row_start = cfg.get("near_row_start", 0.45)

    @staticmethod
    def _clip_roi(roi, width, height):
        """把归一化 ROI 转换为合法像素范围。"""
        x1 = max(0, min(width - 1, int(roi[0] * width)))
        y1 = max(0, min(height - 1, int(roi[1] * height)))
        x2 = max(x1 + 1, min(width, int(roi[2] * width)))
        y2 = max(y1 + 1, min(height, int(roi[3] * height)))
        return x1, y1, x2, y2

    def _floor_difference_mask(self, crop):
        """以底部参考带估计地面颜色，返回偏离地面的像素掩码。"""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        height = hsv.shape[0]
        reference_height = max(2, int(height * self.floor_reference_height))
        reference = hsv[-reference_height:, :]
        floor_hsv = np.median(reference.reshape(-1, 3), axis=0)

        hue_difference = np.abs(hsv[:, :, 0].astype(np.float32) - floor_hsv[0])
        hue_difference = np.minimum(hue_difference, 180.0 - hue_difference) / 90.0
        saturation_difference = (
            np.abs(hsv[:, :, 1].astype(np.float32) - floor_hsv[1]) / 255.0
        )
        value_difference = (
            np.abs(hsv[:, :, 2].astype(np.float32) - floor_hsv[2]) / 255.0
        )
        difference = (
            hue_difference * 0.45
            + saturation_difference * 0.30
            + value_difference * 0.25
        )
        return (difference >= self.color_difference_threshold).astype(np.uint8) * 255

    def _obstacle_mask(self, crop):
        """融合颜色差异和边缘，生成填充后的障碍候选区域。"""
        color_mask = self._floor_difference_mask(crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        candidate = cv2.bitwise_or(color_mask, edges)
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            np.ones((9, 9), np.uint8),
            iterations=2,
        )
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )

        contours, _ = cv2.findContours(
            candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        obstacle_mask = np.zeros_like(candidate)
        min_area = crop.shape[0] * crop.shape[1] * self.min_contour_area_ratio
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            cv2.drawContours(obstacle_mask, [contour], -1, 255, thickness=-1)
        return obstacle_mask

    def _sector_observation(self, mask, x1_ratio, x2_ratio):
        """计算一个水平方向区域的占用率、最近行位置和综合风险。"""
        height, width = mask.shape[:2]
        x1 = max(0, min(width - 1, int(x1_ratio * width)))
        x2 = max(x1 + 1, min(width, int(x2_ratio * width)))
        sector = mask[:, x1:x2] > 0
        occupied_ratio = float(np.count_nonzero(sector)) / float(sector.size)

        rows = np.where(sector)[0]
        nearest_row_ratio = (
            float(rows.max()) / float(max(1, height - 1)) if rows.size else 0.0
        )
        occupancy_score = min(
            1.0, occupied_ratio / max(1e-6, self.risk_occupancy_ratio)
        )
        proximity_score = max(
            0.0,
            (nearest_row_ratio - self.near_row_start)
            / max(1e-6, 1.0 - self.near_row_start),
        )
        risk = min(1.0, occupancy_score * 0.65 + proximity_score * 0.35)
        return VisualSectorObservation(risk, occupied_ratio, nearest_row_ratio)

    @staticmethod
    def _frame_confidence(frame):
        """根据曝光和清晰度估计本帧是否足以判断有障碍或无障碍。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_value = float(np.mean(gray))
        exposure_score = max(0.0, 1.0 - abs(mean_value - 127.5) / 127.5)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = min(1.0, sharpness / 80.0)
        # 黑暗或过曝画面不能仅凭“收到图像”获得基础置信度。
        return min(1.0, exposure_score * 0.70 + sharpness_score * 0.30)

    def detect(self, frame):
        """检测一帧 BGR 图像，返回三方向风险和总体置信度。"""
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            empty = _empty_sector()
            return VisualObstacleObservation(
                False, 0.0, empty, empty, empty, "frame_invalid"
            )

        height, width = frame.shape[:2]
        if height < 32 or width < 32:
            empty = _empty_sector()
            return VisualObstacleObservation(
                False, 0.0, empty, empty, empty, "frame_too_small"
            )

        x1, y1, x2, y2 = self._clip_roi(self.roi, width, height)
        crop = frame[y1:y2, x1:x2]
        mask = self._obstacle_mask(crop)

        # 三个方向适度重叠，避免障碍刚好位于分区边界时被漏检。
        left = self._sector_observation(mask, 0.00, 0.38)
        front = self._sector_observation(mask, 0.28, 0.72)
        right = self._sector_observation(mask, 0.62, 1.00)
        confidence = self._frame_confidence(crop)
        return VisualObstacleObservation(True, confidence, front, left, right)

    def draw_debug(self, frame, observation):
        """绘制检测 ROI、方向风险和当前融合所需信息。"""
        output = frame.copy()
        height, width = output.shape[:2]
        x1, y1, x2, y2 = self._clip_roi(self.roi, width, height)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)

        labels = (
            ("L", observation.left.risk, 0.15),
            ("F", observation.front.risk, 0.50),
            ("R", observation.right.risk, 0.85),
        )
        for label, risk, position in labels:
            color = (0, 0, 255) if risk >= 0.75 else (0, 255, 0)
            text = f"{label}:{risk:.2f}"
            cv2.putText(
                output,
                text,
                (int(width * position) - 30, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
        return output
