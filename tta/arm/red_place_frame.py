# -*- coding: utf-8 -*-
"""红色镂空放置框检测（自 ArmPi Pro grasp_vision/test2.py 移植，无 ROS 依赖）。"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class RedFrameDetection(object):
    def __init__(
        self, center_u, center_v, area, bbox, contour, mask, roi,
        ring_fill, aspect, hole_ratio, has_hole,
    ):
        self.center_u = center_u
        self.center_v = center_v
        self.area = area          # 环带面积（红像素）
        self.bbox = bbox
        self.contour = contour    # 外轮廓（crop 坐标系）
        self.mask = mask
        self.roi = roi
        self.rect_fill = ring_fill
        self.aspect = aspect
        self.hole_ratio = hole_ratio
        self.has_hole = has_hole


class RedPlacementFrameDetector(object):
    """检测空心红色放置框：膨胀连边 + 边框/内部红占比判空心 + 台面位置偏好。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = "scout"  # scout | track
        self.lower1 = np.array(cfg["hsv_lower1"], dtype=np.uint8)
        self.upper1 = np.array(cfg["hsv_upper1"], dtype=np.uint8)
        self.lower2 = np.array(cfg["hsv_lower2"], dtype=np.uint8)
        self.upper2 = np.array(cfg["hsv_upper2"], dtype=np.uint8)
        self.min_r = int(cfg["min_r"])
        self.min_r_minus_g = int(cfg["min_r_minus_g"])
        self.min_r_minus_b = int(cfg["min_r_minus_b"])
        self.max_g = int(cfg["max_g"])
        self.min_hole_ratio = float(cfg["min_hole_ratio"])
        self.min_ring_fill = float(cfg["min_ring_fill"])
        self.max_interior_red = float(cfg.get("max_interior_red", 0.48))
        self.min_border_red = float(cfg.get("min_border_red", 0.06))
        self.max_solid_interior = float(cfg.get("max_solid_interior", 0.75))
        self.thin_ring_fill_max = float(cfg.get("thin_ring_fill_max", 0.40))
        self.aspect_min = float(cfg["aspect_min"])
        self.aspect_max = float(cfg["aspect_max"])
        self.prefer_square_weight = float(cfg["prefer_square_weight"])
        self.prefer_hole_weight = float(cfg["prefer_hole_weight"])
        self.prefer_table_weight = float(cfg.get("prefer_table_weight", 1.20))
        self.prefer_midsize_area = float(cfg.get("prefer_midsize_area", 3500))
        self.table_v_lo = float(cfg.get("table_v_norm_lo", 0.18))
        self.table_v_hi = float(cfg.get("table_v_norm_hi", 0.55))
        self.open_kernel = int(cfg.get("open_kernel", 2))
        self.close_kernel = int(cfg.get("close_kernel", 3))
        self.close_iters = int(cfg.get("close_iters", 1))
        self.dilate_kernel = int(cfg.get("dilate_kernel", 7))
        self.dilate_iters = int(cfg.get("dilate_iters", 2))
        self._apply_mode_params()

    def set_mode(self, mode):
        """scout=侦察筛选更严；track=跟踪略放宽 ROI/门槛。"""
        mode = str(mode or "scout").lower()
        if mode not in ("scout", "track"):
            mode = "scout"
        self.mode = mode
        self._apply_mode_params()

    def _apply_mode_params(self):
        cfg = self.cfg
        if self.mode == "track":
            self.roi_norm = list(cfg.get("track_roi", cfg["roi"]))
            self.min_bbox_w = int(cfg.get("track_min_bbox_w", cfg["min_bbox_w"]))
            self.min_bbox_h = int(cfg.get("track_min_bbox_h", cfg["min_bbox_h"]))
            self.min_bbox_area = float(cfg.get("track_min_bbox_area", cfg.get("min_bbox_area", 900)))
            self.min_outer_area = float(cfg.get("track_min_outer_area", cfg["min_outer_area"]))
            self.min_ring_area = float(cfg.get("track_min_ring_area", cfg["min_ring_area"]))
            self.max_ring_fill = float(cfg.get("track_max_ring_fill", cfg["max_ring_fill"]))
            self.min_center_v_norm = float(cfg.get("track_min_center_v_norm", cfg.get("min_center_v_norm", 0.16)))
            self.max_center_v_norm = float(cfg.get("track_max_center_v_norm", cfg.get("max_center_v_norm", 0.62)))
            self.aspect_max = float(cfg.get("track_aspect_max", cfg["aspect_max"]))
        else:
            self.roi_norm = list(cfg["roi"])
            self.min_bbox_w = int(cfg["min_bbox_w"])
            self.min_bbox_h = int(cfg["min_bbox_h"])
            self.min_bbox_area = float(cfg.get("min_bbox_area", 900))
            self.min_outer_area = float(cfg["min_outer_area"])
            self.min_ring_area = float(cfg["min_ring_area"])
            self.max_ring_fill = float(cfg["max_ring_fill"])
            self.min_center_v_norm = float(cfg.get("min_center_v_norm", 0.16))
            self.max_center_v_norm = float(cfg.get("max_center_v_norm", 0.62))
            self.aspect_max = float(cfg["aspect_max"])

    def _roi(self, frame):
        h, w = frame.shape[:2]
        x1n, y1n, x2n, y2n = self.roi_norm
        x1 = max(0, min(w - 1, int(w * x1n)))
        y1 = max(0, min(h - 1, int(h * y1n)))
        x2 = max(x1 + 1, min(w, int(w * x2n)))
        y2 = max(y1 + 1, min(h, int(h * y2n)))
        return x1, y1, x2, y2

    def _red_mask(self, crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower1, self.upper1) | cv2.inRange(hsv, self.lower2, self.upper2)
        b, g, r = cv2.split(crop)
        r = r.astype(np.int16)
        g = g.astype(np.int16)
        b = b.astype(np.int16)
        rgb_mask = (
            (r >= self.min_r)
            & ((r - g) >= self.min_r_minus_g)
            & ((r - b) >= self.min_r_minus_b)
            & (g <= self.max_g)
        ).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, rgb_mask)
        if self.open_kernel > 1:
            k = np.ones((self.open_kernel, self.open_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        if self.close_kernel > 1 and self.close_iters > 0:
            k = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=self.close_iters)
        return mask

    def _dilate_for_link(self, mask):
        if self.dilate_kernel <= 1 or self.dilate_iters <= 0:
            return mask
        k = np.ones((self.dilate_kernel, self.dilate_kernel), np.uint8)
        return cv2.dilate(mask, k, iterations=self.dilate_iters)

    @staticmethod
    def _border_interior_stats(mask, x, y, w, h):
        """在 bbox 内统计边框带与内部区域的红色占比。"""
        roi = mask[y:y + h, x:x + w]
        if roi.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        border = max(2, int(min(w, h) * 0.22))
        border = min(border, max(1, w // 3), max(1, h // 3))
        band = np.zeros((h, w), dtype=np.uint8)
        band[:border, :] = 255
        band[-border:, :] = 255
        band[:, :border] = 255
        band[:, -border:] = 255
        interior = np.zeros((h, w), dtype=np.uint8)
        if h > 2 * border and w > 2 * border:
            interior[border:h - border, border:w - border] = 255
        band_n = float(max(cv2.countNonZero(band), 1))
        int_n = float(max(cv2.countNonZero(interior), 1))
        border_red = float(cv2.countNonZero(cv2.bitwise_and(roi, band))) / band_n
        interior_red = float(cv2.countNonZero(cv2.bitwise_and(roi, interior))) / int_n
        ring_px = float(cv2.countNonZero(cv2.bitwise_and(roi, band)))
        all_red = float(cv2.countNonZero(roi))
        return border_red, interior_red, ring_px, all_red

    @staticmethod
    def _largest_child_hole(hierarchy, parent_idx, contours, min_hole_area):
        if hierarchy is None:
            return None, 0.0
        hier = hierarchy[0]
        child = hier[parent_idx][2]
        best_idx, best_area = None, 0.0
        while child >= 0:
            area = float(cv2.contourArea(contours[child]))
            if area >= min_hole_area and area > best_area:
                best_area = area
                best_idx = child
            child = hier[child][0]
        return best_idx, best_area

    def detect(self, frame, mode=None):
        if mode is not None:
            self.set_mode(mode)
        if frame is None:
            return None
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = self._roi(frame)
        crop = frame[y1:y2, x1:x2]
        mask = self._red_mask(crop)
        linked = self._dilate_for_link(mask)

        found = cv2.findContours(linked, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(found) == 3:
            _, contours, hierarchy = found
        else:
            contours, hierarchy = found
        if not contours:
            return None

        best, best_score = None, 0.0
        for i, contour in enumerate(contours):
            if hierarchy is not None and hierarchy[0][i][3] != -1:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            bbox_area = float(w * h)
            if w < self.min_bbox_w or h < self.min_bbox_h:
                continue
            if bbox_area < self.min_bbox_area:
                continue
            aspect = float(w) / float(h)
            if aspect < self.aspect_min or aspect > self.aspect_max:
                continue

            outer_area = float(max(cv2.contourArea(contour), bbox_area * 0.35))
            if outer_area < self.min_outer_area:
                continue

            border_red, interior_red, ring_px, all_red = self._border_interior_stats(mask, x, y, w, h)
            ring_area = max(ring_px, all_red * 0.5)
            if ring_area < self.min_ring_area:
                continue

            ring_fill = all_red / float(max(bbox_area, 1))
            if ring_fill < self.min_ring_fill or ring_fill > self.max_ring_fill:
                continue

            hole_idx, hole_area = self._largest_child_hole(
                hierarchy, i, contours, min_hole_area=outer_area * 0.05
            )
            has_hole = hole_idx is not None and hole_area > 0
            hole_ratio = hole_area / outer_area if outer_area > 1e-6 else 0.0

            hollow_by_ratio = (
                interior_red <= self.max_interior_red
                and border_red >= self.min_border_red
                and (
                    border_red + 0.02 >= interior_red
                    or ring_fill <= self.thin_ring_fill_max
                )
            )
            hollow_by_thin = (
                ring_fill <= self.thin_ring_fill_max
                and border_red >= self.min_border_red
                and interior_red < self.max_solid_interior
            )
            hollow_by_hole = has_hole and hole_ratio >= self.min_hole_ratio
            if not (hollow_by_ratio or hollow_by_thin or hollow_by_hole):
                continue
            if ring_fill > 0.55 and interior_red > 0.55:
                continue
            if interior_red >= self.max_solid_interior and ring_fill > 0.45:
                continue

            cu = x + w * 0.5 + x1
            cv_ = y + h * 0.5 + y1
            if has_hole:
                hx, hy, hw, hh = cv2.boundingRect(contours[hole_idx])
                cu = 0.5 * (cu + (hx + hw * 0.5 + x1))
                cv_ = 0.5 * (cv_ + (hy + hh * 0.5 + y1))

            v_norm = float(cv_) / float(max(fh, 1))
            u_norm = float(cu) / float(max(fw, 1))
            if v_norm < self.min_center_v_norm or v_norm > self.max_center_v_norm:
                continue
            if (y <= 4) and bbox_area < self.min_bbox_area * 1.5:
                continue

            square_score = 1.0 - min(1.0, abs(math.log(max(aspect, 1e-6))))
            hollow_score = max(hole_ratio, max(0.0, 1.0 - interior_red), border_red)
            is_hollow = bool(has_hole or hollow_by_ratio or hollow_by_thin)

            mid = max(self.prefer_midsize_area, 1.0)
            size_ratio = bbox_area / mid
            midsize_score = math.exp(-abs(math.log(max(size_ratio, 1e-3))) * 0.85)
            fill_score = 1.0 - abs(ring_fill - 0.18)
            fill_score = max(0.35, min(1.15, fill_score + 0.5))
            in_table = self.table_v_lo <= v_norm <= self.table_v_hi
            table_score = (1.0 + self.prefer_table_weight) if in_table else 0.55
            center_u_score = 1.0 - min(0.45, abs(u_norm - 0.5))

            score = (
                bbox_area
                * (0.35 + 0.65 * hollow_score)
                * (1.0 + self.prefer_hole_weight * (1.0 if is_hollow else 0.0))
                * (1.0 + self.prefer_square_weight * square_score)
                * (1.15 - min(0.55, ring_fill))
                * (1.0 + 0.45 * border_red)
                * midsize_score
                * fill_score
                * table_score
                * (0.75 + 0.5 * center_u_score)
            )
            if score > best_score:
                best_score = score
                best = RedFrameDetection(
                    cu, cv_, ring_area,
                    (x + x1, y + y1, x + x1 + w, y + y1 + h),
                    contour.copy(), mask.copy(), (x1, y1, x2, y2),
                    ring_fill, aspect,
                    max(hole_ratio, max(0.0, 1.0 - interior_red)),
                    is_hollow,
                )
        return best

    def draw(self, frame, det, desired=None):
        vis = frame.copy()
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = det.roi if det is not None else self._roi(frame)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if desired is not None:
            cv2.drawMarker(
                vis, (int(desired[0]), int(desired[1])),
                (255, 255, 0), cv2.MARKER_CROSS, 18, 2,
            )
        if det is None:
            cv2.putText(vis, "no red place frame", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return vis
        bx1, by1, bx2, by2 = det.bbox
        contour = det.contour + np.array([[[x1, y1]]], dtype=np.int32)
        cv2.drawContours(vis, [contour], -1, (0, 255, 255), 2)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        cv2.circle(vis, (int(det.center_u), int(det.center_v)), 5, (255, 0, 0), -1)
        cv2.putText(
            vis,
            "ring={:.0f} fill={:.2f} hole={:.2f} asp={:.2f}".format(
                det.area, det.rect_fill, det.hole_ratio, det.aspect
            ),
            (bx1, max(20, by1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2,
        )
        if det.mask is not None:
            preview = cv2.cvtColor(det.mask, cv2.COLOR_GRAY2BGR)
            preview = cv2.resize(preview, (220, 160))
            vis[10:170, w - 230:w - 10] = preview
            cv2.rectangle(vis, (w - 230, 10), (w - 10, 170), (255, 255, 255), 1)
        if desired is not None:
            cv2.putText(
                vis,
                "err_u={:+.0f} err_v={:+.0f}".format(
                    det.center_u - desired[0], det.center_v - desired[1]
                ),
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2,
            )
        return vis


def default_red_frame_cfg() -> Dict[str, Any]:
    """与 test2.py CFG 中检测相关默认值一致。"""
    return {
        "roi": [0.08, 0.18, 0.92, 0.66],
        "track_roi": [0.05, 0.12, 0.95, 0.72],
        "hsv_lower1": [0, 55, 35],
        "hsv_upper1": [20, 255, 255],
        "hsv_lower2": [160, 55, 35],
        "hsv_upper2": [180, 255, 255],
        "min_r": 80,
        "min_r_minus_g": 20,
        "min_r_minus_b": 15,
        "max_g": 220,
        "min_bbox_w": 28,
        "min_bbox_h": 22,
        "min_bbox_area": 900,
        "min_outer_area": 200,
        "min_ring_area": 80,
        "min_hole_ratio": 0.12,
        "max_ring_fill": 0.50,
        "min_ring_fill": 0.04,
        "max_interior_red": 0.48,
        "min_border_red": 0.06,
        "max_solid_interior": 0.75,
        "thin_ring_fill_max": 0.40,
        "aspect_min": 0.40,
        "aspect_max": 3.20,
        "min_center_v_norm": 0.16,
        "max_center_v_norm": 0.62,
        "table_v_norm_lo": 0.18,
        "table_v_norm_hi": 0.55,
        "prefer_table_weight": 1.20,
        "prefer_midsize_area": 3500,
        "prefer_square_weight": 0.25,
        "prefer_hole_weight": 0.70,
        "open_kernel": 2,
        "close_kernel": 3,
        "close_iters": 1,
        "dilate_kernel": 7,
        "dilate_iters": 2,
        "track_min_bbox_w": 22,
        "track_min_bbox_h": 16,
        "track_min_bbox_area": 500,
        "track_min_outer_area": 120,
        "track_min_ring_area": 50,
        "track_min_center_v_norm": 0.12,
        "track_max_center_v_norm": 0.70,
        "track_max_ring_fill": 0.55,
        "track_aspect_max": 3.50,
    }


def merge_red_frame_cfg(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = default_red_frame_cfg()
    if overrides:
        cfg.update(dict(overrides))
    return cfg
