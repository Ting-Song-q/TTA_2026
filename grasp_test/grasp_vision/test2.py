#!/usr/bin/env python3
# coding=utf8
"""ArmPi Pro：视觉定位红色放置框并将已抓取物资放入框内。

对应赛题任务：智能车携带物资后，通过视觉识别红色放置框（如装货区/
救援区指定物料放置区），用机械臂将物资放置到识别框内。

流程：
1. 放置前机械臂抬高到侦察位（夹爪保持闭合持物）
2. 高位侦察空心红色放置框（前伸俯视；找不到则 y/z/pitch 搜索），并保存识别结果
3. 原厂风格视觉伺服（舵机6 左右 / y_dis 前后）对准框中心（不对底盘做进/退补距）
4. 相对跟踪位额外前伸 place_y_extra；超臂上限则夹在 y_limits 内放置
5. 高位接近 → 下降至放置高度 → 张开夹爪释放 → 竖直抬起
6. 放置成功后：小车后退约 10cm，机械臂复位，夹爪保持张开

依赖（树莓派上）：
  source /opt/ros/melodic/setup.bash
  source ~/armpi_pro/devel/setup.bash
  已启动相机、舵机控制器、chassis_control
  建议先跑 grasp_vision/test1_grasp.py 完成抓取后再运行本脚本
"""

from __future__ import print_function

import argparse
import copy
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

SCRIPT_DIR = Path(__file__).resolve().parent
GRASP_TEST_ROOT = SCRIPT_DIR.parent
WORKSPACE_CANDIDATES = [
    GRASP_TEST_ROOT.parent / "lx" / "armpi_pro",
    Path.home() / "armpi_pro",
    Path("/home/ubuntu/armpi_pro"),
]


def _add_path(path):
    path = Path(path)
    if path.exists():
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def _bootstrap_armpi():
    for root in WORKSPACE_CANDIDATES:
        _add_path(root / "devel" / "lib" / "python3" / "dist-packages")
        _add_path(root / "devel" / "lib" / "python2.7" / "dist-packages")
        _add_path(root / "src" / "armpi_pro_common")
        _add_path(root / "src" / "armpi_pro_kinematics")


_bootstrap_armpi()

try:
    from hiwonder_servo_msgs.msg import MultiRawIdPosDur
    from armpi_pro import bus_servo_control, PID, Misc
    from kinematics import ik_transform
    from chassis_control.msg import SetVelocity
except Exception as exc:
    raise SystemExit(
        "无法导入 ArmPi Pro 模块（舵机/IK/底盘）。请先 source armpi_pro/devel/setup.bash。\n"
        "原始错误: {}".format(exc)
    )


# ---------------------------------------------------------------------------
# 可调参数（现场主要改这里）
# ---------------------------------------------------------------------------
CFG = {
    "image_topic": "/usb_cam/image_raw",
    "settle_frames": 12,
    "frame_timeout": 5.0,
    "save_debug": True,
    "debug_dir": str(SCRIPT_DIR / "debug" / "grasp_vision_test2"),
    # 红色空心放置框检测（细边/断边友好，抑制所持实心红块与远处误检）
    # 侦察 ROI：上边界下移，避开远处背景红噪声；下边界避开夹爪红块
    "roi": [0.08, 0.18, 0.92, 0.66],
    # 跟踪 ROI：略放宽，台面框更容易进视野
    "track_roi": [0.05, 0.12, 0.95, 0.72],
    "hsv_lower1": [0, 55, 35],
    "hsv_upper1": [20, 255, 255],
    "hsv_lower2": [160, 55, 35],
    "hsv_upper2": [180, 255, 255],
    "min_r": 80,
    "min_r_minus_g": 20,
    "min_r_minus_b": 15,
    "max_g": 220,
    # 几何门槛（侦察）
    "min_bbox_w": 28,
    "min_bbox_h": 22,
    "min_bbox_area": 900,         # 禁止顶部小红斑（旧假阳性约 30x26）
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
    "aspect_max": 3.20,           # 透视下台面框可很扁（实测约 2.9）
    # 位置约束：丢弃贴图像顶部的候选（整幅归一化 v）
    "min_center_v_norm": 0.16,    # center_v/h 过小 = 远处背景
    "max_center_v_norm": 0.62,    # 过低接近夹爪所持红块
    # 台面偏好带（打分加权，非整幅硬裁）
    "table_v_norm_lo": 0.18,
    "table_v_norm_hi": 0.55,
    "prefer_table_weight": 1.20,
    "prefer_midsize_area": 3500,   # 期望 bbox 面积量级（台面框）
    "prefer_square_weight": 0.25,
    "prefer_hole_weight": 0.70,
    "open_kernel": 2,
    "close_kernel": 3,
    "close_iters": 1,
    "dilate_kernel": 7,
    "dilate_iters": 2,
    # 跟踪模式：相对侦察放宽，便于对准时持续看到框
    "track_min_bbox_w": 22,
    "track_min_bbox_h": 16,
    "track_min_bbox_area": 500,
    "track_min_outer_area": 120,
    "track_min_ring_area": 50,
    "track_min_center_v_norm": 0.12,
    "track_max_center_v_norm": 0.70,
    "track_max_ring_fill": 0.55,
    "track_aspect_max": 3.50,
    # 放置前侦察：前伸+俯视，优先能看见台面红框的姿态
    "scout_xyz": [0.0, 0.16, 0.22],
    "scout_pitch": -115.0,
    "scout_pitch_range": [-145.0, -85.0],
    "scout_settle": 2.0,
    "scout_frames": 3,
    "scout_retry": 1,
    "scout_search_enable": True,
    # 每项 [x, y, z, pitch]；先基准，再微调前后/高度/俯仰
    "scout_search_poses": [
        [0.0, 0.16, 0.22, -115.0],
        [0.0, 0.18, 0.20, -118.0],
        [0.0, 0.14, 0.20, -112.0],
        [0.0, 0.18, 0.18, -120.0],
        [0.0, 0.16, 0.24, -110.0],
        [0.0, 0.12, 0.26, -105.0],
        [0.0, 0.20, 0.20, -118.0],
    ],
    "scout_search_settle": 1.2,
    "scout_search_move_ms": 1000,
    # 视觉伺服图像目标点（像素，v 向下增大）
    # y_center 偏大(偏下) → 框在目标上方时 err_v<0 → 臂前伸；偏小(偏上)则倾向收臂
    # 放置框多在台面中部，取约画面中线略偏上，避免误判过近而收臂
    "x_center": 320.0,
    "y_center": 280.0,
    "x_pid_p": 0.06,
    "y_pid_p": 0.00003,
    "x_dis_init": 500,
    "y_dis_init": 0.16,
    "x_limits": [200, 800],
    # 臂前伸工作范围；超出上限时放置阶段直接夹紧，不再底盘补距
    "y_limits": [0.12, 0.28],
    "stable_err_u": 28.0,
    "stable_err_v": 32.0,
    "stable_frames": 3,
    "track_timeout": 14.0,
    # 跟踪 / 放置笛卡尔位姿（米）；原厂约定 IK 的 x 恒为 0，左右靠舵机6
    "observe_xyz": [0.0, 0.16, 0.22],
    "observe_pitch": -115.0,
    "track_x": 0.0,
    "track_z": 0.22,
    "track_pitch": -115.0,
    "track_pitch_range": [-145.0, -90.0],
    # 防碰放置：先高位悬停，再下降，张开释放，再抬起
    # place_y_extra 加大：方块最终落点更靠前，进入红框内部（而非框前缘）
    "approach_z": 0.19,
    "place_z": 0.155,
    "place_y_extra": 0.06,
    "place_pitch": -120.0,
    "place_pitch_range": [-145.0, -90.0],
    "lift_z": 0.22,
    "open_servo": 120,
    "close_servo": 450,
    "wrist_servo": 500,
    # 底盘：放置前视觉伺服/放置补距不再进退；仅放置成功后后退使用
    "chassis_enable": True,
    "chassis_speed": 90.0,
    "chassis_move_time": 0.50,
    "chassis_cooldown": 0.6,
    "chassis_max_moves": 10,
    # 放置成功后：后退 + 臂复位（夹爪保持张开）
    "post_place_retreat_cm": 10.0,
    "post_place_retreat_speed": 100.0,
    "post_place_reset_xyz": [0.0, 0.15, 0.18],
    "post_place_reset_pitch": -115.0,
}


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

            # 1) 禁止图像顶部过小/过远候选（旧假阳性 v≈37）
            v_norm = float(cv_) / float(max(fh, 1))
            u_norm = float(cu) / float(max(fw, 1))
            if v_norm < self.min_center_v_norm or v_norm > self.max_center_v_norm:
                continue
            # 贴 ROI 顶边的小目标再砍一刀
            if (y <= 4) and bbox_area < self.min_bbox_area * 1.5:
                continue

            square_score = 1.0 - min(1.0, abs(math.log(max(aspect, 1e-6))))
            hollow_score = max(hole_ratio, max(0.0, 1.0 - interior_red), border_red)
            is_hollow = bool(has_hole or hollow_by_ratio or hollow_by_thin)

            # 2) 偏好台面中等大小环：面积接近 prefer_midsize，fill 适中，落在 table 带
            mid = max(self.prefer_midsize_area, 1.0)
            size_ratio = bbox_area / mid
            # 峰值在 1.0：过小/过大都降分
            midsize_score = math.exp(-abs(math.log(max(size_ratio, 1e-3))) * 0.85)
            fill_score = 1.0 - abs(ring_fill - 0.18)  # 台面细框 fill 常约 0.1~0.3
            fill_score = max(0.35, min(1.15, fill_score + 0.5))
            in_table = self.table_v_lo <= v_norm <= self.table_v_hi
            table_score = (1.0 + self.prefer_table_weight) if in_table else 0.55
            # 略偏好水平居中
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


class Camera(object):
    def __init__(self, topic, timeout, settle):
        self.topic = topic
        self.timeout = timeout
        self.settle = settle

    def _decode(self, msg):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        expected = int(msg.height) * int(msg.width) * 3
        if data.size == expected:
            image = data.reshape(msg.height, msg.width, 3)
            if msg.encoding == "bgr8":
                return image.copy()
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if msg.encoding in ("yuyv", "yuyv422", "yuv422"):
            image = data.reshape(msg.height, msg.width, 2)
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)
        if msg.encoding in ("bgr8", "8UC3"):
            return data.reshape(msg.height, msg.width, 3).copy()
        if msg.encoding == "rgb8":
            return cv2.cvtColor(data.reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)
        rospy.logwarn("unsupported encoding: %s", msg.encoding)
        return None

    def get_frame(self, discard=None):
        n = self.settle if discard is None else max(1, int(discard))
        frame = None
        for _ in range(n):
            try:
                msg = rospy.wait_for_message(self.topic, Image, timeout=self.timeout)
            except rospy.ROSException:
                return None
            frame = self._decode(msg)
            if frame is None:
                return None
        return frame


class FrontRedPlace(object):
    """前方红色放置框：视觉伺服对准 + 防碰放置（放置前不动底盘）。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.ik = ik_transform.ArmIK()
        self.detector = RedPlacementFrameDetector(cfg)
        self.camera = Camera(cfg["image_topic"], cfg["frame_timeout"], cfg["settle_frames"])
        self.joints_pub = rospy.Publisher(
            "/servo_controllers/port_id_1/multi_id_pos_dur", MultiRawIdPosDur, queue_size=1
        )
        self.chassis_pub = rospy.Publisher(
            "/chassis_control/set_velocity", SetVelocity, queue_size=1
        )
        self.x_pid = PID.PID(P=float(cfg["x_pid_p"]), I=0, D=0)
        self.y_pid = PID.PID(P=float(cfg["y_pid_p"]), I=0, D=0)
        self.debug_dir = Path(cfg["debug_dir"])
        self._dbg_i = 0
        self.x_dis = float(cfg["x_dis_init"])
        self.y_dis = float(cfg["y_dis_init"])
        self._chassis_moves = 0
        self._last_chassis_t = 0.0
        rospy.sleep(0.6)

    def _log(self, msg, *args):
        stamp = datetime.now().strftime("%H:%M:%S")
        text = msg % args if args else msg
        rospy.loginfo("[test2 %s] %s", stamp, text)

    def _set_servos(self, duration_ms, servos):
        bus_servo_control.set_servos(self.joints_pub, int(duration_ms), tuple(servos))

    def _ik_move(self, xyz, x_dis=None, duration_ms=800, settle=0.5,
                 alpha=None, alpha_range=None, gripper=None):
        if alpha is None:
            alpha = self.cfg["track_pitch"]
        if alpha_range is None:
            alpha_range = self.cfg["track_pitch_range"]
        target = self.ik.setPitchRanges(
            tuple(xyz), float(alpha), float(alpha_range[0]), float(alpha_range[1])
        )
        if not target:
            return None
        sd = target[1]
        servos = []
        if gripper is not None:
            servos.append((1, int(gripper)))
        servos.append((2, int(self.cfg["wrist_servo"])))
        servos.extend([
            (3, sd["servo3"]),
            (4, sd["servo4"]),
            (5, sd["servo5"]),
            (6, int(self.x_dis if x_dis is None else x_dis)),
        ])
        self._set_servos(duration_ms, servos)
        rospy.sleep(settle)
        return target

    def _save_debug(self, frame, tag, det=None):
        if not self.cfg["save_debug"] or frame is None:
            return None
        desired = (self.cfg["x_center"], self.cfg["y_center"])
        vis = self.detector.draw(frame, det, desired)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        path = self.debug_dir / "{:03d}_{}.jpg".format(self._dbg_i, tag)
        self._dbg_i += 1
        cv2.imwrite(str(path), vis)
        self._log("debug -> %s", path)
        return path

    def save_scout_result(self, frame, det, tag="scout"):
        """保存侦察阶段红色框识别结果：标注图、原图、mask、信息文本。"""
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        idx = self._dbg_i
        self._dbg_i += 1
        prefix = "{:03d}_{}_{}".format(idx, tag, stamp)

        desired = (self.cfg["x_center"], self.cfg["y_center"])
        annotated = self.detector.draw(frame, det, desired) if frame is not None else None
        paths = {}

        if annotated is not None:
            p = self.debug_dir / "{}_annotated.jpg".format(prefix)
            cv2.imwrite(str(p), annotated)
            paths["annotated"] = p
            # 兼容旧命名，再存一份简短 scout 图
            p2 = self.debug_dir / "{:03d}_scout.jpg".format(idx)
            cv2.imwrite(str(p2), annotated)
            paths["scout"] = p2

        if frame is not None:
            p = self.debug_dir / "{}_raw.jpg".format(prefix)
            cv2.imwrite(str(p), frame)
            paths["raw"] = p

        if det is not None and det.mask is not None:
            p = self.debug_dir / "{}_mask.jpg".format(prefix)
            cv2.imwrite(str(p), det.mask)
            paths["mask"] = p

        info_path = self.debug_dir / "{}_info.txt".format(prefix)
        with open(str(info_path), "w", encoding="utf-8") as f:
            f.write("stamp={}\n".format(stamp))
            f.write("found={}\n".format(det is not None))
            f.write("scout_xyz={}\n".format(self.cfg.get("scout_xyz")))
            f.write("scout_pitch={}\n".format(self.cfg.get("scout_pitch")))
            if det is not None:
                f.write("center_u={:.2f}\n".format(det.center_u))
                f.write("center_v={:.2f}\n".format(det.center_v))
                f.write("ring_area={:.1f}\n".format(det.area))
                f.write("ring_fill={:.4f}\n".format(det.rect_fill))
                f.write("hole_ratio={:.4f}\n".format(det.hole_ratio))
                f.write("has_hole={}\n".format(det.has_hole))
                f.write("aspect={:.4f}\n".format(det.aspect))
                f.write("bbox={}\n".format(det.bbox))
                f.write("roi={}\n".format(det.roi))
                f.write(
                    "err_u={:.2f}\nerr_v={:.2f}\n".format(
                        det.center_u - desired[0], det.center_v - desired[1]
                    )
                )
            else:
                f.write("reason=no_red_place_frame\n")
        paths["info"] = info_path

        self._log("scout result saved under %s (found=%s)", self.debug_dir, det is not None)
        for key, path in paths.items():
            self._log("  %s -> %s", key, path)
        return paths

    def stop_chassis(self):
        self.chassis_pub.publish(0.0, 0.0, 0.0)
        rospy.sleep(0.05)

    def move_chassis(self, direction_deg, duration=None, speed=None):
        """麦轮：direction 90=前进，270=后退。短时开环移动后停车。"""
        if not self.cfg["chassis_enable"]:
            return False
        if self._chassis_moves >= int(self.cfg["chassis_max_moves"]):
            self._log("chassis assist limit reached (%d)", self._chassis_moves)
            return False
        now = time.time()
        if now - self._last_chassis_t < float(self.cfg["chassis_cooldown"]):
            return False
        duration = float(self.cfg["chassis_move_time"] if duration is None else duration)
        speed = float(self.cfg["chassis_speed"] if speed is None else speed)
        self._log("chassis move dir=%.0f speed=%.0f time=%.2fs", direction_deg, speed, duration)
        self.chassis_pub.publish(speed, float(direction_deg), 0.0)
        rospy.sleep(duration)
        self.stop_chassis()
        self._chassis_moves += 1
        self._last_chassis_t = time.time()
        return True

    def go_scout_holding(self, xyz=None, pitch=None, settle=None, duration_ms=None):
        """放置前抬高机械臂到侦察位，夹爪保持闭合，便于俯视/前视红色放置框。"""
        cfg = self.cfg
        if xyz is None:
            xyz = list(cfg.get("scout_xyz", [0.0, 0.16, 0.22]))
        else:
            xyz = list(xyz)
        if pitch is None:
            pitch = float(cfg.get("scout_pitch", -115.0))
        else:
            pitch = float(pitch)
        pitch_range = cfg.get("scout_pitch_range", cfg["track_pitch_range"])
        if settle is None:
            settle = float(cfg.get("scout_settle", 2.0))
        if duration_ms is None:
            duration_ms = 1800
        self._log(
            "scout raise arm xyz=%s pitch=%.1f gripper closed=%d",
            xyz, pitch, int(cfg["close_servo"]),
        )
        ok = self._ik_move(
            xyz,
            x_dis=int(cfg["x_dis_init"]),
            duration_ms=int(duration_ms),
            settle=float(settle),
            alpha=pitch,
            alpha_range=pitch_range,
            gripper=cfg["close_servo"],
        )
        if ok is None:
            self._log("scout IK failed for xyz=%s, try nearby z", xyz)
            for dz in (0.0, -0.02, 0.02, -0.04, 0.04):
                alt = [xyz[0], xyz[1], round(xyz[2] + dz, 4)]
                ok = self._ik_move(
                    alt,
                    x_dis=int(cfg["x_dis_init"]),
                    duration_ms=int(duration_ms),
                    settle=float(settle),
                    alpha=pitch,
                    alpha_range=pitch_range,
                    gripper=cfg["close_servo"],
                )
                if ok is not None:
                    xyz = alt
                    break
        if ok is None:
            self._log("scout IK failed, fallback high pose")
            self._set_servos(
                1800,
                (
                    (1, int(cfg["close_servo"])),
                    (2, 500), (3, 60), (4, 750), (5, 550), (6, 500),
                ),
            )
            rospy.sleep(float(settle))
        self.x_dis = float(cfg["x_dis_init"])
        self.y_dis = float(xyz[1]) if len(xyz) > 1 else float(cfg["y_dis_init"])
        self.x_pid.clear()
        self.y_pid.clear()
        self._set_servos(300, ((1, int(cfg["close_servo"])),))
        rospy.sleep(0.25)
        return True

    def _scout_detect_at_height(self, attempt_tag):
        """在当前臂姿下采多帧检测红框，返回 (best_det, last_frame)。"""
        cfg = self.cfg
        n_frames = max(1, int(cfg.get("scout_frames", 3)))
        retries = max(0, int(cfg.get("scout_retry", 1)))
        best = None
        last_frame = None
        for attempt in range(retries + 1):
            for i in range(n_frames):
                frame = self.camera.get_frame(discard=2 if i > 0 else None)
                if frame is None:
                    rospy.logwarn("scout: no frame (%s attempt=%d i=%d)", attempt_tag, attempt, i)
                    continue
                last_frame = frame
                det = self.detector.detect(frame, mode="scout")
                self._save_debug(frame, "scout_{}_{}_{}".format(attempt_tag, attempt, i), det)
                if det is None:
                    continue
                if best is None or (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1]) > (
                    best.bbox[2] - best.bbox[0]
                ) * (best.bbox[3] - best.bbox[1]):
                    best = det
                self._log(
                    "scout hit [%s] u=%.0f v=%.0f ring=%.0f hole=%.2f",
                    attempt_tag, det.center_u, det.center_v, det.area, det.hole_ratio,
                )
            if best is not None:
                break
        return best, last_frame

    def _scout_pose_list(self):
        """生成侦察搜索姿态列表：[x,y,z,pitch]。"""
        cfg = self.cfg
        base = list(cfg.get("scout_xyz", [0.0, 0.16, 0.22]))
        base_pitch = float(cfg.get("scout_pitch", -115.0))
        poses = cfg.get("scout_search_poses")
        out = []
        seen = set()

        def _add(x, y, z, pitch):
            key = (round(float(x), 4), round(float(y), 4), round(float(z), 4), round(float(pitch), 1))
            if key in seen:
                return
            seen.add(key)
            out.append([key[0], key[1], key[2], key[3]])

        if poses:
            for p in poses:
                if len(p) >= 4:
                    _add(p[0], p[1], p[2], p[3])
                elif len(p) == 3:
                    _add(p[0], p[1], p[2], base_pitch)
        else:
            # 兼容旧配置：只给 z 列表
            z_list = cfg.get("scout_search_z", [base[2]])
            for z in z_list:
                _add(base[0], base[1], z, base_pitch)
        if not out:
            _add(base[0], base[1], base[2], base_pitch)
        if not bool(cfg.get("scout_search_enable", True)):
            return [out[0]]
        return out

    def scout_place_frame(self):
        """高位侦察红色放置框；找不到则上下/前后/俯仰搜索，并保存识别结果。"""
        cfg = self.cfg
        poses = self._scout_pose_list()
        best = None
        last_frame = None
        found_pose = None

        self._log("=== scout red place frame start (poses=%d) ===", len(poses))
        move_ms = int(cfg.get("scout_search_move_ms", 1000))
        search_settle = float(cfg.get("scout_search_settle", 1.2))
        base = list(cfg.get("scout_xyz", [0.0, 0.16, 0.22]))
        base_pitch = float(cfg.get("scout_pitch", -115.0))

        for hi, pose in enumerate(poses):
            xyz = [pose[0], pose[1], pose[2]]
            pitch = float(pose[3])
            tag = "y{:.0f}_z{:.0f}_p{:.0f}".format(xyz[1] * 1000, xyz[2] * 1000, abs(pitch))
            same_as_base = (
                abs(xyz[0] - base[0]) < 1e-4
                and abs(xyz[1] - base[1]) < 1e-4
                and abs(xyz[2] - base[2]) < 1e-4
                and abs(pitch - base_pitch) < 1e-3
            )
            if hi == 0 and same_as_base:
                pass  # run() 已到基准侦察位
            else:
                self._log("scout search move -> xyz=%s pitch=%.1f", xyz, pitch)
                self.go_scout_holding(
                    xyz=xyz, pitch=pitch, settle=search_settle, duration_ms=move_ms
                )

            det, frame = self._scout_detect_at_height(tag)
            if frame is not None:
                last_frame = frame
            if det is not None:
                best = det
                found_pose = xyz + [pitch]
                self._log("scout found at xyz=%s pitch=%.1f", xyz, pitch)
                break
            self._log("scout miss at %s (%d/%d)", tag, hi + 1, len(poses))

        save_frame = last_frame
        if best is not None:
            fresh = self.camera.get_frame(discard=2)
            if fresh is not None:
                save_frame = fresh
                redet = self.detector.detect(fresh, mode="scout")
                if redet is not None:
                    best = redet
        paths = self.save_scout_result(save_frame, best, tag="scout")
        if found_pose is not None and paths.get("info"):
            try:
                with open(str(paths["info"]), "a", encoding="utf-8") as f:
                    f.write("found_pose={}\n".format(found_pose))
                    f.write("search_poses={}\n".format(poses))
            except Exception:
                pass

        if best is None:
            rospy.logerr("scout: 姿态搜索后仍未检测到红色放置框")
            return None
        self._log(
            "scout ok u=%.0f v=%.0f ring=%.0f fill=%.2f hole=%.2f asp=%.2f pose=%s",
            best.center_u, best.center_v, best.area, best.rect_fill,
            best.hole_ratio, best.aspect, found_pose,
        )
        return best

    def go_observe_holding(self):
        """跟踪观察位：夹爪保持闭合（携带物资）。"""
        xyz = self.cfg["observe_xyz"]
        self._log("observe xyz=%s hold gripper closed=%d", xyz, int(self.cfg["close_servo"]))
        ok = self._ik_move(
            xyz,
            x_dis=int(self.cfg["x_dis_init"]),
            duration_ms=1600,
            settle=1.8,
            alpha=self.cfg["observe_pitch"],
            alpha_range=self.cfg["track_pitch_range"],
            gripper=self.cfg["close_servo"],
        )
        if ok is None:
            self._set_servos(
                1500,
                (
                    (1, int(self.cfg["close_servo"])),
                    (2, 500), (3, 80), (4, 825), (5, 625), (6, 500),
                ),
            )
            rospy.sleep(1.8)
        self.x_dis = float(self.cfg["x_dis_init"])
        self.y_dis = float(self.cfg["y_dis_init"])
        self.x_pid.clear()
        self.y_pid.clear()

    def visual_servo_align(self):
        """视觉伺服对准红色放置框（仅机械臂，放置前不动底盘）。返回最后一次检测。"""
        cfg = self.cfg
        self.detector.set_mode("track")
        x_center = float(cfg["x_center"])
        y_center = float(cfg["y_center"])
        x_low, x_high = cfg["x_limits"]
        y_low, y_high = cfg["y_limits"]
        stable_need = int(cfg["stable_frames"])
        stable_count = 0
        deadline = time.time() + float(cfg["track_timeout"])
        last_det = None
        first = True

        # 对准过程保持夹爪闭合，防止物资掉落
        self._log("keep gripper closed while tracking place frame (mode=track)")
        self._set_servos(400, ((1, int(cfg["close_servo"])),))
        rospy.sleep(0.35)

        while time.time() < deadline and not rospy.is_shutdown():
            frame = self.camera.get_frame(discard=2)
            if frame is None:
                rospy.logwarn("no camera frame")
                continue
            det = self.detector.detect(frame, mode="track")
            self._save_debug(frame, "track", det)
            if det is None:
                stable_count = 0
                rospy.sleep(0.05)
                continue

            err_u = det.center_u - x_center
            err_v = det.center_v - y_center
            last_det = det

            if abs(err_u) < 10:
                self.x_pid.SetPoint = det.center_u
            else:
                self.x_pid.SetPoint = x_center
            self.x_pid.update(det.center_u)
            dx = self.x_pid.output

            if abs(err_v) < 10:
                self.y_pid.SetPoint = det.center_v
            else:
                self.y_pid.SetPoint = y_center
            self.y_pid.update(det.center_v)
            dy = self.y_pid.output

            self.x_dis = max(float(x_low), min(float(x_high), self.x_dis + dx))
            self.y_dis = max(float(y_low), min(float(y_high), self.y_dis + dy))

            xyz = [float(cfg["track_x"]), round(self.y_dis, 4), float(cfg["track_z"])]
            duration = 600 if first else 120
            settle = 0.25 if first else 0.08
            first = False
            moved = self._ik_move(
                xyz,
                x_dis=int(self.x_dis),
                duration_ms=duration,
                settle=settle,
                alpha=cfg["track_pitch"],
                alpha_range=cfg["track_pitch_range"],
                gripper=cfg["close_servo"],
            )
            self._log(
                "servo err_u=%+.1f err_v=%+.1f x_dis=%d y_dis=%.4f ik=%s",
                err_u, err_v, int(self.x_dis), self.y_dis, bool(moved),
            )

            if abs(err_u) < float(cfg["stable_err_u"]) and abs(err_v) < float(cfg["stable_err_v"]):
                stable_count += 1
                if stable_count >= stable_need:
                    self._log("aligned stable_count=%d", stable_count)
                    break
            else:
                stable_count = 0

        if last_det is None:
            return None
        if stable_count < stable_need:
            self._log("align timeout, use last pose x_dis=%d y_dis=%.4f", int(self.x_dis), self.y_dis)
        return last_det

    def collision_safe_place(self):
        """高位接近 → 下降 → 张开释放 → 竖直抬起。全程夹爪先合后开，避免提前掉落。

        相对跟踪位额外前伸 place_y_extra，使方块落在红框内部；
        超出臂工作半径时夹在 y_limits 上限，不再底盘前进补距。
        """
        cfg = self.cfg
        y_low, y_high = float(cfg["y_limits"][0]), float(cfg["y_limits"][1])
        desired_y = round(self.y_dis + float(cfg["place_y_extra"]), 4)
        x_dis = int(self.x_dis)

        if desired_y > y_high:
            self._log(
                "place y=%.4f exceeds arm max=%.4f by %.1fcm -> clamp (no chassis)",
                desired_y, y_high, (desired_y - y_high) * 100.0,
            )
            y = y_high
        else:
            y = max(y_low, desired_y)
        self._log("place reach y=%.4f (track y_dis=%.4f + extra=%.3f)",
                  y, self.y_dis, float(cfg["place_y_extra"]))

        # 1) 确认闭合（仍持有物资）
        self._log("place step1: ensure gripper closed (holding)")
        self._set_servos(400, ((1, int(cfg["close_servo"])),))
        rospy.sleep(0.4)

        # 2) 高位悬停（尽量用目标前伸 y）
        approach = [0.0, y, float(cfg["approach_z"])]
        self._log("place step2: approach hover xyz=%s", approach)
        if self._ik_move(
            approach, x_dis=x_dis, duration_ms=900, settle=1.0,
            alpha=cfg["place_pitch"], alpha_range=cfg["place_pitch_range"],
            gripper=cfg["close_servo"],
        ) is None:
            self._log("approach IK failed, try slightly shorter y / track_z")
            for y_try in (round(y - 0.01, 4), round(y - 0.02, 4), round(self.y_dis, 4)):
                y_try = max(y_low, min(y_high, y_try))
                approach = [0.0, y_try, float(cfg["track_z"])]
                if self._ik_move(
                    approach, x_dis=x_dis, duration_ms=900, settle=1.0,
                    alpha=cfg["track_pitch"], alpha_range=cfg["track_pitch_range"],
                    gripper=cfg["close_servo"],
                ) is not None:
                    y = y_try
                    break
            else:
                rospy.logerr("no IK for approach")
                return False

        # 3) 竖直下降到放置高度（保持前伸 y，让方块进框）
        place_xyz = [0.0, y, float(cfg["place_z"])]
        self._log("place step3: descend xyz=%s", place_xyz)
        candidates = [
            (place_xyz, cfg["place_pitch"], cfg["place_pitch_range"]),
            ([0.0, y, float(cfg["approach_z"]) - 0.02], cfg["place_pitch"], cfg["place_pitch_range"]),
            ([0.0, round(max(y_low, y - 0.01), 4), float(cfg["place_z"])],
             cfg["place_pitch"], cfg["place_pitch_range"]),
            ([0.0, round(self.y_dis, 4), float(cfg["place_z"])], cfg["track_pitch"], cfg["track_pitch_range"]),
        ]
        moved = None
        for xyz, pitch, prange in candidates:
            moved = self._ik_move(
                xyz, x_dis=x_dis, duration_ms=1000, settle=1.2,
                alpha=pitch, alpha_range=prange, gripper=cfg["close_servo"],
            )
            if moved is not None:
                y = float(xyz[1])
                self._log("descend ok xyz=%s", xyz)
                break
        if moved is None:
            rospy.logerr("no IK for place descend")
            return False

        # 4) 张开夹爪释放物资
        self._log("place step4: open gripper to release")
        self._set_servos(500, ((1, int(cfg["open_servo"])),))
        rospy.sleep(0.9)

        # 5) 抬高离开放置区
        lift = [0.0, y, float(cfg["lift_z"])]
        self._log("place step5: lift xyz=%s", lift)
        if self._ik_move(
            lift, x_dis=x_dis, duration_ms=1000, settle=1.0,
            alpha=cfg["track_pitch"], alpha_range=cfg["track_pitch_range"],
            gripper=cfg["open_servo"],
        ) is None:
            self._set_servos(
                1200,
                (
                    (1, int(cfg["open_servo"])),
                    (2, 500), (3, 80), (4, 825), (5, 625), (6, 500),
                ),
            )
            rospy.sleep(1.3)

        self._log("place finished (release at y=%.4f)", y)
        return True

    def retreat_and_reset_open(self):
        """放置后：后退指定距离 → 臂复位，夹爪保持张开。"""
        cfg = self.cfg
        if not cfg.get("chassis_enable", True):
            self._log("post-place: chassis disabled, skip retreat")
        else:
            dist_mm = float(cfg.get("post_place_retreat_cm", 10.0)) * 10.0
            speed = float(cfg.get("post_place_retreat_speed", 100.0))
            duration = max(0.1, dist_mm / max(speed, 1.0))

            self._log(
                "post-place: retreat %.0fcm (speed=%.0fmm/s, time=%.2fs)",
                dist_mm / 10.0, speed, duration,
            )
            self.chassis_pub.publish(speed, 270.0, 0.0)
            rospy.sleep(duration)
            self.stop_chassis()
            rospy.sleep(0.2)

        reset_xyz = cfg.get("post_place_reset_xyz", cfg["observe_xyz"])
        reset_pitch = float(cfg.get("post_place_reset_pitch", cfg["observe_pitch"]))
        self._log("post-place: arm reset xyz=%s, gripper stay open=%d",
                  reset_xyz, int(cfg["open_servo"]))
        moved = self._ik_move(
            reset_xyz,
            x_dis=500,
            duration_ms=1600,
            settle=1.8,
            alpha=reset_pitch,
            alpha_range=cfg["track_pitch_range"],
            gripper=cfg["open_servo"],
        )
        if moved is None:
            self._set_servos(
                1500,
                (
                    (1, int(cfg["open_servo"])),
                    (2, 500), (3, 80), (4, 825), (5, 625), (6, 500),
                ),
            )
            rospy.sleep(1.8)
        self._set_servos(300, ((1, int(cfg["open_servo"])),))
        rospy.sleep(0.3)
        self._log("post-place retreat+reset done (open)")
        return True

    def run(self, dry_run=False):
        self._log("=== front red place-frame place start ===")
        self.stop_chassis()

        # 1) 放置前：机械臂抬高侦察位
        self.go_scout_holding()

        self._log("wait camera after scout raise...")
        frame = self.camera.get_frame()
        if frame is None:
            rospy.logerr("camera not ready on %s", self.cfg["image_topic"])
            return False
        rospy.sleep(0.5)

        # 2) 高位侦察红色框，并保存识别结果
        det = self.scout_place_frame()
        if det is None:
            rospy.logerr("放置前侦察失败：未检测到红色放置框")
            return False

        if dry_run:
            self._log("dry-run: scout done, skip servo/chassis/place")
            return True

        # 3) 转入跟踪观察位后视觉伺服对准
        self.go_observe_holding()
        aligned = self.visual_servo_align()
        if aligned is None:
            rospy.logerr("visual align failed")
            self.stop_chassis()
            return False

        frame = self.camera.get_frame(discard=2)
        self._save_debug(frame, "aligned", aligned)

        ok = self.collision_safe_place()
        self.stop_chassis()

        after = self.camera.get_frame(discard=2)
        self._save_debug(after, "after_place", self.detector.detect(after, mode="track") if after is not None else None)

        if ok:
            self.retreat_and_reset_open()
        return ok


def parse_args():
    p = argparse.ArgumentParser(description="视觉定位红色放置框并将物资放入框内")
    p.add_argument("--dry-run", action="store_true", help="只检测并保存调试图，不运动")
    p.add_argument("--no-chassis", action="store_true", help="禁用放置后底盘后退")
    return p.parse_args()


def main():
    args = parse_args()
    rospy.init_node("grasp_vision_test2", anonymous=True)
    cfg = copy.deepcopy(CFG)
    if args.no_chassis:
        cfg["chassis_enable"] = False
    runner = FrontRedPlace(cfg)
    ok = runner.run(dry_run=args.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
