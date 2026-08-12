#!/usr/bin/env python3
# coding=utf8
"""ArmPi Pro：视觉定位前方红色方块并抓取。

流程：
1. 机械臂到前向观察位，夹爪完全张开
2. HSV+RGB 检测红色方块中心
3. 原厂风格视觉伺服（舵机6 左右 / y_dis 前后）对准
4. 若臂工作半径不够，麦轮底盘前进/后退补距后再对准
5. 高位接近 → 下降抓取 → 闭合 → 竖直抬起（减少夹爪与桌面/车体碰撞）
6. 抓取成功后：小车后退约 10cm，机械臂复位，夹爪保持闭合以便后续移动

依赖（树莓派上）：
  source /opt/ros/melodic/setup.bash
  source ~/armpi_pro/devel/setup.bash
  已启动相机、舵机控制器、chassis_control
"""

from __future__ import print_function

import argparse
import copy
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
    "debug_dir": str(Path.home() / "Desktop" / "grasp_test" / "debug" / "grasp_vision_test1"),
    # 红色检测
    "roi": [0.08, 0.15, 0.92, 0.92],
    "hsv_lower1": [0, 75, 45],
    "hsv_upper1": [18, 255, 255],
    "hsv_lower2": [165, 75, 45],
    "hsv_upper2": [180, 255, 255],
    "min_r": 95,
    "min_r_minus_g": 28,
    "min_r_minus_b": 22,
    "max_g": 205,
    "min_area": 500,
    "min_rect_fill": 0.40,
    # 视觉伺服图像目标点（640x480 前向观察位）
    "x_center": 320.0,
    "y_center": 395.0,
    "x_pid_p": 0.06,
    "y_pid_p": 0.00003,
    "x_dis_init": 500,
    "y_dis_init": 0.15,
    "x_limits": [200, 800],
    "y_limits": [0.12, 0.26],
    "stable_err_u": 22.0,
    "stable_err_v": 28.0,
    "stable_frames": 3,
    "track_timeout": 14.0,
    # 跟踪 / 抓取笛卡尔位姿（米）；原厂约定 IK 的 x 恒为 0，左右靠舵机6
    "observe_xyz": [0.0, 0.15, 0.18],
    "observe_pitch": -115.0,
    "track_x": 0.0,
    "track_z": 0.21,
    "track_pitch": -115.0,
    "track_pitch_range": [-145.0, -90.0],
    # 防碰：先高位悬停，再下降
    "approach_z": 0.18,
    "grasp_z": 0.16,
    "grasp_y_extra": 0.04,
    "grasp_pitch": -120.0,
    "grasp_pitch_range": [-145.0, -90.0],
    "lift_z": 0.22,
    "open_servo": 120,
    "close_servo": 450,
    "wrist_servo": 500,  # 舵机2 保持中位，减少侧向刮擦
    # 底盘补距：臂饱和且像素误差仍大时移动小车
    "chassis_enable": True,
    "chassis_speed": 80.0,  # mm/s
    "chassis_move_time": 0.45,  # s
    "chassis_cooldown": 0.8,
    "chassis_max_moves": 6,
    "chassis_pixel_trigger": 35.0,  # 饱和后 |err| 仍大于此才动车
    # 抓取成功后：后退 + 臂复位（夹爪保持闭合）
    "post_grasp_retreat_cm": 10.0,       # 后退距离 cm
    "post_grasp_retreat_speed": 100.0,   # mm/s；时间 = 距离/速度
    "post_grasp_reset_xyz": [0.0, 0.15, 0.18],  # 复位笛卡尔位姿
    "post_grasp_reset_pitch": -115.0,
}


class RedDetection(object):
    def __init__(self, center_u, center_v, area, bbox, contour, mask, roi, rect_fill):
        self.center_u = center_u
        self.center_v = center_v
        self.area = area
        self.bbox = bbox
        self.contour = contour
        self.mask = mask
        self.roi = roi
        self.rect_fill = rect_fill


class RedDetector(object):
    def __init__(self, cfg):
        self.roi_norm = cfg["roi"]
        self.lower1 = np.array(cfg["hsv_lower1"], dtype=np.uint8)
        self.upper1 = np.array(cfg["hsv_upper1"], dtype=np.uint8)
        self.lower2 = np.array(cfg["hsv_lower2"], dtype=np.uint8)
        self.upper2 = np.array(cfg["hsv_upper2"], dtype=np.uint8)
        self.min_r = int(cfg["min_r"])
        self.min_r_minus_g = int(cfg["min_r_minus_g"])
        self.min_r_minus_b = int(cfg["min_r_minus_b"])
        self.max_g = int(cfg["max_g"])
        self.min_area = float(cfg["min_area"])
        self.min_rect_fill = float(cfg["min_rect_fill"])

    def _roi(self, frame):
        h, w = frame.shape[:2]
        x1n, y1n, x2n, y2n = self.roi_norm
        x1 = max(0, min(w - 1, int(w * x1n)))
        y1 = max(0, min(h - 1, int(h * y1n)))
        x2 = max(x1 + 1, min(w, int(w * x2n)))
        y2 = max(y1 + 1, min(h, int(h * y2n)))
        return x1, y1, x2, y2

    def detect(self, frame):
        if frame is None:
            return None
        x1, y1, x2, y2 = self._roi(frame)
        crop = frame[y1:y2, x1:x2]
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
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            fill = area / float(max(w * h, 1))
            if fill < self.min_rect_fill:
                continue
            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue
            cu = m["m10"] / m["m00"] + x1
            cv_ = m["m01"] / m["m00"] + y1
            score = area * fill
            if score > best_score:
                best_score = score
                best = RedDetection(
                    cu, cv_, area,
                    (x + x1, y + y1, x + x1 + w, y + y1 + h),
                    contour.copy(), mask.copy(), (x1, y1, x2, y2), fill,
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
            cv2.putText(vis, "no red block", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return vis
        bx1, by1, bx2, by2 = det.bbox
        contour = det.contour + np.array([[[x1, y1]]], dtype=np.int32)
        cv2.drawContours(vis, [contour], -1, (0, 255, 255), 2)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        cv2.circle(vis, (int(det.center_u), int(det.center_v)), 5, (255, 0, 0), -1)
        if desired is not None:
            cv2.putText(
                vis,
                "err_u={:+.0f} err_v={:+.0f}".format(det.center_u - desired[0], det.center_v - desired[1]),
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


class FrontRedGrasp(object):
    """前方红色方块：视觉伺服 + 可选底盘补距 + 防碰抓取。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.ik = ik_transform.ArmIK()
        self.detector = RedDetector(cfg)
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
        rospy.loginfo("[test1 %s] %s", stamp, text)

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
            return
        desired = (self.cfg["x_center"], self.cfg["y_center"])
        vis = self.detector.draw(frame, det, desired)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        path = self.debug_dir / "{:03d}_{}.jpg".format(self._dbg_i, tag)
        self._dbg_i += 1
        cv2.imwrite(str(path), vis)
        self._log("debug -> %s", path)

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

    def go_observe(self):
        """观察位：夹爪张开、臂抬高，避免后续扫到方块/车体。"""
        xyz = self.cfg["observe_xyz"]
        self._log("observe xyz=%s open gripper", xyz)
        ok = self._ik_move(
            xyz,
            x_dis=int(self.cfg["x_dis_init"]),
            duration_ms=1600,
            settle=1.8,
            alpha=self.cfg["observe_pitch"],
            alpha_range=self.cfg["track_pitch_range"],
            gripper=self.cfg["open_servo"],
        )
        if ok is None:
            # 兜底：原厂抬臂姿态
            self._set_servos(
                1500,
                (
                    (1, int(self.cfg["open_servo"])),
                    (2, 500), (3, 80), (4, 825), (5, 625), (6, 500),
                ),
            )
            rospy.sleep(1.8)
        self.x_dis = float(self.cfg["x_dis_init"])
        self.y_dis = float(self.cfg["y_dis_init"])
        self.x_pid.clear()
        self.y_pid.clear()

    def _maybe_chassis_assist(self, err_u, err_v, x_low, x_high, y_low, y_high):
        """臂已顶到工作半径，像素误差仍大 → 动底盘把目标送进臂可达区。"""
        trigger = float(self.cfg["chassis_pixel_trigger"])
        # 前后：画面中目标偏上(err_v<0)通常更远 → 前进；偏下更近 → 后退
        if self.y_dis >= y_high - 1e-6 and err_v < -trigger:
            return self.move_chassis(90.0)
        if self.y_dis <= y_low + 1e-6 and err_v > trigger:
            return self.move_chassis(270.0)
        # 左右：舵机6 饱和后用底盘原地旋转微调
        if self.x_dis >= x_high - 1e-6 and err_u > trigger:
            self._log("chassis spin right (x saturated)")
            self.chassis_pub.publish(0.0, 90.0, -0.35)
            rospy.sleep(float(self.cfg["chassis_move_time"]))
            self.stop_chassis()
            self._chassis_moves += 1
            self._last_chassis_t = time.time()
            return True
        if self.x_dis <= x_low + 1e-6 and err_u < -trigger:
            self._log("chassis spin left (x saturated)")
            self.chassis_pub.publish(0.0, 90.0, 0.35)
            rospy.sleep(float(self.cfg["chassis_move_time"]))
            self.stop_chassis()
            self._chassis_moves += 1
            self._last_chassis_t = time.time()
            return True
        return False

    def visual_servo_align(self):
        """视觉伺服对准红色方块；必要时底盘补距。返回最后一次检测。"""
        cfg = self.cfg
        x_center = float(cfg["x_center"])
        y_center = float(cfg["y_center"])
        x_low, x_high = cfg["x_limits"]
        y_low, y_high = cfg["y_limits"]
        stable_need = int(cfg["stable_frames"])
        stable_count = 0
        deadline = time.time() + float(cfg["track_timeout"])
        last_det = None
        first = True

        self._log("open gripper before tracking")
        self._set_servos(400, ((1, int(cfg["open_servo"])),))
        rospy.sleep(0.35)

        while time.time() < deadline and not rospy.is_shutdown():
            frame = self.camera.get_frame(discard=2)
            if frame is None:
                rospy.logwarn("no camera frame")
                continue
            det = self.detector.detect(frame)
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

            # 饱和且误差大 → 动车，然后重新观察位伺服
            if self._maybe_chassis_assist(err_u, err_v, x_low, x_high, y_low, y_high):
                self.go_observe()
                stable_count = 0
                first = True
                continue

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

    def collision_safe_grasp(self):
        """高位接近 → 下降 → 闭合 → 竖直抬起。夹爪全程先开后合，避免侧扫。"""
        cfg = self.cfg
        y = round(self.y_dis + float(cfg["grasp_y_extra"]), 4)
        y = max(float(cfg["y_limits"][0]), min(float(cfg["y_limits"][1]), y))
        x_dis = int(self.x_dis)

        # 1) 确认张开
        self._log("grasp step1: ensure gripper open")
        self._set_servos(400, ((1, int(cfg["open_servo"])),))
        rospy.sleep(0.4)

        # 2) 高位悬停（接近高度，仍高于方块）
        approach = [0.0, y, float(cfg["approach_z"])]
        self._log("grasp step2: approach hover xyz=%s", approach)
        if self._ik_move(
            approach, x_dis=x_dis, duration_ms=900, settle=1.0,
            alpha=cfg["grasp_pitch"], alpha_range=cfg["grasp_pitch_range"],
            gripper=cfg["open_servo"],
        ) is None:
            self._log("approach IK failed, try track_z")
            approach = [0.0, y, float(cfg["track_z"])]
            if self._ik_move(
                approach, x_dis=x_dis, duration_ms=900, settle=1.0,
                alpha=cfg["track_pitch"], alpha_range=cfg["track_pitch_range"],
                gripper=cfg["open_servo"],
            ) is None:
                rospy.logerr("no IK for approach")
                return False

        # 3) 竖直下降到抓取高度（不左右摆）
        grasp_xyz = [0.0, y, float(cfg["grasp_z"])]
        self._log("grasp step3: descend xyz=%s", grasp_xyz)
        candidates = [
            (grasp_xyz, cfg["grasp_pitch"], cfg["grasp_pitch_range"]),
            ([0.0, y, float(cfg["approach_z"]) - 0.02], cfg["grasp_pitch"], cfg["grasp_pitch_range"]),
            ([0.0, round(self.y_dis, 4), float(cfg["grasp_z"])], cfg["track_pitch"], cfg["track_pitch_range"]),
        ]
        moved = None
        for xyz, pitch, prange in candidates:
            moved = self._ik_move(
                xyz, x_dis=x_dis, duration_ms=1000, settle=1.2,
                alpha=pitch, alpha_range=prange, gripper=cfg["open_servo"],
            )
            if moved is not None:
                self._log("descend ok xyz=%s", xyz)
                break
        if moved is None:
            rospy.logerr("no IK for grasp descend")
            return False

        # 4) 闭合夹爪
        self._log("grasp step4: close gripper")
        self._set_servos(500, ((1, int(cfg["close_servo"])),))
        rospy.sleep(0.85)

        # 5) 先抬高再回收，减少拖刮
        lift = [0.0, y, float(cfg["lift_z"])]
        self._log("grasp step5: lift xyz=%s", lift)
        if self._ik_move(
            lift, x_dis=x_dis, duration_ms=1000, settle=1.0,
            alpha=cfg["track_pitch"], alpha_range=cfg["track_pitch_range"],
            gripper=cfg["close_servo"],
        ) is None:
            self._set_servos(
                1200,
                (
                    (1, int(cfg["close_servo"])),
                    (2, 500), (3, 80), (4, 825), (5, 625), (6, 500),
                ),
            )
            rospy.sleep(1.3)

        self._log("grasp finished")
        return True

    def retreat_and_reset_holding(self):
        """抓取后：后退指定距离 → 臂复位，夹爪保持闭合。"""
        cfg = self.cfg
        dist_mm = float(cfg.get("post_grasp_retreat_cm", 10.0)) * 10.0
        speed = float(cfg.get("post_grasp_retreat_speed", 100.0))
        duration = max(0.1, dist_mm / max(speed, 1.0))

        self._log(
            "post-grasp: retreat %.0fcm (speed=%.0fmm/s, time=%.2fs)",
            dist_mm / 10.0, speed, duration,
        )
        # 270° = 后退；不受视觉补距次数限制
        self.chassis_pub.publish(speed, 270.0, 0.0)
        rospy.sleep(duration)
        self.stop_chassis()
        rospy.sleep(0.2)

        reset_xyz = cfg.get("post_grasp_reset_xyz", cfg["observe_xyz"])
        reset_pitch = float(cfg.get("post_grasp_reset_pitch", cfg["observe_pitch"]))
        self._log("post-grasp: arm reset xyz=%s, gripper stay closed=%d",
                  reset_xyz, int(cfg["close_servo"]))
        moved = self._ik_move(
            reset_xyz,
            x_dis=500,
            duration_ms=1600,
            settle=1.8,
            alpha=reset_pitch,
            alpha_range=cfg["track_pitch_range"],
            gripper=cfg["close_servo"],
        )
        if moved is None:
            # 兜底：原厂抬臂姿态，夹爪仍闭合
            self._set_servos(
                1500,
                (
                    (1, int(cfg["close_servo"])),
                    (2, 500), (3, 80), (4, 825), (5, 625), (6, 500),
                ),
            )
            rospy.sleep(1.8)
        # 再发一次闭合，避免复位过程中夹爪被带开
        self._set_servos(300, ((1, int(cfg["close_servo"])),))
        rospy.sleep(0.3)
        self._log("post-grasp retreat+reset done (holding)")
        return True

    def run(self, dry_run=False):
        self._log("=== front red block grasp start ===")
        self.stop_chassis()
        self.go_observe()

        self._log("wait camera...")
        frame = self.camera.get_frame()
        if frame is None:
            rospy.logerr("camera not ready on %s", self.cfg["image_topic"])
            return False
        rospy.sleep(0.8)

        det = self.detector.detect(frame)
        self._save_debug(frame, "detect0", det)
        if det is None:
            # 再采几帧
            for i in range(5):
                frame = self.camera.get_frame(discard=2)
                det = self.detector.detect(frame) if frame is not None else None
                self._save_debug(frame, "detect{}".format(i + 1), det)
                if det is not None:
                    break
                rospy.sleep(0.15)
        if det is None:
            rospy.logerr("前方未检测到红色方块")
            return False
        self._log("found red u=%.0f v=%.0f area=%.0f", det.center_u, det.center_v, det.area)

        if dry_run:
            self._log("dry-run: skip servo/chassis/grasp")
            return True

        aligned = self.visual_servo_align()
        if aligned is None:
            rospy.logerr("visual align failed")
            self.stop_chassis()
            return False

        frame = self.camera.get_frame(discard=2)
        self._save_debug(frame, "aligned", aligned)

        ok = self.collision_safe_grasp()
        self.stop_chassis()

        after = self.camera.get_frame(discard=2)
        self._save_debug(after, "after_grasp", self.detector.detect(after) if after is not None else None)

        if ok:
            self.retreat_and_reset_holding()
        return ok


def parse_args():
    p = argparse.ArgumentParser(description="视觉定位并抓取小车前方红色方块")
    p.add_argument("--dry-run", action="store_true", help="只检测并保存调试图，不运动")
    p.add_argument("--no-chassis", action="store_true", help="禁用底盘补距")
    return p.parse_args()


def main():
    args = parse_args()
    rospy.init_node("grasp_vision_test1", anonymous=True)
    cfg = copy.deepcopy(CFG)
    if args.no_chassis:
        cfg["chassis_enable"] = False
    runner = FrontRedGrasp(cfg)
    ok = runner.run(dry_run=args.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
