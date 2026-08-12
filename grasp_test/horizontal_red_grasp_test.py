#!/usr/bin/env python3
# coding=UTF-8

"""ArmPi Pro front-facing grasp test.

Main idea:
1. Keep a fixed, known-good forward observe pose for the camera.
2. Detect a front object in image space.
3. Use ArmPi Pro style visual servoing:
   - x_dis handles left/right approach
   - y_dis handles front/back reach
4. Use ArmPi Pro IK for the real forward grasp motion.

This is intentionally closer to the original ArmPi Pro color-grasp examples
than to a pure joint-trajectory demo.
"""

import argparse
import copy
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rospy
import yaml
from sensor_msgs.msg import CompressedImage, Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

SCRIPT_FILE = Path(__file__).resolve()
DEFAULT_CONFIG_PATH = SCRIPT_FILE.with_name("horizontal_red_grasp_config.yaml")


def _add_python_path(path):
    path = Path(path).expanduser()
    if path.exists():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _add_armpi_factory_paths():
    """Find ArmPi Pro factory modules when this script is copied standalone."""
    workspace_roots = [
        SCRIPT_FILE.parent.parent / "armpi_pro",
        SCRIPT_FILE.parent / "armpi_pro",
        Path.home() / "armpi_pro",
        Path("/home/ubuntu/armpi_pro"),
    ]
    for root in workspace_roots:
        _add_python_path(root / "devel" / "lib" / "python3" / "dist-packages")
        _add_python_path(root / "devel" / "lib" / "python2.7" / "dist-packages")
        _add_python_path(root / "src" / "armpi_pro_common")
        _add_python_path(root / "src" / "armpi_pro_kinematics")


_add_armpi_factory_paths()

try:
    from hiwonder_servo_msgs.msg import MultiRawIdPosDur
    from armpi_pro import bus_servo_control
    from armpi_pro import PID
    from armpi_pro import Misc
    from kinematics import ik_transform

    FACTORY_IK_IMPORT_ERROR = None
except Exception as exc:
    MultiRawIdPosDur = None
    bus_servo_control = None
    PID = None
    Misc = None
    ik_transform = None
    FACTORY_IK_IMPORT_ERROR = exc


ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5"]
GRIPPER_JOINT = "r_joint"

DEFAULT_CONFIG = {
    "vision": {
        "image_topic": "/usb_cam/image_raw",
        "settle_frames": 15,
        "frame_timeout": 5.0,
        "frame_max_age": 0.5,
        "save_debug": True,
        "debug_dir": str(Path.home() / "Desktop" / "grasp_test" / "debug" / "horizontal_red_grasp"),
        "detector": {
            "roi": [0.08, 0.15, 0.92, 0.92],
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
            "min_area": 500,
            "min_rect_fill": 0.40,
            "open_kernel": 3,
            "close_kernel": 5,
        },
        "grasp": {
            "backend": "joint_trajectory",
            "align": {
                "enabled": True,
                "mode": "arm_only",
                "pixel_deadband": 36,
                "center_offset": [0, 45],
                "max_iters": 8,
                "arm_gain_u": 0.0008,
                "arm_gain_v": 0.0008,
                "arm_step_limit": 0.04,
                "arm_limits": {
                    "joint1": [-0.80, 0.80],
                    "joint2": [0.25, 1.10],
                },
            },
            "factory_ik": {
                "enabled": True,
                "use_factory_observe": False,
                "x_dis": 500,
                "y_dis": 0.15,
                "x_center": 320,
                "y_center": 370,
                "x_pid_p": 0.06,
                "y_pid_p": 0.00003,
                "x_limits": [200, 800],
                "y_limits": [0.12, 0.26],
                "stable_err_u": 18,
                "stable_err_v": 20,
                "stable_frames": 4,
                "track_timeout": 10.0,
                "observe_xyz": [0.0, 0.15, 0.03],
                "track_x": 0.03,
                "track_z": 0.10,
                "track_pitch": -115.0,
                "track_pitch_range": [-145.0, -90.0],
                "track_enter_duration_ms": 600,
                "track_enter_settle": 0.25,
                "track_move_duration_ms": 120,
                "track_move_settle": 0.08,
                "grasp_x": 0.0,
                "grasp_z": 0.14,
                "grasp_y_extra": 0.02,
                "grasp_y_comp": 0.0,
                "grasp_z_comp": 0.0,
                "grasp_pitch": -120.0,
                "grasp_pitch_range": [-145.0, -90.0],
                "grasp_pitch_map_in": [-145.0, -95.0],
                "grasp_pitch_map_out": [-0.02, 0.02],
                "open_servo": 120,
                "close_servo": 450,
                "lift_servos": [450, 500, 80, 825, 625, 500],
            },
            "arm": {
                "observe": [0.0, 0.85, -0.85, -1.40, 0.0],
                "home": [0.0, 0.85, -0.85, -1.40, 0.0],
                "pregrasp": [0.0, 0.78, -0.78, -1.25, 0.0],
                "grasp": [0.0, 0.72, -0.70, -1.30, 0.0],
                "retreat": [0.0, 0.92, -0.82, -1.20, 0.0],
                "relative_to_aligned": True,
                "pregrasp_delta": [0.0, -0.10, 0.10, -0.08, 0.0],
                "grasp_delta": [0.0, -0.24, 0.28, -0.20, 0.0],
                "retreat_delta": [0.0, 0.10, 0.02, 0.08, 0.0],
                "joint_limits": {
                    "joint1": [-0.80, 0.80],
                    "joint2": [0.20, 1.10],
                    "joint3": [-1.45, -0.35],
                    "joint4": [-1.80, -0.80],
                    "joint5": [-1.20, 1.20],
                },
            },
            "gripper": {
                "open_step": 0.55,
                "close_step": 0.45,
            },
        },
    }
}


class RedDetection:
    def __init__(self, center_u, center_v, area, bbox, contour, mask, roi, rect_fill):
        self.center_u = center_u
        self.center_v = center_v
        self.area = area
        self.bbox = bbox
        self.contour = contour
        self.mask = mask
        self.roi = roi
        self.rect_fill = rect_fill


class CameraCapture:
    def __init__(self, config=None, image_topic="/usb_cam/image_raw/compressed", camera_yaml=None):
        vision_cfg = (config or {}).get("vision", {})
        self.image_topic = vision_cfg.get("image_topic", image_topic)
        self.fallback_image_topic = vision_cfg.get("fallback_image_topic", "/usb_cam/image_raw/compressed")
        self.settle_frames = int(vision_cfg.get("settle_frames", 5))
        self.timeout = float(vision_cfg.get("frame_timeout", 5.0))
        self.max_frame_age = float(vision_cfg.get("frame_max_age", 0.5))

        cam_path = vision_cfg.get("camera_yaml") or camera_yaml
        self._camera_matrix = None
        self._dist_coeffs = None
        if cam_path:
            self._load_camera_yaml(cam_path)

    def _load_camera_yaml(self, path):
        path = Path(path)
        if not path.exists():
            rospy.logwarn("camera yaml not found: %s", path)
            return
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        self._camera_matrix = np.array(data["camera_matrix"]["data"]).reshape(3, 3)
        self._dist_coeffs = np.array(data["distortion_coefficients"]["data"])

    def _undistort(self, image):
        if self._camera_matrix is None:
            return image
        return cv2.undistort(image, self._camera_matrix, self._dist_coeffs)

    def _decode_compressed(self, msg, topic_name):
        array = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            rospy.logwarn("failed to decode compressed image from %s", topic_name)
            return None
        return self._undistort(image)

    def _decode_raw(self, msg, topic_name):
        data = np.frombuffer(msg.data, dtype=np.uint8)

        # ArmPi Pro stock scripts often treat /usb_cam/image_raw as a plain
        # HxWx3 image buffer regardless of the declared encoding. Keep that
        # permissive path first because it matches the original working code.
        expected_rgb_bytes = int(msg.height) * int(msg.width) * 3
        if data.size == expected_rgb_bytes:
            try:
                image = data.reshape(msg.height, msg.width, 3)
                if msg.encoding == "bgr8":
                    return self._undistort(image.copy())
                else:
                    return self._undistort(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            except Exception:
                pass

        channels = 3
        if msg.encoding in ("mono8", "8UC1"):
            channels = 1
        elif msg.encoding in ("bgr8", "rgb8", "8UC3", "yuyv", "yuyv422", "yuv422"):
            channels = 3
        elif msg.encoding in ("uyvy", "uyvy422"):
            channels = 3
        elif msg.encoding not in ("bgr8", "rgb8", "8UC3"):
            rospy.logwarn("unsupported raw image encoding on %s: %s", topic_name, msg.encoding)
            return None

        try:
            if channels == 1:
                image = data.reshape(msg.height, msg.width)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                if msg.encoding in ("yuyv", "yuyv422", "yuv422"):
                    image = data.reshape(msg.height, msg.width, 2)
                    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)
                elif msg.encoding in ("uyvy", "uyvy422"):
                    image = data.reshape(msg.height, msg.width, 2)
                    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
                else:
                    image = data.reshape(msg.height, msg.width, channels)
                if msg.encoding == "rgb8":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except ValueError:
            rospy.logwarn("failed to reshape raw image from %s", topic_name)
            return None

        return self._undistort(image.copy())

    def _get_frame_from_topic(self, topic_name, timeout):
        if topic_name.endswith("/compressed"):
            msg = rospy.wait_for_message(topic_name, CompressedImage, timeout=timeout)
            return self._decode_compressed(msg, topic_name)
        msg = rospy.wait_for_message(topic_name, Image, timeout=timeout)
        return self._decode_raw(msg, topic_name)

    def wait_for_stream(self, timeout=None):
        timeout = self.timeout if timeout is None else float(timeout)
        try:
            frame = self._get_frame_from_topic(self.image_topic, timeout)
            if frame is not None:
                return True
        except rospy.ROSException:
            pass

        if self.fallback_image_topic and self.fallback_image_topic != self.image_topic:
            rospy.logwarn(
                "primary image topic %s did not yield usable frames, switch to fallback %s",
                self.image_topic,
                self.fallback_image_topic,
            )
            try:
                frame = self._get_frame_from_topic(self.fallback_image_topic, timeout)
                if frame is not None:
                    self.image_topic = self.fallback_image_topic
                    return True
            except rospy.ROSException:
                pass
        return False

    def get_frame(self, discard=0):
        if not self.wait_for_stream():
            return None
        frame = None
        total = max(int(discard), self.settle_frames)
        total = max(1, total)
        for _ in range(total):
            try:
                frame = self._get_frame_from_topic(self.image_topic, self.timeout)
            except rospy.ROSException:
                return None
            if frame is None:
                return None
        return frame.copy() if frame is not None else None

    def save_debug(self, directory, name, image):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        cv2.imwrite(str(path), image)
        rospy.loginfo("saved debug image: %s", path)
        return path


def deep_update(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


class HorizontalRedDetector:
    def __init__(self, config):
        cfg = config["vision"]["detector"]
        hsv = cfg["red_hsv"]
        self.roi_norm = cfg["roi"]
        self.lower1 = np.array(hsv["lower1"], dtype=np.uint8)
        self.upper1 = np.array(hsv["upper1"], dtype=np.uint8)
        self.lower2 = np.array(hsv["lower2"], dtype=np.uint8)
        self.upper2 = np.array(hsv["upper2"], dtype=np.uint8)
        rgb = cfg.get("red_rgb", {})
        self.min_r = int(rgb.get("min_r", 120))
        self.min_r_minus_g = int(rgb.get("min_r_minus_g", 55))
        self.min_r_minus_b = int(rgb.get("min_r_minus_b", 45))
        self.max_g = int(rgb.get("max_g", 170))
        self.min_area = float(cfg["min_area"])
        self.min_rect_fill = float(cfg["min_rect_fill"])
        self.open_kernel = int(cfg["open_kernel"])
        self.close_kernel = int(cfg["close_kernel"])

    def _roi_rect(self, frame):
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

        x1, y1, x2, y2 = self._roi_rect(frame)
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, self.lower1, self.upper1) | cv2.inRange(hsv, self.lower2, self.upper2)

        b, g, r = cv2.split(crop)
        b = b.astype(np.int16)
        g = g.astype(np.int16)
        r = r.astype(np.int16)
        rgb_mask = (
            (r >= self.min_r)
            & ((r - g) >= self.min_r_minus_g)
            & ((r - b) >= self.min_r_minus_b)
            & (g <= self.max_g)
        ).astype(np.uint8) * 255

        mask = cv2.bitwise_and(hsv_mask, rgb_mask)

        if self.open_kernel > 1:
            kernel = np.ones((self.open_kernel, self.open_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        if self.close_kernel > 1:
            kernel = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            rect_area = float(max(w * h, 1))
            rect_fill = area / rect_area
            if rect_fill < self.min_rect_fill:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            center_u = float(moments["m10"] / moments["m00"]) + x1
            center_v = float(moments["m01"] / moments["m00"]) + y1
            score = area * rect_fill
            if score > best_score:
                best_score = score
                best = RedDetection(
                    center_u=center_u,
                    center_v=center_v,
                    area=area,
                    bbox=(x + x1, y + y1, x + x1 + w, y + y1 + h),
                    contour=contour.copy(),
                    mask=mask.copy(),
                    roi=(x1, y1, x2, y2),
                    rect_fill=rect_fill,
                )
        return best

    def draw_debug(self, frame, detection, desired_center=None):
        vis = frame.copy()
        h, w = frame.shape[:2]

        if detection is not None:
            x1, y1, x2, y2 = detection.roi
        else:
            x1, y1, x2, y2 = self._roi_rect(frame)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if desired_center is not None:
            cv2.drawMarker(
                vis,
                (int(desired_center[0]), int(desired_center[1])),
                (255, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )

        if detection is not None:
            bx1, by1, bx2, by2 = detection.bbox
            contour = detection.contour + np.array([[[x1, y1]]], dtype=np.int32)
            cv2.drawContours(vis, [contour], -1, (0, 255, 255), 2)
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            cv2.circle(vis, (int(detection.center_u), int(detection.center_v)), 5, (255, 0, 0), -1)
            cv2.putText(
                vis,
                "area={:.0f} fill={:.2f}".format(detection.area, detection.rect_fill),
                (bx1, max(20, by1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
            if desired_center is not None:
                err_u = detection.center_u - desired_center[0]
                err_v = detection.center_v - desired_center[1]
                cv2.putText(
                    vis,
                    "err_u={:+.1f} err_v={:+.1f}".format(err_u, err_v),
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (50, 220, 50),
                    2,
                )

            mask_bgr = cv2.cvtColor(detection.mask, cv2.COLOR_GRAY2BGR)
            mask_bgr = cv2.resize(mask_bgr, (220, 160))
            vis[10:170, w - 230 : w - 10] = mask_bgr
            cv2.rectangle(vis, (w - 230, 10), (w - 10, 170), (255, 255, 255), 1)
        else:
            cv2.putText(
                vis,
                "no valid red target detected",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )
        return vis


class HorizontalGraspRunner:
    def __init__(self, config):
        self.config = config
        self.debug_dir = Path(config["vision"]["debug_dir"])
        self.camera = CameraCapture(config=config)
        self.detector = HorizontalRedDetector(config)
        self.arm_pub = rospy.Publisher("/arm_controller/command", JointTrajectory, queue_size=1)
        self.gripper_pub = rospy.Publisher("/gripper_controller/command", JointTrajectory, queue_size=1)

        grasp_cfg = config["vision"]["grasp"]
        self.grasp_cfg = grasp_cfg
        self.align_cfg = grasp_cfg["align"]
        self.arm_cfg = grasp_cfg["arm"]
        self.gripper_cfg = grasp_cfg["gripper"]
        self.factory_cfg = grasp_cfg.get("factory_ik", {})
        self.factory_available = FACTORY_IK_IMPORT_ERROR is None
        self.ik = ik_transform.ArmIK() if self.factory_available else None
        self.factory_pub = None
        self.factory_x_pid = None
        self.factory_y_pid = None
        if self.factory_available:
            self.factory_pub = rospy.Publisher(
                "/servo_controllers/port_id_1/multi_id_pos_dur",
                MultiRawIdPosDur,
                queue_size=1,
            )
            if PID is not None:
                self.factory_x_pid = PID.PID(P=float(self.factory_cfg.get("x_pid_p", 0.06)), I=0, D=0)
                self.factory_y_pid = PID.PID(P=float(self.factory_cfg.get("y_pid_p", 0.00003)), I=0, D=0)
        self._save_index = 0

        current = self._wait_joint_state()
        self.current_arm_pose = [current[name] for name in ARM_JOINTS]
        self.current_gripper = float(current[GRIPPER_JOINT])
        rospy.loginfo("current arm pose: %s", self.current_arm_pose)
        rospy.loginfo("current gripper: %.3f", self.current_gripper)
        self.observe_pose = list(self.arm_cfg["observe"])
        self.home_pose = list(self.arm_cfg["home"])
        self.pregrasp_pose = list(self.arm_cfg["pregrasp"])
        self.grasp_pose = list(self.arm_cfg["grasp"])
        self.retreat_pose = list(self.arm_cfg["retreat"])
        self.aligned_arm_pose = copy.copy(self.current_arm_pose)

        rospy.sleep(0.8)

    def _log_stage(self, message, *args):
        stamp = datetime.now().strftime("%H:%M:%S")
        text = message % args if args else message
        rospy.loginfo("[stage %s] %s", stamp, text)

    def _selected_backend(self):
        backend = str(self.grasp_cfg.get("backend", "auto")).lower()
        if backend == "auto":
            rospy.logwarn_once(
                "backend=auto is treated as joint_trajectory for safety; "
                "set backend=factory_ik explicitly to test factory IK"
            )
            return "joint_trajectory"
        if backend == "factory_ik" and not self.factory_available:
            rospy.logwarn_once(
                "backend=factory_ik requested but factory modules are unavailable: %s; "
                "fallback to joint_trajectory. Did you source /home/ubuntu/armpi_pro/devel/setup.bash?",
                FACTORY_IK_IMPORT_ERROR,
            )
            return "joint_trajectory"
        return backend

    def _factory_set_servos(self, duration_ms, servos):
        if not self.factory_available:
            raise RuntimeError("factory IK modules unavailable: {}".format(FACTORY_IK_IMPORT_ERROR))
        bus_servo_control.set_servos(self.factory_pub, int(duration_ms), tuple(servos))

    def _reset_factory_pid(self):
        if self.factory_x_pid is not None:
            self.factory_x_pid.clear()
        if self.factory_y_pid is not None:
            self.factory_y_pid.clear()

    def _factory_ik_target(self, xyz, alpha=None, alpha_range=None):
        if alpha is None:
            alpha = -180
        if alpha_range is None:
            alpha_range = (-180, 0)
        alpha_min = float(alpha_range[0])
        alpha_max = float(alpha_range[1])
        target = self.ik.setPitchRanges(tuple(xyz), float(alpha), alpha_min, alpha_max)
        if not target:
            return None
        return target

    def _factory_move_xyz(self, xyz, x_dis=None, duration_ms=1000, settle=0.5, include_gripper=None, alpha=None, alpha_range=None):
        target = self._factory_ik_target(xyz, alpha=alpha, alpha_range=alpha_range)
        if not target:
            rospy.logwarn("factory IK has no solution for xyz=%s", xyz)
            return None
        servo_data = target[1]
        servos = []
        if include_gripper is not None:
            servos.append((1, int(include_gripper)))
        servos.extend(
            [
                (3, servo_data["servo3"]),
                (4, servo_data["servo4"]),
                (5, servo_data["servo5"]),
            ]
        )
        if x_dis is not None:
            servos.append((6, int(x_dis)))
        else:
            servos.append((6, int(self.factory_cfg.get("x_dis", 500))))
        self._factory_set_servos(duration_ms, servos)
        rospy.sleep(settle)
        return target

    def _map_linear(self, value, in_min, in_max, out_min, out_max):
        if abs(in_max - in_min) < 1e-6:
            return out_min
        if Misc is not None:
            return Misc.map(value, in_min, in_max, out_min, out_max)
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def _factory_move_observe(self):
        cfg = self.factory_cfg
        xyz = cfg.get("observe_xyz", [0.0, 0.15, 0.03])
        x_dis = int(cfg.get("x_dis", 500))
        rospy.loginfo("factory IK observe xyz=%s x_dis=%d", xyz, x_dis)
        target = self._factory_ik_target(xyz)
        if not target:
            rospy.logwarn("factory IK observe failed, fallback to joint trajectory observe")
            return False
        servo_data = target[1]
        self._factory_set_servos(
            1800,
            (
                (1, int(cfg.get("open_servo", 120))),
                (2, 500),
                (3, servo_data["servo3"]),
                (4, servo_data["servo4"]),
                (5, servo_data["servo5"]),
                (6, x_dis),
            ),
        )
        rospy.sleep(2.0)
        return True

    def _wait_joint_state(self, timeout=5.0):
        msg = rospy.wait_for_message("/joint_states", JointState, timeout=timeout)
        state = dict(zip(msg.name, msg.position))
        missing = [name for name in ARM_JOINTS + [GRIPPER_JOINT] if name not in state]
        if missing:
            raise rospy.ROSException("missing joints in /joint_states: {}".format(missing))
        return state

    def _publish_trajectory(self, pub, joint_names, positions, duration):
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = list(joint_names)

        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        point.time_from_start = rospy.Duration(float(duration))
        msg.points.append(point)

        for _ in range(3):
            pub.publish(msg)
            rospy.sleep(0.05)

    def _save_debug(self, frame, tag, detection=None, desired_center=None):
        if not self.config["vision"].get("save_debug", True) or frame is None:
            return
        visual = self.detector.draw_debug(frame, detection, desired_center)
        name = "{:03d}_{}.jpg".format(self._save_index, tag)
        self._save_index += 1
        self.camera.save_debug(self.debug_dir, name, visual)

    def _desired_center(self, frame):
        h, w = frame.shape[:2]
        offset_u, offset_v = self.align_cfg["center_offset"]
        return w / 2.0 + float(offset_u), h / 2.0 + float(offset_v)

    def _arm_move(self, pose, duration=1.0, settle=0.6):
        self._publish_trajectory(self.arm_pub, ARM_JOINTS, pose, duration)
        self.current_arm_pose = list(pose)
        rospy.sleep(settle)
        try:
            state = self._wait_joint_state(timeout=1.0)
            actual = [state[name] for name in ARM_JOINTS]
            rospy.loginfo("actual arm pose after command: %s", actual)
        except rospy.ROSException as exc:
            rospy.logwarn("could not read actual arm pose after command: %s", exc)
        return True

    def _gripper_to(self, position, duration=0.8, settle=0.5):
        self._publish_trajectory(self.gripper_pub, [GRIPPER_JOINT], [position], duration)
        self.current_gripper = float(position)
        rospy.sleep(settle)
        try:
            state = self._wait_joint_state(timeout=1.0)
            actual = float(state[GRIPPER_JOINT])
            self.current_gripper = actual
            self._log_stage("actual gripper after command: %.3f", actual)
        except rospy.ROSException as exc:
            rospy.logwarn("could not read actual gripper after command: %s", exc)
        return True

    def _clamp_pose(self, pose):
        limits = self.arm_cfg.get("joint_limits", {})
        clamped = list(pose)
        for index, joint in enumerate(ARM_JOINTS):
            if joint not in limits:
                continue
            low, high = limits[joint]
            clamped[index] = max(float(low), min(float(high), float(clamped[index])))
        return clamped

    def _offset_pose(self, base_pose, delta):
        return self._clamp_pose([float(value) + float(delta[index]) for index, value in enumerate(base_pose)])

    def _build_grasp_poses(self):
        if not self.arm_cfg.get("relative_to_aligned", True):
            return self.home_pose, self.pregrasp_pose, self.grasp_pose, self.retreat_pose

        base = self._clamp_pose(self.aligned_arm_pose)
        pregrasp = self._offset_pose(base, self.arm_cfg.get("pregrasp_delta", [0, 0, 0, 0, 0]))
        grasp = self._offset_pose(base, self.arm_cfg.get("grasp_delta", [0, 0, 0, 0, 0]))
        retreat = self._offset_pose(base, self.arm_cfg.get("retreat_delta", [0, 0, 0, 0, 0]))
        return base, pregrasp, grasp, retreat

    def move_arm_to_observe_pose(self, dry_run=False):
        if self._selected_backend() == "factory_ik" and self.factory_cfg.get("use_factory_observe", False):
            observe_xyz = self.factory_cfg.get("observe_xyz", [0.0, 0.15, 0.03])
            self._log_stage("move arm to factory IK observe xyz: %s", observe_xyz)
            if dry_run:
                return True
            if self._factory_move_observe():
                return True
        self._log_stage("move arm to observe pose: %s", self.observe_pose)
        if dry_run:
            return True
        return self._arm_move(self.observe_pose, duration=1.2, settle=2.0)

    def startup_open_gripper(self, dry_run=False):
        startup_open_target = self.gripper_cfg.get("startup_open_target", -0.90)
        target = float(startup_open_target)
        target = max(-2.8, min(2.8, target))
        self._log_stage("startup open gripper -> %.3f", target)
        if dry_run:
            self.current_gripper = target
            return True
        return self._gripper_to(target, duration=1.0, settle=0.8)

    def _gripper_open(self, dry_run=False):
        open_target = self.gripper_cfg.get("open_target", -0.90)
        target = float(open_target)
        target = max(-2.8, min(2.8, target))
        self._log_stage("open gripper -> %.3f", target)
        if dry_run:
            return True
        return self._gripper_to(target)

    def _gripper_close(self, dry_run=False):
        close_target = self.gripper_cfg.get("close_target", -1.95)
        target = float(close_target)
        target = max(-2.8, min(2.8, target))
        self._log_stage("close gripper -> %.3f", target)
        if dry_run:
            return True
        return self._gripper_to(target)

    def detect_once(self):
        self._log_stage("capture frame and detect target")
        frame = self.camera.get_frame(discard=1)
        if frame is None:
            rospy.logwarn("no camera frame available")
            return None
        detection = self.detector.detect(frame)
        desired = self._desired_center(frame)
        self._save_debug(frame, "detect" if detection else "detect_fail", detection, desired)
        return detection

    def _align_with_arm_only(self, dry_run=False):
        current = self.detect_once()
        if current is None:
            return None

        max_iters = int(self.align_cfg["max_iters"])
        deadband = float(self.align_cfg["pixel_deadband"])
        gain_u = float(self.align_cfg.get("arm_gain_u", 0.0008))
        gain_v = float(self.align_cfg.get("arm_gain_v", 0.0008))
        step_limit = float(self.align_cfg.get("arm_step_limit", 0.04))
        limits = self.align_cfg.get("arm_limits", {})
        joint1_limits = limits.get("joint1", [-0.80, 0.80])
        joint2_limits = limits.get("joint2", [0.25, 1.10])

        pose = copy.copy(self.current_arm_pose)
        for index in range(max_iters):
            frame = self.camera.get_frame(discard=1)
            if frame is None:
                rospy.logwarn("arm-align %d: frame timeout", index + 1)
                return None

            current = self.detector.detect(frame)
            desired = self._desired_center(frame)
            if current is None:
                self._save_debug(frame, "arm_align_{:02d}_miss".format(index + 1), None, desired)
                rospy.logwarn("arm-align %d: target lost", index + 1)
                return None

            err_u = current.center_u - desired[0]
            err_v = current.center_v - desired[1]
            self._save_debug(frame, "arm_align_{:02d}".format(index + 1), current, desired)

            if abs(err_u) <= deadband and abs(err_v) <= deadband:
                self.aligned_arm_pose = copy.copy(pose)
                rospy.loginfo(
                    "arm-only aligned: err_u=%.1f err_v=%.1f after %d iterations",
                    err_u,
                    err_v,
                    index + 1,
                )
                return current

            delta_joint1 = max(-step_limit, min(step_limit, -err_u * gain_u))
            delta_joint2 = max(-step_limit, min(step_limit, -err_v * gain_v))
            pose[0] = max(float(joint1_limits[0]), min(float(joint1_limits[1]), pose[0] + delta_joint1))
            pose[1] = max(float(joint2_limits[0]), min(float(joint2_limits[1]), pose[1] + delta_joint2))

            rospy.loginfo(
                "arm-align %d: err_u=%.1f err_v=%.1f joint1=%.3f joint2=%.3f",
                index + 1,
                err_u,
                err_v,
                pose[0],
                pose[1],
            )

            if not dry_run:
                self._arm_move(pose, duration=0.8, settle=0.35)
            else:
                self.current_arm_pose = copy.copy(pose)

        rospy.logwarn("arm-only align failed: max iterations reached")
        self.aligned_arm_pose = copy.copy(pose)
        return current

    def align_to_target(self, dry_run=False):
        if not self.align_cfg.get("enabled", True):
            return self.detect_once()

        mode = str(self.align_cfg.get("mode", "arm_only")).lower()
        if mode != "arm_only":
            rospy.logwarn("align mode %s unsupported on ArmPi Pro, fallback to arm_only", mode)
        return self._align_with_arm_only(dry_run=dry_run)

    def execute_horizontal_grasp(self, dry_run=False):
        if self._selected_backend() == "factory_ik":
            return self.execute_factory_ik_grasp(dry_run=dry_run)

        home_pose, pregrasp_pose, grasp_pose, retreat_pose = self._build_grasp_poses()
        self._log_stage(
            "grasp sequence: home=%s pregrasp=%s grasp=%s retreat=%s",
            home_pose,
            pregrasp_pose,
            grasp_pose,
            retreat_pose,
        )

        if not self._gripper_open(dry_run=dry_run):
            return False
        if not dry_run:
            self._log_stage("start move home")
            self._arm_move(home_pose, duration=0.8, settle=0.3)
            self._log_stage("start move pregrasp")
            self._arm_move(pregrasp_pose, duration=1.0, settle=0.5)
            self._log_stage("start move grasp")
            self._arm_move(grasp_pose, duration=1.0, settle=0.5)
        if not self._gripper_close(dry_run=dry_run):
            return False
        if not dry_run:
            self._log_stage("start retreat")
            self._arm_move(retreat_pose, duration=1.0, settle=0.5)
        self._log_stage("grasp sequence finished")
        return True

    def execute_factory_ik_grasp(self, dry_run=False):
        cfg = self.factory_cfg
        if not self.factory_available:
            rospy.logerr("factory IK unavailable: %s", FACTORY_IK_IMPORT_ERROR)
            return False

        self._log_stage(
            "factory IK grasp backend enabled: target image center=(%.0f, %.0f), y_dis range=%s",
            float(cfg.get("x_center", 320)),
            float(cfg.get("y_center", 410)),
            cfg.get("y_limits", [0.12, 0.30]),
        )
        if dry_run:
            self._log_stage("dry-run enabled; skip factory IK tracking/grasp motion")
            return True

        x_dis = float(cfg.get("x_dis", 500))
        y_dis = float(cfg.get("y_dis", 0.15))
        x_center = float(cfg.get("x_center", 320))
        y_center = float(cfg.get("y_center", 410))
        x_low, x_high = cfg.get("x_limits", [200, 800])
        y_low, y_high = cfg.get("y_limits", [0.12, 0.30])
        stable_count = 0
        deadline = time.time() + float(cfg.get("track_timeout", 10.0))
        last_target = None
        last_detection = None
        stable_err_u = float(cfg.get("stable_err_u", 8))
        stable_err_v = float(cfg.get("stable_err_v", 10))
        self._reset_factory_pid()

        self._log_stage("factory IK open gripper servo -> %d", int(cfg.get("open_servo", 120)))
        self._factory_set_servos(500, ((1, int(cfg.get("open_servo", 120))),))
        rospy.sleep(0.5)

        track_x = float(cfg.get("track_x", 0.03))
        track_pitch = float(cfg.get("track_pitch", -115.0))
        track_pitch_range = cfg.get("track_pitch_range", [-145.0, -90.0])
        grasp_pitch = float(cfg.get("grasp_pitch", -125.0))
        grasp_pitch_range = cfg.get("grasp_pitch_range", [-145.0, -95.0])

        while time.time() < deadline and not rospy.is_shutdown():
            frame = self.camera.get_frame(discard=1)
            if frame is None:
                rospy.logwarn("factory IK track: no frame")
                continue
            detection = self.detector.detect(frame)
            desired = (x_center, y_center)
            self._save_debug(frame, "factory_track", detection, desired)
            if detection is None:
                stable_count = 0
                rospy.sleep(0.05)
                continue

            err_x = detection.center_u - x_center
            err_y = detection.center_v - y_center
            last_detection = detection

            if self.factory_x_pid is not None:
                if abs(err_x) < 10:
                    self.factory_x_pid.SetPoint = detection.center_u
                else:
                    self.factory_x_pid.SetPoint = x_center
                self.factory_x_pid.update(detection.center_u)
                dx = self.factory_x_pid.output
            else:
                dx = 0.0 if abs(err_x) < 10 else -err_x * float(cfg.get("x_pid_p", 0.06))

            if self.factory_y_pid is not None:
                if abs(err_y) < 10:
                    self.factory_y_pid.SetPoint = detection.center_v
                else:
                    self.factory_y_pid.SetPoint = y_center
                self.factory_y_pid.update(detection.center_v)
                dy = self.factory_y_pid.output
            else:
                dy = 0.0 if abs(err_y) < 10 else -err_y * float(cfg.get("y_pid_p", 0.00003))

            x_dis = max(float(x_low), min(float(x_high), x_dis + dx))
            y_dis = max(float(y_low), min(float(y_high), y_dis + dy))

            xyz = [track_x, round(y_dis, 4), float(cfg.get("track_z", 0.10))]
            if last_target is None:
                duration_ms = int(cfg.get("track_enter_duration_ms", 600))
                settle = float(cfg.get("track_enter_settle", 0.25))
            else:
                duration_ms = int(cfg.get("track_move_duration_ms", 120))
                settle = float(cfg.get("track_move_settle", 0.08))
            last_target = self._factory_move_xyz(
                xyz,
                x_dis=x_dis,
                duration_ms=duration_ms,
                settle=settle,
                alpha=track_pitch,
                alpha_range=track_pitch_range,
            )
            self._log_stage(
                "factory-track: err_x=%.1f err_y=%.1f x_dis=%d y_dis=%.4f track_xyz=%s track_pitch=%.1f duration_ms=%d",
                err_x,
                err_y,
                int(x_dis),
                y_dis,
                xyz,
                track_pitch,
                duration_ms,
            )

            if abs(err_x) < stable_err_u and abs(err_y) < stable_err_v:
                stable_count += 1
                if stable_count >= int(cfg.get("stable_frames", 10)):
                    break
            else:
                stable_count = 0

        if stable_count < int(cfg.get("stable_frames", 10)):
            rospy.logwarn("factory IK track timeout; continue with last y_dis=%.4f x_dis=%d", y_dis, int(x_dis))

        offset_y = 0.0
        if last_target is not None and len(last_target) > 2:
            pitch = float(last_target[2])
            pitch_min, pitch_max = cfg.get("grasp_pitch_map_in", [-180.0, -150.0])
            out_min, out_max = cfg.get("grasp_pitch_map_out", [-0.04, 0.03])
            offset_y = self._map_linear(pitch, float(pitch_min), float(pitch_max), float(out_min), float(out_max))

        if last_detection is not None:
            self._log_stage(
                "factory-track locked target at image=(%.1f, %.1f), err=(%.1f, %.1f), final y_dis=%.4f",
                last_detection.center_u,
                last_detection.center_v,
                last_detection.center_u - x_center,
                last_detection.center_v - y_center,
                y_dis,
            )

        grasp_y = y_dis + offset_y + float(cfg.get("grasp_y_extra", 0.0)) + float(cfg.get("grasp_y_comp", 0.0))
        grasp_y = round(max(float(y_low), min(float(y_high), grasp_y)), 4)
        grasp_z = round(float(cfg.get("grasp_z", 0.14)) + float(cfg.get("grasp_z_comp", 0.0)), 4)
        # 原厂色块抓取：笛卡尔 x 恒为 0，左右只靠舵机 6 (x_dis)
        grasp_x = float(cfg.get("grasp_x", 0.0))
        track_z = float(cfg.get("track_z", 0.10))
        grasp_candidates = [
            ([grasp_x, grasp_y, grasp_z], grasp_pitch, grasp_pitch_range),
            ([0.0, grasp_y, grasp_z], grasp_pitch, grasp_pitch_range),
            ([0.0, grasp_y, round(track_z - 0.05, 4)], track_pitch, track_pitch_range),
            ([0.0, round(y_dis, 4), round(track_z - 0.03, 4)], track_pitch, track_pitch_range),
            ([0.0, round(y_dis, 4), track_z], track_pitch, track_pitch_range),
        ]

        self._log_stage(
            "factory IK grasp compensation: y_comp=%.4f z_comp=%.4f grasp_pitch=%.1f -> primary_xyz=%s",
            float(cfg.get("grasp_y_comp", 0.0)),
            float(cfg.get("grasp_z_comp", 0.0)),
            grasp_pitch,
            [grasp_x, grasp_y, grasp_z],
        )

        moved = None
        for idx, (candidate_xyz, alpha, alpha_range) in enumerate(grasp_candidates):
            self._log_stage(
                "try factory IK grasp candidate#%d xyz=%s pitch=%.1f range=%s x_dis=%d",
                idx + 1,
                candidate_xyz,
                float(alpha),
                alpha_range,
                int(x_dis),
            )
            moved = self._factory_move_xyz(
                candidate_xyz,
                x_dis=x_dis,
                duration_ms=1000,
                settle=1.5,
                alpha=alpha,
                alpha_range=alpha_range,
            )
            if moved is not None:
                break
        if moved is None:
            rospy.logerr("factory IK grasp failed: no reachable candidate near track pose")
            return False

        self._log_stage("factory IK close gripper servo -> %d", int(cfg.get("close_servo", 450)))
        self._factory_set_servos(500, ((1, int(cfg.get("close_servo", 450))),))
        rospy.sleep(0.8)

        lift = cfg.get("lift_servos", [450, 500, 80, 825, 625, 500])
        self._log_stage("factory IK lift/retreat")
        self._factory_set_servos(
            1500,
            (
                (1, int(lift[0])),
                (2, int(lift[1])),
                (3, int(lift[2])),
                (4, int(lift[3])),
                (5, int(lift[4])),
                (6, int(lift[5])),
            ),
        )
        rospy.sleep(1.5)
        self._log_stage("factory IK grasp finished")
        return True

    def run_once(self, dry_run=False):
        self._log_stage("run start")
        self.startup_open_gripper(dry_run=dry_run)
        self.move_arm_to_observe_pose(dry_run=False)
        self._log_stage("waiting for fresh camera stream...")
        if not self.camera.wait_for_stream():
            rospy.logerr("camera stream not ready")
            return False
        self._log_stage("settling camera after observe pose...")
        rospy.sleep(1.0)

        if self._selected_backend() == "factory_ik":
            self._log_stage("detect target before factory IK grasp")
            target = self.detect_once()
        else:
            self._log_stage("start arm-only alignment")
            target = self.align_to_target(dry_run=dry_run)
        if target is None:
            rospy.logerr("no valid red target detected")
            return False

        frame = self.camera.get_frame(discard=1)
        if frame is not None:
            self._log_stage("save aligned debug frame")
            self._save_debug(frame, "aligned", target, self._desired_center(frame))

        self._log_stage("start grasp execution")
        success = self.execute_horizontal_grasp(dry_run=dry_run)
        verify = self.camera.get_frame(discard=2)
        if verify is not None:
            self._log_stage("save after-grasp debug frame")
            self._save_debug(verify, "after_grasp", self.detector.detect(verify), self._desired_center(verify))

        if success:
            self._log_stage("horizontal red grasp flow completed")
        else:
            rospy.logerr("horizontal red grasp flow failed during execute")
        return success


def load_config(config_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if not config_path:
        return config

    path = Path(config_path)
    if not path.exists():
        rospy.logwarn("config file not found, using defaults: %s", path)
        return config

    with open(path, "r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    return deep_update(config, loaded)


def parse_args():
    parser = argparse.ArgumentParser(description="ArmPi Pro front-facing horizontal red grasp test")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="yaml config path; defaults to horizontal_red_grasp_config.yaml next to this script",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="move to observe pose, then detect/save debug images only; skip alignment motion and grasp",
    )
    parser.add_argument(
        "--check-factory-ik",
        action="store_true",
        help="check whether ArmPi Pro factory IK modules are importable, then exit without moving",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rospy.init_node("horizontal_red_grasp_test", log_level=rospy.INFO)
    if args.check_factory_ik:
        if FACTORY_IK_IMPORT_ERROR is None:
            ik = ik_transform.ArmIK()
            target = ik.setPitchRanges((0, 0.15, 0.03), -180, -180, 0)
            rospy.loginfo("factory IK import ok; test target=%s", bool(target))
            return 0 if target else 2
        rospy.logerr("factory IK import failed: %s", FACTORY_IK_IMPORT_ERROR)
        return 1
    config = load_config(args.config)
    runner = HorizontalGraspRunner(config)
    ok = runner.run_once(dry_run=args.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
