#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reusable AprilTag refinement utilities extracted from the legacy flight script."""

import contextlib
import hashlib
import math
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rospy
import yaml

from tta_m3e_rtsp.msg import flightByVel


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

POST_MOVE_HOVER_SEC = 0.35
RTSP_URL = "rtsp://127.0.0.1:8554/live"
FRESH_FRAME_EACH_STEP = False
FRAME_FLUSH_COUNT = 2
DETECT_MAX_WIDTH = 960
LATEST_FRAME_POLL_SEC = 0.01
LATEST_FRAME_TIMEOUT_SEC = 2.0
DEBUG_IMAGE_DIR = Path(
    os.environ.get(
        "APRILTAG_DEBUG_DIR",
        str(Path.home() / "Pictures" / "apriltag_refine_debug_legacy"),
    )
)
DEBUG_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EDGE_MARGIN_PX = 20

REFINE_TIMEOUT_SEC = 25.0
MAX_REFINE_STEPS = 80
REFINE_SPEED_MPS = 0.06
REFINE_MIN_SPEED_MPS = 0.04
REFINE_MAX_SPEED_MPS = 0.20
REFINE_DURATION_SEC = 0.35
REFINE_MAX_ERROR_PX_X = 240.0
REFINE_MAX_ERROR_PX_Y = 240.0
PIXEL_TOLERANCE = 18
ALIGNED_REQUIRED_FRAMES = 3
ALIGNMENT_OBSERVE_SEC = 0.20
POST_ALIGN_HOVER_SEC = 1.20
SEND_FLY_TIME_AS_MS = True
SKIP_EDGE_CONTROL = True
PRESEARCH_MAX_BACKWARD_M = 0.30
PRESEARCH_BACKWARD_SPEED_MPS = 0.05
PRESEARCH_FRAME_INTERVAL_SEC = 0.15
FRAME_SIGNATURE_WIDTH = 160
YAW_CONTROL_ENABLED = True
YAW_TOLERANCE_DEG = 3.0
YAW_RATE_DPS = 12.0
YAW_MAX_STEP_DEG = 4.0
YAW_GAIN = 0.65
YAW_MAX_CORRECTIONS = 8
YAW_CORRECTION_SIGN = 1.0
TAG_SIZE_M = 0.225
AIR_TO_CAMERA_X = 0.0
AIR_TO_CAMERA_Y = 0.03
POSE_TOLERANCE_M = 0.05
CAMERA_INFO_PATH = PACKAGE_ROOT / "cfg" / "ost.yaml"
FLIGHT_CONFIG_PATH = PACKAGE_ROOT / "cfg" / "flight_2026_cfg.yaml"
APRILTAG_CAMERA_POSE_TOPIC = "/apriltag/camera_pose"
APRILTAG_BODY_POSE_TOPIC = "/apriltag/body_pose"
APRILTAG_CAMERA_FRAME_ID = "camera_optical_frame"
APRILTAG_BODY_FRAME_ID = "body"

LAST_FRAME_TIMESTAMP = None
LAST_FRAME_SIGNATURE = None
APRILTAG_CAMERA_POSE_PUB = None
APRILTAG_BODY_POSE_PUB = None


@contextlib.contextmanager
def suppress_stderr():
    try:
        fd = sys.stderr.fileno()
    except Exception:
        yield
        return

    saved_fd = None
    devnull_fd = None
    try:
        saved_fd = os.dup(fd)
        devnull_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull_fd, fd)
        yield
    finally:
        if saved_fd is not None:
            try:
                os.dup2(saved_fd, fd)
                os.close(saved_fd)
            except Exception:
                pass
        if devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except Exception:
                pass


class RtspFrameSource:
    def __init__(self, url: str = RTSP_URL, fresh_each_step: bool = FRESH_FRAME_EACH_STEP):
        self.url = url
        self.cap = None
        self.fresh_each_step = fresh_each_step
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_frame_time = None
        self._latest_frame_index = 0
        self._reader_thread = None
        self._stop_reader = False

    def _open(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            return
        with suppress_stderr():
            self.cap = cv2.VideoCapture(self.url)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Cannot open RTSP stream: {self.url}")

    def _reader_loop(self) -> None:
        while not self._stop_reader and not rospy.is_shutdown():
            try:
                self._open()
                with suppress_stderr():
                    ok, frame = self.cap.read()
                if ok and frame is not None:
                    now = time.time()
                    with self._lock:
                        self._latest_frame = frame
                        self._latest_frame_time = now
                        self._latest_frame_index += 1
                else:
                    rospy.sleep(0.02)
            except Exception as exc:
                rospy.logwarn("RTSP后台取帧线程异常: %s", exc)
                self.close_capture_only()
                rospy.sleep(0.2)

    def start(self) -> None:
        if self._reader_thread is not None and self._reader_thread.is_alive():
            return
        self._stop_reader = False
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        rospy.loginfo("已启动RTSP后台最新帧读取线程: %s", self.url)

    def read(self):
        if not self.fresh_each_step:
            self.start()
            deadline = time.time() + LATEST_FRAME_TIMEOUT_SEC
            with self._lock:
                last_seen_index = self._latest_frame_index
            while not rospy.is_shutdown() and time.time() < deadline:
                with self._lock:
                    frame = None if self._latest_frame is None else self._latest_frame.copy()
                    frame_time = self._latest_frame_time
                    frame_index = self._latest_frame_index
                if frame is not None and frame_index != last_seen_index:
                    age = time.time() - frame_time if frame_time is not None else -1.0
                    rospy.loginfo(
                        "读取到最新RTSP画面: 帧序号=%d 画面年龄=%.3fs",
                        frame_index,
                        age,
                    )
                    return frame
                rospy.sleep(LATEST_FRAME_POLL_SEC)

            with self._lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
                frame_time = self._latest_frame_time
                frame_index = self._latest_frame_index
            if frame is not None:
                age = time.time() - frame_time if frame_time is not None else -1.0
                rospy.logwarn(
                    "等待更新RTSP画面超时，继续使用当前最新帧: 帧序号=%d 画面年龄=%.3fs",
                    frame_index,
                    age,
                )
                return frame
            raise RuntimeError("Cannot read a valid frame from RTSP background reader")

        if self.fresh_each_step:
            self.close()
        self._open()

        with suppress_stderr():
            for _ in range(FRAME_FLUSH_COUNT):
                self.cap.grab()

        frame = None
        for _ in range(5):
            with suppress_stderr():
                ok, frame = self.cap.read()
            if ok and frame is not None:
                return frame
            rospy.sleep(0.08)

        self.close()
        self._open()
        with suppress_stderr():
            for _ in range(FRAME_FLUSH_COUNT):
                self.cap.grab()
        for _ in range(5):
            with suppress_stderr():
                ok, frame = self.cap.read()
            if ok and frame is not None:
                return frame
            rospy.sleep(0.08)
        raise RuntimeError("Cannot read a valid frame from RTSP stream")

    def close_capture_only(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    def close(self) -> None:
        self._stop_reader = True
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self._reader_thread = None
        self.close_capture_only()


def stop_motion(pub) -> None:
    stop = flightByVel()
    stop.vel_n = 0.0
    stop.vel_e = 0.0
    stop.vel_d = 0.0
    stop.targetYaw = 0.0
    stop.fly_time = 0.0
    pub.publish(stop)


def publish_velocity(pub, vel_n: float, vel_e: float, vel_d: float, target_yaw: float, duration: float) -> None:
    msg = flightByVel()
    msg.vel_n = float(vel_n)
    msg.vel_e = float(vel_e)
    msg.vel_d = float(vel_d)
    msg.targetYaw = float(target_yaw)
    fly_time_to_send = float(duration * 1000.0) if SEND_FLY_TIME_AS_MS else float(duration)
    msg.fly_time = fly_time_to_send

    rospy.loginfo(
        "发布速度vel_n=%.3f vel_e=%.3f vel_d=%.3f yaw=%.3f duration_sec=%.3f fly_time_sent=%.1f unit=%s",
        msg.vel_n,
        msg.vel_e,
        msg.vel_d,
        msg.targetYaw,
        duration,
        msg.fly_time,
        "ms" if SEND_FLY_TIME_AS_MS else "sec",
    )

    rate = rospy.Rate(10)
    end_time = rospy.Time.now() + rospy.Duration(duration)
    while not rospy.is_shutdown() and rospy.Time.now() < end_time:
        pub.publish(msg)
        rate.sleep()

    stop_motion(pub)
    rospy.sleep(POST_MOVE_HOVER_SEC)


def reset_debug_image_dir(debug_dir: Path = DEBUG_IMAGE_DIR) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for path in debug_dir.iterdir():
        if path.is_file() and path.suffix.lower() in DEBUG_IMAGE_EXTENSIONS:
            path.unlink()


def classify_tag_bounds(frame, corners, margin_px: int = EDGE_MARGIN_PX) -> str:
    height, width = frame.shape[:2]
    xs = corners[:, 0]
    ys = corners[:, 1]
    if xs.min() < 0 or ys.min() < 0 or xs.max() >= width or ys.max() >= height:
        return "out_of_frame"
    if xs.min() < margin_px or ys.min() < margin_px:
        return "near_edge"
    if xs.max() > width - margin_px or ys.max() > height - margin_px:
        return "near_edge"
    return "in_frame"


def _wrap_angle_deg(angle_deg: float) -> float:
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg <= -180.0:
        angle_deg += 360.0
    return angle_deg


def _angle_to_nearest_tag_axis_deg(angle_deg: float) -> float:
    # A square tag provides four edges; each edge should be close to either the
    # image horizontal or vertical axis when the aircraft yaw is aligned.
    return ((angle_deg + 45.0) % 90.0) - 45.0


def estimate_tag_yaw_error_deg(corners) -> float:
    edge_errors = []
    for idx in range(4):
        p0 = corners[idx]
        p1 = corners[(idx + 1) % 4]
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        angle_deg = math.degrees(math.atan2(dy, dx))
        edge_errors.append(_angle_to_nearest_tag_axis_deg(angle_deg))

    if not edge_errors:
        return 0.0

    return _wrap_angle_deg(sum(edge_errors) / float(len(edge_errors)))


def resolve_yaw_command(yaw_error_deg: float,
                        yaw_tolerance_deg: float,
                        yaw_rate_dps: float,
                        yaw_max_step_deg: float,
                        yaw_gain: float,
                        yaw_correction_sign: float) -> tuple:
    if abs(yaw_error_deg) <= yaw_tolerance_deg:
        return 0.0, 0.0, 0.0

    target_step_deg = min(abs(yaw_error_deg) * max(yaw_gain, 0.0), max(yaw_max_step_deg, 0.0))
    if target_step_deg <= 1e-6:
        return 0.0, 0.0, 0.0

    yaw_rate_abs = max(abs(yaw_rate_dps), 1e-6)
    target_yaw = math.copysign(yaw_rate_abs, yaw_error_deg * yaw_correction_sign)
    duration = max(target_step_deg / yaw_rate_abs, 0.08)
    return target_yaw, duration, target_step_deg


def save_debug_image(step: int,
                     frame,
                     corners,
                     tag_id,
                     frame_timestamp_text: str,
                     frame_delta_sec,
                     frame_signature: str,
                     frame_changed: bool,
                     frame_mean: float,
                     frame_std: float,
                     center_x: float,
                     center_y: float,
                     image_center_x: float,
                     image_center_y: float,
                     error_x: float,
                     error_y: float,
                     distance_px: float,
                     vel_n: float,
                     vel_e: float,
                     aligned: bool,
                     bounds_status: str,
                     yaw_error_deg=None,
                     target_yaw: float = 0.0,
                     yaw_correction_count: int = 0,
                     yaw_max_corrections: int = 0,
                     debug_dir: Path = DEBUG_IMAGE_DIR) -> None:
    annotated = frame.copy()
    points = corners.astype(int)

    for index, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        cv2.circle(annotated, (x, y), 7, (0, 255, 255), -1)
        cv2.putText(
            annotated,
            f"p{index}=({x},{y})",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

    cv2.polylines(annotated, [points.reshape((-1, 1, 2))], True, (0, 255, 0), 2)

    tag_center = (int(round(center_x)), int(round(center_y)))
    image_center = (int(round(image_center_x)), int(round(image_center_y)))
    cv2.circle(annotated, tag_center, 9, (0, 0, 255), -1)
    cv2.circle(annotated, image_center, 9, (255, 0, 0), -1)
    cv2.line(annotated, image_center, tag_center, (255, 255, 255), 2)

    status = "aligned" if aligned else "moving"
    lines = [
        f"round={step:03d} tag_id={tag_id} status={status} bounds={bounds_status}\n",
        f"frame_time={frame_timestamp_text} frame_dt={frame_delta_sec if frame_delta_sec is not None else 'n/a'}s \n",
        f"frame_sig={frame_signature} changed={frame_changed} mean={frame_mean:.1f} std={frame_std:.1f}\n",
        f"tag_center=({center_x:.1f},{center_y:.1f}) image_center=({image_center_x:.1f},{image_center_y:.1f})\n",
        f"dx={error_x:.1f}px dy={error_y:.1f}px dist={distance_px:.1f}px\n",
        f"cmd vel_n={vel_n:.3f} vel_e={vel_e:.3f} dur={REFINE_DURATION_SEC:.2f}s send={REFINE_DURATION_SEC * 1000.0 if SEND_FLY_TIME_AS_MS else REFINE_DURATION_SEC:.1f}{'ms' if SEND_FLY_TIME_AS_MS else 's'}\n",
        f"yaw_error={yaw_error_deg if yaw_error_deg is not None else 'n/a'}deg cmd_yaw={target_yaw:.2f} yaw_fix={yaw_correction_count}/{yaw_max_corrections}\n",
    ]
    for line_index, line in enumerate(lines):
        y = 32 + line_index * 28
        cv2.putText(annotated, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4)
        cv2.putText(annotated, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    debug_dir.mkdir(parents=True, exist_ok=True)
    output_path = debug_dir / (
        f"round_{step:03d}_{bounds_status}_{status}_dx_{error_x:+.0f}_dy_{error_y:+.0f}.jpg"
    )
    cv2.imwrite(str(output_path), annotated)


def save_no_tag_debug_image(step: int, frame, debug_dir: Path = DEBUG_IMAGE_DIR) -> None:
    global LAST_FRAME_TIMESTAMP, LAST_FRAME_SIGNATURE
    annotated = frame.copy()
    height, width = frame.shape[:2]
    image_center = (width // 2, height // 2)
    cv2.circle(annotated, image_center, 9, (255, 0, 0), -1)
    frame_timestamp = time.time()
    frame_timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(frame_timestamp)) + f".{int((frame_timestamp % 1) * 1000):03d}"
    frame_delta_sec = None if LAST_FRAME_TIMESTAMP is None else frame_timestamp - LAST_FRAME_TIMESTAMP
    LAST_FRAME_TIMESTAMP = frame_timestamp
    frame_signature, frame_mean, frame_std = compute_frame_signature(frame)
    frame_changed = LAST_FRAME_SIGNATURE is None or frame_signature != LAST_FRAME_SIGNATURE
    LAST_FRAME_SIGNATURE = frame_signature
    lines = [
        f"round={step:03d} tag_id=None status=no_tag bounds=not_detected\n",
        f"frame_time={frame_timestamp_text} frame_dt={frame_delta_sec if frame_delta_sec is not None else 'n/a'}s \n",
        f"frame_sig={frame_signature} changed={frame_changed} mean={frame_mean:.1f} std={frame_std:.1f}\n",
        f"image_center=({image_center[0]},{image_center[1]})\n",
        f"cmd vel_n=0.000 vel_e=0.000 dur={REFINE_DURATION_SEC:.2f}s send={REFINE_DURATION_SEC * 1000.0 if SEND_FLY_TIME_AS_MS else REFINE_DURATION_SEC:.1f}{'ms' if SEND_FLY_TIME_AS_MS else 's'}\n",
    ]
    for line_index, line in enumerate(lines):
        y = 32 + line_index * 28
        cv2.putText(annotated, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4)
        cv2.putText(annotated, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    debug_dir.mkdir(parents=True, exist_ok=True)
    output_path = debug_dir / f"round_{step:03d}_not_detected_no_tag.jpg"
    cv2.imwrite(str(output_path), annotated)


def compute_frame_signature(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = FRAME_SIGNATURE_WIDTH / float(gray.shape[1])
    small = cv2.resize(
        gray,
        (FRAME_SIGNATURE_WIDTH, int(gray.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    digest = hashlib.md5(small.tobytes()).hexdigest()[:12]
    return digest, float(small.mean()), float(small.std())


def make_detector():
    try:
        import apriltag
    except ImportError:
        rospy.logwarn("未安装apriltag模块，跳过AprilTag检测")
        return None
    return apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))


def detect_complete_apriltag(frame, detector):
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = 1.0
    gray = gray_full
    if gray_full.shape[1] > DETECT_MAX_WIDTH:
        scale = DETECT_MAX_WIDTH / float(gray_full.shape[1])
        gray = cv2.resize(
            gray_full,
            (DETECT_MAX_WIDTH, int(gray_full.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    results = detector.detect(gray)
    if not results:
        return False, "not_detected", None

    tag = results[0]
    corners = tag.corners / scale
    bounds_status = classify_tag_bounds(frame, corners)
    return bounds_status == "in_frame", bounds_status, getattr(tag, "tag_id", None)


def publish_velocity_sample(pub, vel_n: float, vel_e: float, vel_d: float, target_yaw: float, duration: float) -> None:
    msg = flightByVel()
    msg.vel_n = float(vel_n)
    msg.vel_e = float(vel_e)
    msg.vel_d = float(vel_d)
    msg.targetYaw = float(target_yaw)
    msg.fly_time = float(duration * 1000.0) if SEND_FLY_TIME_AS_MS else float(duration)
    pub.publish(msg)


def search_complete_apriltag_by_backward_motion(pub,
                                                frame_source: RtspFrameSource,
                                                max_backward_m: float = PRESEARCH_MAX_BACKWARD_M,
                                                backward_speed_mps: float = PRESEARCH_BACKWARD_SPEED_MPS,
                                                frame_interval_sec: float = PRESEARCH_FRAME_INTERVAL_SEC,
                                                stage_label: str = "apriltag_presearch") -> bool:
    detector = make_detector()
    if detector is None:
        return False
    if frame_source is None:
        frame_source = RtspFrameSource()

    max_backward_m = max(0.0, float(max_backward_m))
    backward_speed_mps = max(abs(float(backward_speed_mps)), 1e-6)
    frame_interval_sec = max(float(frame_interval_sec), 0.05)
    max_duration_sec = max_backward_m / backward_speed_mps if max_backward_m > 0.0 else 0.0

    rospy.loginfo(
        "[APRILTAG][%s] PRESEARCH_START max_backward=%.2fm speed=%.3fm/s interval=%.2fs max_duration=%.2fs",
        stage_label,
        max_backward_m,
        backward_speed_mps,
        frame_interval_sec,
        max_duration_sec,
    )

    try:
        frame = frame_source.read()
        complete, bounds_status, tag_id = detect_complete_apriltag(frame, detector)
    except Exception as exc:
        complete, bounds_status, tag_id = False, "read_failed", None
        rospy.logwarn("[APRILTAG][%s] PRESEARCH_INITIAL_READ_FAILED error=%s", stage_label, exc)

    if complete:
        rospy.loginfo(
            "[APRILTAG][%s] PRESEARCH_SKIP reason=already_complete tag_id=%s bounds=%s",
            stage_label,
            tag_id,
            bounds_status,
        )
        return True

    if max_duration_sec <= 0.0:
        rospy.logwarn(
            "[APRILTAG][%s] PRESEARCH_DISABLED_OR_ZERO_DISTANCE initial_bounds=%s; continue stage-1",
            stage_label,
            bounds_status,
        )
        return False

    start_time = time.time()
    moved_m = 0.0
    backward_vel_n = -backward_speed_mps
    last_bounds_status = bounds_status
    last_tag_id = tag_id

    while not rospy.is_shutdown():
        elapsed = time.time() - start_time
        if elapsed >= max_duration_sec:
            break

        publish_velocity_sample(pub, backward_vel_n, 0.0, 0.0, 0.0, frame_interval_sec)
        rospy.sleep(frame_interval_sec)
        moved_m = min(max_backward_m, (time.time() - start_time) * backward_speed_mps)

        try:
            frame = frame_source.read()
            complete, bounds_status, tag_id = detect_complete_apriltag(frame, detector)
            last_bounds_status = bounds_status
            last_tag_id = tag_id
        except Exception as exc:
            rospy.logwarn("[APRILTAG][%s] PRESEARCH_READ_FAILED moved=%.2fm error=%s", stage_label, moved_m, exc)
            continue

        rospy.loginfo(
            "[APRILTAG][%s] PRESEARCH_CHECK moved=%.2fm tag_id=%s bounds=%s complete=%s",
            stage_label,
            moved_m,
            tag_id,
            bounds_status,
            complete,
        )
        if complete:
            stop_motion(pub)
            rospy.loginfo(
                "[APRILTAG][%s] PRESEARCH_FOUND moved=%.2fm tag_id=%s; enter normal refine stages",
                stage_label,
                moved_m,
                tag_id,
            )
            return True

    stop_motion(pub)
    rospy.logwarn(
        "[APRILTAG][%s] PRESEARCH_NOT_FOUND moved=%.2fm last_tag_id=%s last_bounds=%s; continue stage-1",
        stage_label,
        moved_m,
        last_tag_id,
        last_bounds_status,
    )
    return False


def discrete_step(value: float) -> float:
    abs_value = abs(value)
    print(f"计算出距离差: {value}, 距离绝对值: {abs_value}")
    if abs_value >= 0.8:
        step = 1.0
    elif abs_value >= 0.4:
        step = 0.5
    elif abs_value >= 0.2:
        step = 0.2
    elif abs_value >= 0.1:
        step = 0.1
    else:
        step = 0.05
    return step if value >= 0 else -step


def resolve_axis_velocity(error_px: float,
                          pixel_tolerance: float,
                          max_abs_error_px: float,
                          max_speed_mps: float,
                          min_speed_mps: float,
                          positive_when_error_positive: bool) -> float:
    if abs(error_px) < pixel_tolerance:
        return 0.0

    effective_error = max(0.0, abs(error_px) - pixel_tolerance)
    effective_range = max(1.0, max_abs_error_px - pixel_tolerance)
    scale = max(0.0, min(1.0, effective_error / effective_range))
    commanded_speed = min_speed_mps + scale * max(0.0, max_speed_mps - min_speed_mps)

    if positive_when_error_positive:
        return commanded_speed if error_px > 0 else -commanded_speed
    return commanded_speed if error_px < 0 else -commanded_speed


def load_camera_params():
    if CAMERA_INFO_PATH.exists():
        with CAMERA_INFO_PATH.open("r", encoding="utf-8") as handle:
            camera_info = yaml.safe_load(handle)
        fx = camera_info["camera_matrix"]["data"][0]
        fy = camera_info["camera_matrix"]["data"][4]
        cx = camera_info["camera_matrix"]["data"][2]
        cy = camera_info["camera_matrix"]["data"][5]
        return fx, fy, cx, cy

    rospy.logwarn("未找到相机标定文件，回退到内置标定参数")
    return 1029.6, 1029.5, 723.7489, 525.0792


def load_camera_intrinsics():
    fx, fy, cx, cy = load_camera_params()
    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    if CAMERA_INFO_PATH.exists():
        try:
            with CAMERA_INFO_PATH.open("r", encoding="utf-8") as handle:
                camera_info = yaml.safe_load(handle) or {}
            dist_data = camera_info.get("distortion_coefficients", {}).get("data", [])
            if dist_data:
                dist_coeffs = np.array(dist_data, dtype=np.float64).reshape(-1, 1)
        except Exception as exc:
            rospy.logwarn("[APRILTAG_POSE] failed to read distortion coeffs: %s", exc)
    return camera_matrix, dist_coeffs


def _read_landing_config():
    if not FLIGHT_CONFIG_PATH.exists():
        return {}
    try:
        with FLIGHT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return config.get("apriltag_landing", config.get("landing", {})) or {}
    except Exception as exc:
        rospy.logwarn("[APRILTAG_POSE] failed to read config %s: %s", FLIGHT_CONFIG_PATH, exc)
        return {}


def load_apriltag_refine_config():
    landing_cfg = _read_landing_config()
    return {
        "min_speed_mps": float(landing_cfg.get("apriltag_refine_min_speed_mps", REFINE_MIN_SPEED_MPS)),
        "max_speed_mps": float(landing_cfg.get("apriltag_refine_max_speed_mps", REFINE_MAX_SPEED_MPS)),
        "duration_sec": float(landing_cfg.get("apriltag_refine_duration_sec", REFINE_DURATION_SEC)),
        "max_error_px_x": float(landing_cfg.get("apriltag_refine_max_error_px_x", REFINE_MAX_ERROR_PX_X)),
        "max_error_px_y": float(landing_cfg.get("apriltag_refine_max_error_px_y", REFINE_MAX_ERROR_PX_Y)),
    }


def _as_float_list(value, expected_len, fallback):
    if value is None:
        return list(fallback)
    try:
        values = [float(item) for item in value]
    except Exception:
        return list(fallback)
    if len(values) != expected_len:
        return list(fallback)
    return values


def load_apriltag_pose_config():
    landing_cfg = _read_landing_config()
    return {
        "enabled": bool(landing_cfg.get("apriltag_pose_publish_enabled", True)),
        "tag_size_m": float(landing_cfg.get("apriltag_tag_size_m", TAG_SIZE_M)),
        "camera_topic": str(landing_cfg.get("apriltag_camera_pose_topic", APRILTAG_CAMERA_POSE_TOPIC)),
        "body_topic": str(landing_cfg.get("apriltag_body_pose_topic", APRILTAG_BODY_POSE_TOPIC)),
        "camera_frame_id": str(landing_cfg.get("apriltag_camera_frame_id", APRILTAG_CAMERA_FRAME_ID)),
        "body_frame_id": str(landing_cfg.get("apriltag_body_frame_id", APRILTAG_BODY_FRAME_ID)),
        "camera_to_body_translation": _as_float_list(
            landing_cfg.get("apriltag_camera_to_body_translation_m"),
            3,
            [AIR_TO_CAMERA_X, AIR_TO_CAMERA_Y, 0.0],
        ),
        "camera_to_body_euler_deg": _as_float_list(
            landing_cfg.get("apriltag_camera_to_body_euler_deg"),
            3,
            [0.0, 0.0, 0.0],
        ),
    }


def euler_deg_to_rotation_matrix(euler_deg):
    roll, pitch, yaw = [math.radians(float(value)) for value in euler_deg]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    rotation_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rotation_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rotation_z @ rotation_y @ rotation_x


def rotation_matrix_to_quaternion(rotation):
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0, 1] + rotation[1, 0]) / scale
        qz = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / scale
        qx = (rotation[0, 1] + rotation[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / scale
        qx = (rotation[0, 2] + rotation[2, 0]) / scale
        qy = (rotation[1, 2] + rotation[2, 1]) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-9:
        return 0.0, 0.0, 0.0, 1.0
    return qx / norm, qy / norm, qz / norm, qw / norm


def _get_apriltag_pose_publishers(camera_topic, body_topic):
    global APRILTAG_CAMERA_POSE_PUB, APRILTAG_BODY_POSE_PUB
    from geometry_msgs.msg import PoseStamped

    if APRILTAG_CAMERA_POSE_PUB is None:
        APRILTAG_CAMERA_POSE_PUB = rospy.Publisher(camera_topic, PoseStamped, queue_size=5)
    if APRILTAG_BODY_POSE_PUB is None:
        APRILTAG_BODY_POSE_PUB = rospy.Publisher(body_topic, PoseStamped, queue_size=5)
    return PoseStamped, APRILTAG_CAMERA_POSE_PUB, APRILTAG_BODY_POSE_PUB


def _fill_pose_msg(PoseStamped, frame_id, stamp, translation, rotation):
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(translation[0])
    msg.pose.position.y = float(translation[1])
    msg.pose.position.z = float(translation[2])
    qx, qy, qz, qw = rotation_matrix_to_quaternion(rotation)
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw
    return msg


def publish_apriltag_pose_from_corners(corners, camera_matrix, dist_coeffs, pose_cfg, tag_id=None, stage_label="apriltag_refine"):
    if not pose_cfg.get("enabled", True):
        return None

    tag_size_m = max(float(pose_cfg.get("tag_size_m", TAG_SIZE_M)), 1e-6)
    half_size = tag_size_m / 2.0
    object_points = np.array(
        [
            [-half_size, -half_size, 0.0],
            [half_size, -half_size, 0.0],
            [half_size, half_size, 0.0],
            [-half_size, half_size, 0.0],
        ],
        dtype=np.float64,
    )
    image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        rospy.logwarn("[APRILTAG_POSE][%s] solvePnP failed tag_id=%s", stage_label, tag_id)
        return None

    camera_rotation, _ = cv2.Rodrigues(rvec)
    camera_translation = tvec.reshape(3)
    body_from_camera_rotation = euler_deg_to_rotation_matrix(pose_cfg.get("camera_to_body_euler_deg", [0.0, 0.0, 0.0]))
    body_from_camera_translation = np.array(pose_cfg.get("camera_to_body_translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    body_translation = body_from_camera_rotation @ camera_translation + body_from_camera_translation
    body_rotation = body_from_camera_rotation @ camera_rotation

    try:
        PoseStamped, camera_pub, body_pub = _get_apriltag_pose_publishers(
            pose_cfg.get("camera_topic", APRILTAG_CAMERA_POSE_TOPIC),
            pose_cfg.get("body_topic", APRILTAG_BODY_POSE_TOPIC),
        )
        stamp = rospy.Time.now()
        camera_pub.publish(
            _fill_pose_msg(
                PoseStamped,
                pose_cfg.get("camera_frame_id", APRILTAG_CAMERA_FRAME_ID),
                stamp,
                camera_translation,
                camera_rotation,
            )
        )
        body_pub.publish(
            _fill_pose_msg(
                PoseStamped,
                pose_cfg.get("body_frame_id", APRILTAG_BODY_FRAME_ID),
                stamp,
                body_translation,
                body_rotation,
            )
        )
    except Exception as exc:
        rospy.logwarn("[APRILTAG_POSE][%s] publish failed tag_id=%s error=%s", stage_label, tag_id, exc)
        return None

    rospy.loginfo(
        "[APRILTAG_POSE][%s] tag_id=%s camera_xyz=(%.3f, %.3f, %.3f)m body_xyz=(%.3f, %.3f, %.3f)m",
        stage_label,
        tag_id,
        camera_translation[0],
        camera_translation[1],
        camera_translation[2],
        body_translation[0],
        body_translation[1],
        body_translation[2],
    )
    return camera_translation, body_translation


def refine_with_apriltag(pub,
                         timeout_sec: float = REFINE_TIMEOUT_SEC,
                         frame_source: RtspFrameSource = None,
                         pixel_tolerance: float = PIXEL_TOLERANCE,
                         stage_label: str = "apriltag_refine",
                         aligned_required_frames: int = ALIGNED_REQUIRED_FRAMES,
                         alignment_observe_sec: float = ALIGNMENT_OBSERVE_SEC,
                         post_align_hover_sec: float = POST_ALIGN_HOVER_SEC,
                         descent_vel_d: float = 0.0,
                         exit_on_aligned: bool = True,
                         descend_without_tag: bool = False,
                         yaw_control_enabled: bool = YAW_CONTROL_ENABLED,
                         yaw_tolerance_deg: float = YAW_TOLERANCE_DEG,
                         yaw_rate_dps: float = YAW_RATE_DPS,
                         yaw_max_step_deg: float = YAW_MAX_STEP_DEG,
                         yaw_gain: float = YAW_GAIN,
                         yaw_max_corrections: int = YAW_MAX_CORRECTIONS,
                         yaw_correction_sign: float = YAW_CORRECTION_SIGN) -> bool:
    global LAST_FRAME_TIMESTAMP, LAST_FRAME_SIGNATURE
    try:
        import apriltag
    except ImportError:
        rospy.logwarn("未安装apriltag模块，跳过AprilTag微调")
        return False

    if frame_source is None:
        frame_source = RtspFrameSource()

    # Keep the constructor compatible with the apriltag package version on OK3588.
    # The 2025 code used only the family option; some installed apriltag builds
    # reject newer options such as quad_sigma/refine_edges.
    detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
    camera_matrix, dist_coeffs = load_camera_intrinsics()
    apriltag_pose_cfg = load_apriltag_pose_config()
    apriltag_refine_cfg = load_apriltag_refine_config()
    start_time = time.time()
    LAST_FRAME_TIMESTAMP = None
    LAST_FRAME_SIGNATURE = None
    reset_debug_image_dir()
    no_tag_count = 0
    skip_control_count = 0
    move_command_count = 0
    yaw_correction_count = 0
    aligned_count = 0
    ever_aligned = False
    refine_min_speed_mps = max(0.0, float(apriltag_refine_cfg["min_speed_mps"]))
    refine_max_speed_mps = max(float(apriltag_refine_cfg["max_speed_mps"]), refine_min_speed_mps)
    refine_duration_sec = max(0.05, float(apriltag_refine_cfg["duration_sec"]))
    refine_max_error_px_x = max(1.0, float(apriltag_refine_cfg["max_error_px_x"]))
    refine_max_error_px_y = max(1.0, float(apriltag_refine_cfg["max_error_px_y"]))
    rospy.loginfo("AprilTag调试图片保存目录: %s", DEBUG_IMAGE_DIR)
    rospy.loginfo(
        "AprilTag微调参数: 速度=%.3fm/s 单次持续=%.2fs fly_time单位=%s 边缘保护=%s",
        REFINE_SPEED_MPS,
        REFINE_DURATION_SEC,
        "ms" if SEND_FLY_TIME_AS_MS else "sec",
        SKIP_EDGE_CONTROL,
    )
    rospy.loginfo(
        "【%s】AprilTag微调启动: 像素容差=%.1f, 超时=%.1fs, 最大轮数=%d",
        stage_label,
        pixel_tolerance,
        timeout_sec,
        MAX_REFINE_STEPS,
    )
    rospy.loginfo(
        "【%s】控制逻辑说明: 当前实际控制只使用 dx/dy 和像素容差；discrete_step 仅做调试打印，不参与速度控制",
        stage_label,
    )
    rospy.loginfo(
        "[APRILTAG][%s] START tolerance_px=%.1f timeout_sec=%.1f max_steps=%d speed=%.3f duration=%.2f edge_guard=%s",
        stage_label,
        pixel_tolerance,
        timeout_sec,
        MAX_REFINE_STEPS,
        REFINE_SPEED_MPS,
        REFINE_DURATION_SEC,
        SKIP_EDGE_CONTROL,
    )
    rospy.loginfo(
        "[APRILTAG][%s] LINEAR_XY_SPEED min=%.3fm/s max=%.3fm/s duration=%.2fs max_error_px_x=%.1f max_error_px_y=%.1f",
        stage_label,
        refine_min_speed_mps,
        refine_max_speed_mps,
        refine_duration_sec,
        refine_max_error_px_x,
        refine_max_error_px_y,
    )
    rospy.loginfo("[APRILTAG][%s] INFO control=image_center_dxdy discrete_step=debug_only", stage_label)
    rospy.loginfo(
        "[APRILTAG][%s] ALIGN_POLICY required_frames=%d observe_sec=%.2f post_align_hover_sec=%.2f",
        stage_label,
        aligned_required_frames,
        alignment_observe_sec,
        post_align_hover_sec,
    )
    rospy.loginfo(
        "[APRILTAG][%s] DESCENT_POLICY vel_d=%.3f exit_on_aligned=%s descend_without_tag=%s",
        stage_label,
        descent_vel_d,
        exit_on_aligned,
        descend_without_tag,
    )
    rospy.loginfo(
        "[APRILTAG][%s] YAW_POLICY enabled=%s tol=%.2fdeg rate=%.2fdps max_step=%.2fdeg gain=%.2f max_corrections=%d sign=%.1f",
        stage_label,
        yaw_control_enabled,
        yaw_tolerance_deg,
        yaw_rate_dps,
        yaw_max_step_deg,
        yaw_gain,
        yaw_max_corrections,
        yaw_correction_sign,
    )

    for step in range(MAX_REFINE_STEPS):
        elapsed = time.time() - start_time
        if elapsed >= timeout_sec:
            if not exit_on_aligned and ever_aligned:
                stop_motion(pub)
                rospy.loginfo(
                    "[APRILTAG][%s] EXIT reason=timed_final_descent_complete elapsed=%.1f no_tag=%d skip=%d move=%d yaw_fix=%d",
                    stage_label,
                    elapsed,
                    no_tag_count,
                    skip_control_count,
                    move_command_count,
                    yaw_correction_count,
                )
                return True
            rospy.logwarn(
                "[APRILTAG][%s] EXIT reason=timeout elapsed=%.1f no_tag=%d skip=%d move=%d yaw_fix=%d",
                stage_label,
                elapsed,
                no_tag_count,
                skip_control_count,
                move_command_count,
                yaw_correction_count,
            )
            rospy.logwarn(
                "【%s】超时退出: 已耗时=%.1fs, 未检出次数=%d, 跳过控制次数=%d, 实际发控制次数=%d",
                stage_label,
                elapsed,
                no_tag_count,
                skip_control_count,
                move_command_count,
            )
            rospy.logwarn("AprilTag微调超时: 已执行%.1f秒", elapsed)
            return False

        frame = frame_source.read()
        frame_timestamp = time.time()
        frame_timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(frame_timestamp)) + f".{int((frame_timestamp % 1) * 1000):03d}"
        frame_delta_sec = None if LAST_FRAME_TIMESTAMP is None else frame_timestamp - LAST_FRAME_TIMESTAMP
        LAST_FRAME_TIMESTAMP = frame_timestamp
        frame_signature, frame_mean, frame_std = compute_frame_signature(frame)
        frame_changed = LAST_FRAME_SIGNATURE is None or frame_signature != LAST_FRAME_SIGNATURE
        LAST_FRAME_SIGNATURE = frame_signature
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        scale = 1.0
        gray = gray_full
        if gray_full.shape[1] > DETECT_MAX_WIDTH:
            scale = DETECT_MAX_WIDTH / float(gray_full.shape[1])
            gray = cv2.resize(
                gray_full,
                (DETECT_MAX_WIDTH, int(gray_full.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        results = detector.detect(gray)

        if not results:
            no_tag_count += 1
            rospy.logwarn(
                "[APRILTAG][%s] NO_TAG step=%d elapsed=%.1f frame_sig=%s frame_changed=%s mean=%.1f std=%.1f",
                stage_label,
                step + 1,
                elapsed,
                frame_signature,
                frame_changed,
                frame_mean,
                frame_std,
            )
            rospy.logwarn(
                "【%s】第%d轮未检测到AprilTag: 已耗时=%.1fs，本轮不会移动，只会等待下一帧",
                stage_label,
                step + 1,
                elapsed,
            )
            rospy.logwarn("第%d轮微调未检测到AprilTag", step + 1)
            save_no_tag_debug_image(step + 1, frame)
            if (not exit_on_aligned) and descend_without_tag and abs(descent_vel_d) > 1e-6:
                rospy.logwarn(
                    "[APRILTAG][%s] NO_TAG_DESCENT step=%d vel_d=%.3f duration=%.2f",
                    stage_label,
                    step + 1,
                    descent_vel_d,
                    refine_duration_sec,
                )
                publish_velocity(pub, 0.0, 0.0, descent_vel_d, 0.0, refine_duration_sec)
                move_command_count += 1
                continue
            rospy.sleep(0.3)
            continue

        tag = results[0]
        tag_id = getattr(tag, "tag_id", None)
        corners = tag.corners / scale
        center_x = float((corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4.0)
        center_y = float((corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4.0)
        image_center_x = frame.shape[1] / 2.0
        image_center_y = frame.shape[0] / 2.0
        error_x = center_x - image_center_x
        error_y = center_y - image_center_y
        distance_px = (error_x ** 2 + error_y ** 2) ** 0.5
        bounds_status = classify_tag_bounds(frame, corners)
        yaw_error_deg = estimate_tag_yaw_error_deg(corners)
        publish_apriltag_pose_from_corners(
            corners=corners,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            pose_cfg=apriltag_pose_cfg,
            tag_id=tag_id,
            stage_label=stage_label,
        )
        target_yaw, yaw_duration, yaw_step_deg = resolve_yaw_command(
            yaw_error_deg=yaw_error_deg,
            yaw_tolerance_deg=yaw_tolerance_deg,
            yaw_rate_dps=yaw_rate_dps,
            yaw_max_step_deg=yaw_max_step_deg,
            yaw_gain=yaw_gain,
            yaw_correction_sign=yaw_correction_sign,
        )

        rospy.loginfo(
            "第%d轮检测结果: tag_id=%s 边界=%s 画面时间=%s 帧间隔=%s 帧签名=%s 是否变化=%s 均值=%.1f 方差=%.1f 标签中心=(%.1f, %.1f) 图像中心=(%.1f, %.1f) 横向偏差dx=%.1f 纵向偏差dy=%.1f 距中心距离=%.1fpx 已耗时=%.1fs",
            step + 1,
            tag_id,
            bounds_status,
            frame_timestamp_text,
            "n/a" if frame_delta_sec is None else f"{frame_delta_sec:.3f}s",
            frame_signature,
            frame_changed,
            frame_mean,
            frame_std,
            center_x,
            center_y,
            image_center_x,
            image_center_y,
            error_x,
            error_y,
            distance_px,
            elapsed,
        )

        # Yaw is corrected before x/y so later forward motion is not amplified
        # by a crooked aircraft heading. Position control still uses image-center
        # dx/dy, which has been the most stable signal on this camera stack.
        aligned = abs(error_x) < pixel_tolerance and abs(error_y) < pixel_tolerance
        allow_control = not (SKIP_EDGE_CONTROL and bounds_status != "in_frame")
        aligned_count = aligned_count + 1 if aligned else 0
        ever_aligned = ever_aligned or (aligned_count >= aligned_required_frames)
        debug_discrete_x = discrete_step(error_x / max(frame.shape[1], 1))
        debug_discrete_y = discrete_step(error_y / max(frame.shape[0], 1))
        rospy.loginfo(
            "【%s】第%d轮判定详情: bounds=%s, aligned=%s, allow_control=%s, dx=%.1f, dy=%.1f, dist=%.1fpx, tolerance=%.1f, discrete_step调试值 x=%.2f y=%.2f",
            stage_label,
            step + 1,
            bounds_status,
            aligned,
            allow_control,
            error_x,
            error_y,
            distance_px,
            pixel_tolerance,
            debug_discrete_x,
            debug_discrete_y,
        )

        rospy.loginfo(
            "[APRILTAG][%s] DETECT step=%d tag_id=%s bounds=%s aligned=%s allow_control=%s dx=%.1f dy=%.1f dist_px=%.1f tol=%.1f yaw_error=%.2f yaw_cmd=%.2f yaw_step=%.2f yaw_fix=%d/%d ds_x=%.2f ds_y=%.2f",
            stage_label,
            step + 1,
            tag_id,
            bounds_status,
            aligned,
            allow_control,
            error_x,
            error_y,
            distance_px,
            pixel_tolerance,
            yaw_error_deg,
            target_yaw,
            yaw_step_deg,
            yaw_correction_count,
            yaw_max_corrections,
            debug_discrete_x,
            debug_discrete_y,
        )
        rospy.loginfo(
            "[APRILTAG][%s] ALIGN_STATE step=%d aligned_count=%d/%d",
            stage_label,
            step + 1,
            aligned_count,
            aligned_required_frames,
        )

        vel_n = 0.0
        vel_e = 0.0
        if allow_control and not aligned:
            vel_n = resolve_axis_velocity(
                error_px=error_y,
                pixel_tolerance=pixel_tolerance,
                max_abs_error_px=REFINE_MAX_ERROR_PX_Y,
                max_speed_mps=REFINE_MAX_SPEED_MPS,
                min_speed_mps=REFINE_MIN_SPEED_MPS,
                positive_when_error_positive=False,
            )
            vel_e = resolve_axis_velocity(
                error_px=error_x,
                pixel_tolerance=pixel_tolerance,
                max_abs_error_px=REFINE_MAX_ERROR_PX_X,
                max_speed_mps=REFINE_MAX_SPEED_MPS,
                min_speed_mps=REFINE_MIN_SPEED_MPS,
                positive_when_error_positive=True,
            )

        yaw_needs_correction = (
            yaw_control_enabled
            and allow_control
            and abs(target_yaw) > 1e-6
            and yaw_correction_count < max(0, int(yaw_max_corrections))
        )

        save_debug_image(
            step=step + 1,
            frame=frame,
            corners=corners,
            tag_id=tag_id,
            frame_timestamp_text=frame_timestamp_text,
            frame_delta_sec=None if frame_delta_sec is None else f"{frame_delta_sec:.3f}",
            frame_signature=frame_signature,
            frame_changed=frame_changed,
            frame_mean=frame_mean,
            frame_std=frame_std,
            center_x=center_x,
            center_y=center_y,
            image_center_x=image_center_x,
            image_center_y=image_center_y,
            error_x=error_x,
            error_y=error_y,
            distance_px=distance_px,
            vel_n=vel_n,
            vel_e=vel_e,
            aligned=aligned,
            bounds_status=bounds_status,
            yaw_error_deg=round(yaw_error_deg, 2),
            target_yaw=target_yaw,
            yaw_correction_count=yaw_correction_count,
            yaw_max_corrections=int(yaw_max_corrections),
        )

        if yaw_needs_correction:
            yaw_correction_count += 1
            aligned_count = 0
            rospy.loginfo(
                "[APRILTAG][%s] YAW_CORRECT step=%d yaw_error=%.2fdeg target_yaw=%.2fdps duration=%.2fs step=%.2fdeg count=%d/%d",
                stage_label,
                step + 1,
                yaw_error_deg,
                target_yaw,
                yaw_duration,
                yaw_step_deg,
                yaw_correction_count,
                yaw_max_corrections,
            )
            publish_velocity(pub, 0.0, 0.0, 0.0, target_yaw, yaw_duration)
            move_command_count += 1
            continue

        if (
            yaw_control_enabled
            and allow_control
            and abs(target_yaw) > 1e-6
            and yaw_correction_count >= max(0, int(yaw_max_corrections))
        ):
            rospy.logwarn(
                "[APRILTAG][%s] YAW_FORCE_CONTINUE step=%d yaw_error=%.2fdeg max_corrections=%d; continue x/y/descent to avoid blocking mission",
                stage_label,
                step + 1,
                yaw_error_deg,
                yaw_max_corrections,
            )

        if aligned and aligned_count < aligned_required_frames:
            stop_motion(pub)
            if alignment_observe_sec > 0:
                rospy.loginfo(
                    "[APRILTAG][%s] HOLD step=%d observe_sec=%.2f aligned_count=%d/%d",
                    stage_label,
                    step + 1,
                    alignment_observe_sec,
                    aligned_count,
                    aligned_required_frames,
                )
                rospy.sleep(alignment_observe_sec)
            continue
        if aligned and exit_on_aligned:
            rospy.loginfo(
                "[APRILTAG][%s] EXIT reason=aligned step=%d dx=%.1f dy=%.1f tol=%.1f no_tag=%d skip=%d move=%d",
                stage_label,
                step + 1,
                error_x,
                error_y,
                pixel_tolerance,
                no_tag_count,
                skip_control_count,
                move_command_count,
            )
            rospy.loginfo(
                "【%s】第%d轮满足结束条件: dx=%.1f, dy=%.1f, 已进入容差 %.1fpx",
                stage_label,
                step + 1,
                error_x,
                error_y,
                pixel_tolerance,
            )
            rospy.loginfo("AprilTag微调完成: 标签已进入中心容差范围")
            stop_motion(pub)
            if post_align_hover_sec > 0:
                rospy.loginfo(
                    "[APRILTAG][%s] POST_ALIGN_HOVER duration=%.2f",
                    stage_label,
                    post_align_hover_sec,
                )
                rospy.sleep(post_align_hover_sec)
            return True
        if aligned:
            rospy.loginfo(
                "[APRILTAG][%s] FINAL_DESCENT_ALIGNED step=%d vel_d=%.3f duration=%.2f",
                stage_label,
                step + 1,
                descent_vel_d,
                refine_duration_sec,
            )
            publish_velocity(pub, 0.0, 0.0, descent_vel_d, 0.0, refine_duration_sec)
            move_command_count += 1
            continue
        if not allow_control:
            skip_control_count += 1
            rospy.logwarn(
                "[APRILTAG][%s] SKIP step=%d reason=edge_guard bounds=%s dx=%.1f dy=%.1f",
                stage_label,
                step + 1,
                bounds_status,
                error_x,
                error_y,
            )
            rospy.logwarn(
                "【%s】第%d轮跳过控制: 虽然检测到tag，但 bounds=%s，不满足 in_frame；由于开启边缘保护，本轮不会移动",
                stage_label,
                step + 1,
                bounds_status,
            )
            rospy.logwarn(
                "本轮跳过控制: 标签边界状态=%s，且已开启边缘保护",
                bounds_status,
            )
            rospy.sleep(0.3)
            continue

        if abs(error_x) >= pixel_tolerance and abs(error_y) >= pixel_tolerance:
            rospy.loginfo("[APRILTAG][%s] PLAN step=%d mode=xy dx=%.1f dy=%.1f", stage_label, step + 1, error_x, error_y)
            rospy.loginfo(
                "【%s】第%d轮将执行双轴修正: dx=%.1f, dy=%.1f 都未进入容差",
                stage_label,
                step + 1,
                error_x,
                error_y,
            )
        elif abs(error_x) >= pixel_tolerance:
            rospy.loginfo("[APRILTAG][%s] PLAN step=%d mode=x dx=%.1f dy=%.1f", stage_label, step + 1, error_x, error_y)
            rospy.loginfo(
                "【%s】第%d轮将执行横向修正: dx=%.1f 未进入容差, dy=%.1f 已在容差内",
                stage_label,
                step + 1,
                error_x,
                error_y,
            )
        elif abs(error_y) >= pixel_tolerance:
            rospy.loginfo("[APRILTAG][%s] PLAN step=%d mode=y dx=%.1f dy=%.1f", stage_label, step + 1, error_x, error_y)
            rospy.loginfo(
                "【%s】第%d轮将执行纵向修正: dy=%.1f 未进入容差, dx=%.1f 已在容差内",
                stage_label,
                step + 1,
                error_y,
                error_x,
            )

        rospy.loginfo(
            "第%d轮执行微调: 南北方向=%s(vel_n=%.3f) 东西方向=%s(vel_e=%.3f) 高度方向=不动(vel_d=0.000) 持续=%.2fs",
            step + 1,
            "向前" if vel_n > 0 else "向南" if vel_n < 0 else "不动",
            vel_n,
            "向左" if vel_e > 0 else "向西" if vel_e < 0 else "不动",
            vel_e,
            REFINE_DURATION_SEC,
        )
        rospy.loginfo(
            "【%s】第%d轮实际输出速度: vel_n=%.3f, vel_e=%.3f, vel_d=0.000, duration=%.2fs",
            stage_label,
            step + 1,
            vel_n,
            vel_e,
            REFINE_DURATION_SEC,
        )
        rospy.loginfo(
            "[APRILTAG][%s] CMD step=%d vel_n=%.3f vel_e=%.3f vel_d=%.3f duration=%.2f",
            stage_label,
            step + 1,
            vel_n,
            vel_e,
            descent_vel_d,
            REFINE_DURATION_SEC,
        )
        move_command_count += 1
        publish_velocity(pub, vel_n, vel_e, descent_vel_d, 0.0, REFINE_DURATION_SEC)

    rospy.logwarn("AprilTag微调达到最大轮数，仍未收敛到目标中心")
    if not exit_on_aligned and ever_aligned:
        stop_motion(pub)
        rospy.loginfo(
            "[APRILTAG][%s] EXIT reason=max_steps_after_final_descent no_tag=%d skip=%d move=%d",
            stage_label,
            no_tag_count,
            skip_control_count,
            move_command_count,
        )
        return True
    rospy.logwarn(
        "[APRILTAG][%s] EXIT reason=max_steps no_tag=%d skip=%d move=%d",
        stage_label,
        no_tag_count,
        skip_control_count,
        move_command_count,
    )
    return False
