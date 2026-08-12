#!/usr/bin/python3
# coding=UTF-8

import time
from pathlib import Path

import cv2
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class CameraCapture:
    """订阅 ROS 相机话题，去畸变并取稳定帧。"""

    def __init__(self, config=None, image_topic="/ep_cam/image_raw", camera_yaml=None):
        vision_cfg = (config or {}).get("vision", {})
        self.image_topic = vision_cfg.get("image_topic", image_topic)
        self.settle_frames = vision_cfg.get("settle_frames", 5)
        self.timeout = vision_cfg.get("frame_timeout", 5.0)
        self.max_frame_age = float(vision_cfg.get("frame_max_age", 0.5))

        self.bridge = CvBridge()
        self._latest = None
        self._latest_received_at = None
        self._sequence = 0
        self._sub = rospy.Subscriber(
            self.image_topic, Image, self._callback, queue_size=1
        )

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
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._camera_matrix = np.array(data["camera_matrix"]["data"]).reshape(3, 3)
        self._dist_coeffs = np.array(data["distortion_coefficients"]["data"])

    def _callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("cv_bridge error: %s", exc)
            return
        if self._camera_matrix is not None:
            img = cv2.undistort(img, self._camera_matrix, self._dist_coeffs)
        self._latest = img
        self._latest_received_at = time.time()
        self._sequence += 1

    def _stream_is_fresh(self):
        if self._latest is None or self._latest_received_at is None:
            return False
        return time.time() - self._latest_received_at <= self.max_frame_age

    def wait_for_stream(self, timeout=None):
        timeout = timeout or self.timeout
        start = time.time()
        while not self._stream_is_fresh() and not rospy.is_shutdown():
            if time.time() - start > timeout:
                return False
            rospy.sleep(0.05)
        return self._stream_is_fresh()

    def get_frame(self, discard=0):
        if not self.wait_for_stream():
            return None
        target_sequence = self._sequence + max(discard, self.settle_frames)
        deadline = time.time() + self.timeout
        while self._sequence < target_sequence and not rospy.is_shutdown():
            if time.time() >= deadline:
                return None
            rospy.sleep(0.05)
        return self._latest.copy() if self._stream_is_fresh() else None

    def save_debug(self, directory, name, image):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        cv2.imwrite(str(path), image)
        rospy.loginfo("saved debug image: %s", path)
        return path


