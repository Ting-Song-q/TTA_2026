#!/usr/bin/python3
# coding=UTF-8
"""机械臂视觉障碍节点：将相机视野映射为车体四方向风险。"""

import math
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config_loader import load_mission_config
from vision.camera_sector_mapper import CameraScanPlanner, map_camera_risks_to_body
from vision.obstacle_detector import VisualObstacleDetector


class VisionObstacleNode:
    """订阅相机图像并以固定最高频率发布视觉障碍风险。"""

    def __init__(self, rospy_module):
        from cv_bridge import CvBridge, CvBridgeError
        from sensor_msgs.msg import Image
        from std_msgs.msg import Float32, Float32MultiArray

        self.rospy = rospy_module
        self.bridge = CvBridge()
        self.bridge_error = CvBridgeError
        config_path = self.rospy.get_param(
            "~config", str(_SCRIPT_DIR / "mission_config.yaml")
        )
        self.config = load_mission_config(config_path)
        self.visual_cfg = self.config.get("visual_avoidance", {})
        self.detector = VisualObstacleDetector(self.config)
        self.output_type = Float32MultiArray
        self.angle_type = Float32
        self.last_processed_at = 0.0
        self.camera_yaw_deg = float(self.visual_cfg.get("fixed_camera_yaw_deg", 0.0))
        if not math.isfinite(self.camera_yaw_deg):
            self.camera_yaw_deg = 0.0
        self.camera_yaw_received_at = None

        image_topic = self.visual_cfg.get(
            "image_topic",
            self.config.get("vision", {}).get("image_topic", "/ep_cam/image_raw"),
        )
        observation_topic = self.visual_cfg.get(
            "observation_topic", "/vision_obstacle/risks"
        )
        self.publisher = self.rospy.Publisher(
            observation_topic, Float32MultiArray, queue_size=1
        )
        self.debug_publisher = None
        if self.visual_cfg.get("publish_debug_image", True):
            self.debug_publisher = self.rospy.Publisher(
                self.visual_cfg.get(
                    "debug_image_topic", "/vision_obstacle/debug_image"
                ),
                Image,
                queue_size=1,
            )
        self.subscriber = self.rospy.Subscriber(
            image_topic, Image, self._image_callback, queue_size=1
        )
        self.yaw_subscriber = self.rospy.Subscriber(
            self.visual_cfg.get("camera_yaw_topic", "/camera_arm/yaw_deg"),
            Float32,
            self._camera_yaw_callback,
            queue_size=1,
        )

        self.scan_cfg = self.visual_cfg.get("arm_scan", {})
        self.scan_planner = None
        self.scan_target_publisher = None
        self.scan_timer = None
        if self.scan_cfg.get("enabled", False):
            self.scan_planner = CameraScanPlanner(
                angles=self.scan_cfg.get("angles_deg", [0, 90, 180, -90]),
                tolerance_deg=self.scan_cfg.get("tolerance_deg", 5.0),
                dwell_time=self.scan_cfg.get("dwell_time", 0.35),
            )
            self.scan_target_publisher = self.rospy.Publisher(
                self.visual_cfg.get(
                    "camera_yaw_target_topic", "/camera_arm/yaw_target_deg"
                ),
                Float32,
                queue_size=1,
            )
            self.scan_timer = self.rospy.Timer(
                self.rospy.Duration(0.10), self._scan_timer_callback
            )
        self.rospy.loginfo(
            "视觉避障节点已启动: image=%s observation=%s mode=%s",
            image_topic,
            observation_topic,
            self.visual_cfg.get("fusion_mode", "monitor_only"),
        )

    def _camera_yaw_callback(self, message):
        """保存机械臂相机相对 base_link 的实时偏航角。"""
        yaw_deg = float(message.data)
        if not math.isfinite(yaw_deg):
            self.rospy.logwarn_throttle(1.0, "机械臂相机 yaw 反馈不是有限数值")
            return
        self.camera_yaw_deg = yaw_deg
        self.camera_yaw_received_at = time.monotonic()

    def _camera_yaw_is_fresh(self, now):
        """检查机械臂角度反馈；固定相机模式不要求反馈话题。"""
        feedback_required = (
            self.visual_cfg.get("require_camera_yaw_feedback", False)
            or self.scan_planner is not None
        )
        if not feedback_required:
            return True
        max_age = float(self.visual_cfg.get("max_camera_yaw_age", 0.50))
        return (
            self.camera_yaw_received_at is not None
            and now - self.camera_yaw_received_at <= max_age
        )

    def _scan_timer_callback(self, _event):
        """发布扫描目标角；机械臂驱动负责执行并反馈真实 yaw。"""
        now = time.monotonic()
        if self.scan_planner is None or not self._camera_yaw_is_fresh(now):
            return
        target = self.scan_planner.update(self.camera_yaw_deg, now)
        message = self.angle_type()
        message.data = float(target)
        self.scan_target_publisher.publish(message)

    def _image_callback(self, message):
        """处理最新相机帧，过高输入频率时主动跳帧降低计算负载。"""
        now = time.monotonic()
        max_rate = max(0.5, float(self.visual_cfg.get("max_processing_rate", 8.0)))
        if now - self.last_processed_at < 1.0 / max_rate:
            return
        if not self._camera_yaw_is_fresh(now):
            self.rospy.logwarn_throttle(1.0, "机械臂相机 yaw 反馈失效")
            return
        if self.scan_planner is not None and not self.scan_planner.observation_ready(
            self.camera_yaw_deg,
            now,
            self.scan_cfg.get("settle_time", 0.15),
        ):
            self.rospy.loginfo_throttle(1.0, "机械臂转动或尚未稳定，跳过当前图像")
            return
        self.last_processed_at = now

        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except self.bridge_error as exc:
            self.rospy.logwarn_throttle(1.0, "视觉避障图像转换失败: %s", exc)
            return

        observation = self.detector.detect(frame)
        body_risks = map_camera_risks_to_body(
            self.camera_yaw_deg,
            observation.front.risk,
            observation.left.risk,
            observation.right.risk,
            self.visual_cfg.get("camera_horizontal_fov_deg", 70.0),
        )
        output = self.output_type()
        output.data = [
            float(body_risks["front"]),
            float(body_risks["back"]),
            float(body_risks["left"]),
            float(body_risks["right"]),
            float(observation.confidence),
        ]
        self.publisher.publish(output)

        if self.debug_publisher is not None:
            debug = self.detector.draw_debug(frame, observation)
            debug_message = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_message.header = message.header
            self.debug_publisher.publish(debug_message)

        self.rospy.loginfo_throttle(
            1.0,
            "视觉车体风险: front=%.2f back=%.2f left=%.2f right=%.2f yaw=%.1f",
            body_risks["front"],
            body_risks["back"],
            body_risks["left"],
            body_risks["right"],
            self.camera_yaw_deg,
        )


def main():
    """启动视觉障碍检测节点。"""
    import rospy

    rospy.init_node("vision_obstacle")
    VisionObstacleNode(rospy)
    rospy.spin()


if __name__ == "__main__":
    main()
