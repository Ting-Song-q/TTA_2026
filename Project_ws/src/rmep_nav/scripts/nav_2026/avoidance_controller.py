#!/usr/bin/python3
# coding=UTF-8
"""move_base 速度避障守门：过滤 /cmd_vel_nav 后统一发布 /cmd_vel。"""

import math
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config_loader import load_mission_config
from laser_avoidance import (
    ClearanceHistory,
    _make_twist,
    check_proximity_alert,
    compute_bypass_twist,
    fuse_visual_clearances,
    is_emergency,
    scan_validity,
)


class AvoidanceCommandFilter:
    """保存最新激光帧，并对上游速度指令执行失效保护和绕障修正。"""

    def __init__(self, config=None, clock=None, visual_config=None):
        self.config = config or {}
        self.visual_config = visual_config or {}
        self.clock = clock or time.monotonic
        self.scan = None
        self.scan_received_at = None
        self.clearance_history = ClearanceHistory(self.config)
        self.visual_risks = {
            "front": -1.0,
            "back": -1.0,
            "left": -1.0,
            "right": -1.0,
        }
        self.visual_confidence = {
            "front": 0.0,
            "back": 0.0,
            "left": 0.0,
            "right": 0.0,
        }
        self.visual_received_at = {
            "front": None,
            "back": None,
            "left": None,
            "right": None,
        }

    def update_scan(self, scan, received_at=None):
        """更新激光帧；接收时间使用单调时钟，避免系统时间跳变。"""
        timestamp = self.clock() if received_at is None else received_at
        max_age = float(self.config.get("max_scan_age", 0.30))
        if self.scan_received_at is not None:
            gap = timestamp - self.scan_received_at
            if gap < 0.0 or gap > max_age:
                # 雷达断流或时钟回跳后不得沿用旧畅通历史，必须重新积累有效帧。
                self.clearance_history.reset()
        self.scan = scan
        self.scan_received_at = timestamp
        if not self.clearance_history.update(scan):
            # 非法帧同样切断历史，避免恢复后的首帧被旧数据稀释。
            self.clearance_history.reset()

    def update_visual_observation(
        self, front, back, left, right, confidence, received_at=None
    ):
        """更新相机本帧实际看到的车体方向，负风险表示未观察。"""
        timestamp = self.clock() if received_at is None else received_at
        confidence = float(confidence)
        if not math.isfinite(confidence):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        values = {
            "front": front,
            "back": back,
            "left": left,
            "right": right,
        }
        for sector, value in values.items():
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                continue
            self.visual_risks[sector] = max(0.0, min(1.0, value))
            self.visual_confidence[sector] = confidence
            self.visual_received_at[sector] = timestamp

    def scan_status(self, now=None):
        """返回激光是否可用于运动控制以及对应原因。"""
        now = self.clock() if now is None else now
        valid, reason = scan_validity(self.scan, self.config)
        if not valid:
            return False, reason

        max_age = float(self.config.get("max_scan_age", 0.30))
        if self.scan_received_at is None:
            return False, "scan_timestamp_unavailable"
        if now - self.scan_received_at > max_age:
            return False, "scan_stale"
        if not self.clearance_history.ready():
            return False, "scan_history_warming"
        return True, "ok"

    def decision_clearances(self, now=None):
        """返回当前用于控制的雷达/视觉融合净空。"""
        lidar_ready = self.clearance_history.ready()
        lidar_clearances = self.clearance_history.clearances()
        if not self.visual_config.get("enabled", False):
            return lidar_clearances
        now = self.clock() if now is None else now
        max_age = float(self.visual_config.get("max_observation_age", 0.50))
        visual_risks = {}
        visual_confidence = {}
        for sector in ("front", "back", "left", "right"):
            received_at = self.visual_received_at[sector]
            if received_at is None or now - received_at > max_age:
                continue
            visual_risks[sector] = self.visual_risks[sector]
            visual_confidence[sector] = self.visual_confidence[sector]

        if (
            self.visual_config.get("fusion_mode") == "vision_preferred"
            and not lidar_ready
        ):
            # 没有雷达时，未被相机近期观察的方向保持为零净空。
            lidar_clearances = {
                "front": 0.0,
                "back": 0.0,
                "left": 0.0,
                "right": 0.0,
            }
        return fuse_visual_clearances(
            lidar_clearances,
            visual_risks,
            visual_confidence,
            self.config,
            self.visual_config,
        )

    def visual_status(self, now=None):
        """检查四个车体方向的视觉缓存是否都新鲜且可信。"""
        now = self.clock() if now is None else now
        max_age = float(self.visual_config.get("max_observation_age", 0.50))
        min_confidence = float(self.visual_config.get("min_confidence", 0.45))
        for sector in ("front", "back", "left", "right"):
            received_at = self.visual_received_at[sector]
            if received_at is None or now - received_at > max_age:
                return False, f"visual_{sector}_stale"
            if self.visual_confidence[sector] < min_confidence:
                return False, f"visual_{sector}_low_confidence"
        return True, "ok"

    def control_sensor_status(self, now=None):
        """根据融合模式选择雷达或四向视觉作为当前控制依据。"""
        lidar_ok, lidar_reason = self.scan_status(now)
        if self.visual_config.get("fusion_mode") != "vision_preferred":
            return lidar_ok, lidar_reason

        visual_ok, visual_reason = self.visual_status(now)
        if visual_ok:
            return True, "vision_ready"
        if self.visual_config.get("fallback_to_lidar", True) and lidar_ok:
            return True, "lidar_fallback"
        return False, visual_reason

    def filter_command(self, command, now=None):
        """过滤一条速度指令，返回 (输出指令, 工作模式)。"""
        vx = command.linear.x
        vy = command.linear.y
        wz = command.angular.z

        if not self.config.get("enabled", False):
            return _make_twist(vx, vy, wz), "disabled"

        healthy, reason = self.control_sensor_status(now)
        if not healthy:
            if self.config.get("fail_closed", True):
                return _make_twist(), reason
            return _make_twist(vx, vy, wz), reason

        if reason == "vision_ready":
            # 机械臂环视存在方向缓存延迟，视觉单独控制时强制低速。
            max_linear = float(
                self.visual_config.get("vision_only_max_linear_speed", 0.05)
            )
            max_angular = float(
                self.visual_config.get("vision_only_max_angular_speed", 0.20)
            )
            vx = max(-max_linear, min(max_linear, vx))
            vy = max(-max_linear, min(max_linear, vy))
            wz = max(-max_angular, min(max_angular, wz))

        # 当前原始帧只负责急停，多帧中值负责普通绕障，兼顾响应速度和抗跳变。
        lidar_ok, _ = self.scan_status(now)
        if lidar_ok and is_emergency(
            self.scan,
            self.config,
            vx=vx,
            vy=vy,
            wz=wz,
        ):
            return _make_twist(), "emergency_stop"

        clearances = self.decision_clearances(now)
        if is_emergency(
            self.scan,
            self.config,
            vx=vx,
            vy=vy,
            wz=wz,
            clearances=clearances,
        ):
            return _make_twist(), "emergency_stop"

        return compute_bypass_twist(
            self.scan,
            vx,
            vy,
            wz,
            self.config,
            clearances=clearances,
        )


class AvoidanceControllerNode:
    """ROS 节点封装：接收导航速度和激光数据，周期性输出安全速度。"""

    def __init__(self, rospy_module):
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import LaserScan
        from std_msgs.msg import Float32MultiArray

        self.rospy = rospy_module
        config_path = self.rospy.get_param(
            "~config", str(_SCRIPT_DIR / "mission_config.yaml")
        )
        mission_config = load_mission_config(config_path)
        self.config = mission_config.get("obstacle_avoidance", {})
        self.visual_config = mission_config.get("visual_avoidance", {})
        self.filter = AvoidanceCommandFilter(
            self.config, visual_config=self.visual_config
        )

        scan_topic = self.rospy.get_param(
            "~scan_topic", self.config.get("scan_topic", "/scan")
        )
        nav_cmd_topic = self.rospy.get_param(
            "~nav_cmd_topic", self.config.get("nav_cmd_topic", "/cmd_vel_nav")
        )
        output_cmd_topic = self.rospy.get_param(
            "~output_cmd_topic", self.config.get("output_cmd_topic", "/cmd_vel")
        )

        self.cmd_publisher = self.rospy.Publisher(
            output_cmd_topic, Twist, queue_size=1
        )
        self.scan_subscriber = self.rospy.Subscriber(
            scan_topic, LaserScan, self._scan_callback, queue_size=1
        )
        self.nav_cmd_subscriber = self.rospy.Subscriber(
            nav_cmd_topic, Twist, self._nav_cmd_callback, queue_size=1
        )
        self.visual_subscriber = None
        if self.visual_config.get("enabled", False):
            self.visual_subscriber = self.rospy.Subscriber(
                self.visual_config.get(
                    "observation_topic", "/vision_obstacle/risks"
                ),
                Float32MultiArray,
                self._visual_callback,
                queue_size=1,
            )

        self.latest_command = Twist()
        self.command_received_at = None
        self.last_mode = None
        control_rate = max(1.0, float(self.config.get("control_rate", 20)))
        self.timer = self.rospy.Timer(
            self.rospy.Duration(1.0 / control_rate), self._control_callback
        )
        self.rospy.on_shutdown(self.stop)

        self.rospy.loginfo(
            "避障守门节点已启动: %s -> %s, scan=%s",
            nav_cmd_topic,
            output_cmd_topic,
            scan_topic,
        )

    def _scan_callback(self, message):
        """保存最新激光帧，具体安全判断放在控制周期中执行。"""
        self.filter.update_scan(message)

    def _nav_cmd_callback(self, message):
        """保存 move_base 最新速度，避免在回调中直接控制底盘。"""
        self.latest_command = message
        self.command_received_at = time.monotonic()

    def _visual_callback(self, message):
        """接收视觉节点发布的车体四方向风险和总体置信度。"""
        if len(message.data) < 5:
            self.rospy.logwarn_throttle(1.0, "视觉障碍消息长度不足")
            return
        self.filter.update_visual_observation(
            message.data[0],
            message.data[1],
            message.data[2],
            message.data[3],
            message.data[4],
        )

    def _control_callback(self, _event):
        """按固定频率检查指令与激光新鲜度，并发布唯一的底盘速度。"""
        now = time.monotonic()
        max_cmd_age = float(self.config.get("max_cmd_age", 0.50))
        if self.command_received_at is None:
            command = _make_twist()
            mode = "cmd_unavailable"
        elif now - self.command_received_at > max_cmd_age:
            command = _make_twist()
            mode = "cmd_stale"
        else:
            command, mode = self.filter.filter_command(
                self.latest_command, now=now
            )

        self.cmd_publisher.publish(command)
        self._report_mode(mode)
        self._report_proximity()

    def _report_mode(self, mode):
        """仅在模式变化时打印日志，避免控制循环刷屏。"""
        if mode == self.last_mode:
            return
        self.last_mode = mode

        if mode in {
            "scan_unavailable",
            "scan_empty",
            "angle_increment_invalid",
            "valid_points_insufficient",
            "scan_timestamp_unavailable",
            "scan_stale",
            "scan_history_warming",
            "visual_front_stale",
            "visual_back_stale",
            "visual_left_stale",
            "visual_right_stale",
            "emergency_stop",
        }:
            self.rospy.logwarn("避障守门停止输出，原因: %s", mode)
        elif mode.startswith("bypass"):
            self.rospy.loginfo("避障守门进入反应式绕障: %s", mode)
        else:
            self.rospy.loginfo("避障守门模式: %s", mode)

    def _report_proximity(self):
        """记录贴近障碍告警，但不改变已经完成的速度决策。"""
        alert, distance, sector = check_proximity_alert(
            self.filter.scan,
            self.config,
            clearances=self.filter.decision_clearances(),
        )
        if alert:
            self.rospy.logwarn_throttle(
                1.0,
                "障碍贴近告警: sector=%s distance=%.3fm",
                sector,
                distance,
            )

    def stop(self):
        """节点退出前连续发布零速度，降低底盘保留末次指令的风险。"""
        for _ in range(3):
            self.cmd_publisher.publish(_make_twist())
            time.sleep(0.02)


def main():
    """启动独立避障守门节点。"""
    import rospy

    rospy.init_node("avoidance_controller")
    AvoidanceControllerNode(rospy)
    rospy.spin()


if __name__ == "__main__":
    main()
