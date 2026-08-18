#!/usr/bin/python3
# coding=UTF-8
"""nav_rescue_2026_closed_loop：2026 年救援任务闭环。

  等待无人机目标 → 取货区 → 视觉抓取 → 装货区 → 装货/通知起飞 →
  等待无人机投送 → 救援区 → 卸货/通知完成 → 返航停车区

运动控制（默认 map_laser，与 test/run_map_laser_nav 同栈）：

  - map_laser：AMCL 追点 + /map+/scan 融合动态避障（TEB-style 多候选速度）
  - move_base：全局导航 + 到达后 TF 航向/XY 闭环修正（旧方案，--nav-mode move_base）

依赖启动栈：
  1) roscore
  2) roslaunch rmep_base rmep_base.launch
  3) roslaunch rmep_nav map_amcl_move.launch map:=changd.yaml
     # /map + AMCL；航点来自 mission_config（由 map/changd.yaml 像素 zones 换算）
  4) python3 nav_rescue_2026_closed_loop.py [--skip-drone --rescue 2]
"""

from __future__ import print_function

import argparse
import math
import sys
import time
from pathlib import Path

import rospy
import actionlib
import tf
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import LaserScan
from tf.transformations import quaternion_from_euler, euler_from_quaternion

_HERE = Path(__file__).resolve().parent
_TEST_DIR = _HERE / "test"  
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from rescue_protocol import (  # noqa: E402
    RescueOrder,
    wait_for_rescue_target,
    notify_loading_done,
    wait_for_delivery_done,
    notify_unload_done,
)
from config_loader import load_mission_config, as_pose  # noqa: E402
from map_laser_nav import MapLaserNav  # noqa: E402
from laser_avoidance import is_emergency  # noqa: E402

# 视觉模块可选；若未接入可视为占位
_VISION_AVAILABLE = False
try:
    from vision.camera_capture import CameraCapture  # noqa: E402
    from vision.pickup_detector import PickupDetector  # noqa: E402
    from vision.grasp_controller import GraspController  # noqa: E402

    _VISION_AVAILABLE = True
except Exception as _vision_err:  # pragma: no cover
    rospy.logwarn(
        "[ClosedLoop] 视觉模块未导入: %s; vision_grasp 将使用占位实现", _vision_err
    )


class RescueMission:
    WAIT_DRONE_CMD = 0
    GO_TO_PICKUP_AREA = 1
    VISION_GRASP = 2
    GO_TO_LOADING_AREA = 3
    PUT_ON_DRONE = 4
    NOTIFY_TAKEOFF = 5
    GO_TO_RESCUE_AREA = 6
    UNLOAD_GOODS = 7
    NOTIFY_UNLOAD_DONE = 8
    GO_TO_HOME_AREA = 9
    FINISH = 10

    _NAMES = {
        0: "WAIT_DRONE_CMD",
        1: "GO_TO_PICKUP_AREA",
        2: "VISION_GRASP",
        3: "GO_TO_LOADING_AREA",
        4: "PUT_ON_DRONE",
        5: "NOTIFY_TAKEOFF",
        6: "GO_TO_RESCUE_AREA",
        7: "UNLOAD_GOODS",
        8: "NOTIFY_UNLOAD_DONE",
        9: "GO_TO_HOME_AREA",
        10: "FINISH",
    }

    @classmethod
    def name(cls, state):
        return cls._NAMES.get(state, str(state))


class NavRescue2026ClosedLoop(object):
    def __init__(
        self,
        autostart=True,
        skip_drone=False,
        rescue_zone=None,
        cmd_vel_topic="/cmd_vel",
        nav_mode="map_laser",
    ):
        rospy.init_node("nav_rescue_2026_closed_loop", anonymous=True)

        self.skip_drone = bool(skip_drone)
        self.rescue_zone = int(rescue_zone) if rescue_zone is not None else None
        self.cmd_vel_topic = str(cmd_vel_topic) if cmd_vel_topic else "/cmd_vel"
        self.nav_mode = str(nav_mode or "map_laser").strip().lower()
        if self.nav_mode not in ("map_laser", "move_base"):
            rospy.logwarn(
                "[ClosedLoop] 未知 nav_mode=%s，回退为 map_laser", self.nav_mode
            )
            self.nav_mode = "map_laser"

        self.config = self._load_config()
        self.state = RescueMission.WAIT_DRONE_CMD
        self.order = None
        self.map_nav = None
        self.move_base = None

        car_cfg = dict(self.config.get("car") or {})
        self.speed = float(car_cfg.get("speed", 0.3))
        self.turn_speed = float(car_cfg.get("turn_speed", 0.5))
        self.align_speed = float(car_cfg.get("align_speed", 0.5))
        self.avoidance_cfg = self._build_avoidance_cfg()

        # TF / cmd_vel（精修与短距位移共用）
        self.rate = rospy.Rate(100)
        self.tf_listener = tf.TransformListener()
        self.velocity_publisher = rospy.Publisher(
            self.cmd_vel_topic, Twist, queue_size=10
        )
        self.laser_subscriber = rospy.Subscriber(
            "/scan", LaserScan, self.laser_callback, queue_size=1
        )
        self.laser_data = None

        # 视觉模块（可选）
        self._camera = None
        self._detector = None
        self._grasp = None
        self._vision_initialized = False

        # 任务状态
        self._loading_completed = False
        self._unload_completed = False
        self._current_zone = None
        self._current_zone_id = None

        # 初始位姿：使用配置中的 parking 作为 AMCL 初始值
        parking = (self.config.get("zones") or {}).get("parking") or {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
        }
        self.initial_pose_for_amcl(
            float(parking.get("x", 0.0)),
            float(parking.get("y", 0.0)),
            0.0,
            float(parking.get("yaw", 0.0)),
        )

        if self.nav_mode == "map_laser":
            self._init_map_laser_nav()
        else:
            self._init_move_base_client()

        rospy.loginfo(
            "[ClosedLoop] ready nav_mode=%s (dynamic avoidance via %s)",
            self.nav_mode,
            "MapLaserNav map+scan" if self.nav_mode == "map_laser" else "move_base/TEB",
        )
        zones = self.config.get("zones") or {}
        for name in ("parking", "pickup", "loading"):
            pose = zones.get(name) or {}
            rospy.loginfo(
                "[ClosedLoop]  zones.%s = (%.3f, %.3f, yaw=%.3f)",
                name,
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("yaw", 0.0)),
            )
        for zid in sorted((zones.get("rescue") or {}).keys(), key=lambda x: int(x)):
            pose = zones["rescue"][zid]
            rospy.loginfo(
                "[ClosedLoop]  zones.rescue_%s = (%.3f, %.3f, yaw=%.3f)",
                zid,
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("yaw", 0.0)),
            )

        if autostart:
            self.run_mission()

    def _build_avoidance_cfg(self):
        """合并 mission 避障配置与测试脚本 TEB-style 默认值。"""
        cfg = dict(self.config.get("obstacle_avoidance") or {})
        defaults = {
            "enabled": True,
            "fail_closed": True,
            "lidar_mount": "rear",
            "safe_distance": 0.20,
            "side_safe_distance": 0.15,
            "critical_distance": 0.10,
            "emergency_stop_distance": 0.05,
            "bypass_speed": 0.10,
            "max_linear_speed": 0.30,
            "control_rate": 20,
            "move_timeout": 90.0,
            "min_obstacle_dist": 0.10,
            "inflation_dist": 0.40,
            "weight_obstacle": 50.0,
            "weight_goal": 1.0,
            "weight_velocity": 1.0,
            "weight_path": 5.0,
            "stuck_timeout": 5.0,
            "recovery_duration": 1.0,
        }
        for key, value in defaults.items():
            cfg.setdefault(key, value)
        return cfg

    def _init_map_laser_nav(self):
        map_cfg = dict(self.config.get("map_nav") or {})
        map_cfg.setdefault("require_map", True)
        map_cfg.setdefault("require_amcl", True)
        map_cfg.setdefault("max_amcl_age", 1.0)
        map_cfg.setdefault("max_range", 3.0)
        map_cfg.setdefault("inflate_m", 0.08)
        map_cfg.setdefault("pos_tol", 0.12)
        map_cfg.setdefault("yaw_tol", 0.08)
        map_cfg.setdefault("approach_dist", 0.45)
        map_cfg.setdefault("commit_dist", 0.25)
        map_cfg.setdefault("overshoot_accept", 0.28)
        rospy.loginfo("[ClosedLoop] 初始化 MapLaserNav（动态避障）...")
        self.map_nav = MapLaserNav(
            speed=self.speed,
            turn_speed=self.turn_speed,
            avoidance_cfg=self.avoidance_cfg,
            map_cfg=map_cfg,
            log_prefix="[MapNav]",
            node_name="nav_rescue_map_laser",
        )
        # 与任务节点共用 cmd_vel，避免双发布者互抢
        self.map_nav.velocity_publisher = self.velocity_publisher
        if getattr(self.map_nav, "_turn", None) is not None:
            self.map_nav._turn.velocity_publisher = self.velocity_publisher
        rospy.loginfo("[ClosedLoop] MapLaserNav ready")

    def _init_move_base_client(self):
        rospy.loginfo("[ClosedLoop] 等待 move_base action server...")
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        while not self.move_base.wait_for_server(rospy.Duration(1.0)):
            if rospy.is_shutdown():
                sys.exit(0)
        rospy.loginfo("[ClosedLoop] move_base server ready")

    def _load_config(self):
        config_path = Path(__file__).resolve().parent / "mission_config.yaml"
        return load_mission_config(config_path)

    def _set_state(self, state):
        self.state = state
        rospy.loginfo("[Mission] %s", RescueMission.name(state))

    def _abort(self, reason):
        rospy.logerr("[Mission] abort: %s", reason)
        self.stop()
        return False

    def laser_callback(self, laser_data):
        self.laser_data = laser_data

    # ------------------------------------------------------------------
    # 2025 年 move_base + TF 闭环运动控制
    # ------------------------------------------------------------------

    def make_move_base_goal(self, x, y, yaw):
        goal = MoveBaseGoal()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        q = quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        return goal

    def initial_pose_for_amcl(self, x, y, z, yaw):
        initialpose_pub = rospy.Publisher(
            "/initialpose", PoseWithCovarianceStamped, queue_size=10
        )
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.stamp = rospy.Time.now()
        initial_pose.header.frame_id = "map"
        initial_pose.pose.pose.position.x = x
        initial_pose.pose.pose.position.y = y
        initial_pose.pose.pose.position.z = z
        quaternion = quaternion_from_euler(0, 0, yaw)
        initial_pose.pose.pose.orientation.x = quaternion[0]
        initial_pose.pose.pose.orientation.y = quaternion[1]
        initial_pose.pose.pose.orientation.z = quaternion[2]
        initial_pose.pose.pose.orientation.w = quaternion[3]
        initial_pose.pose.covariance = [
            0.25, 0, 0, 0, 0, 0,
            0, 0.25, 0, 0, 0, 0,
            0, 0, 0.25, 0, 0, 0,
            0, 0, 0, 0.068, 0, 0,
            0, 0, 0, 0, 0.068, 0,
            0, 0, 0, 0, 0, 0.068,
        ]
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < 0.5:
            initialpose_pub.publish(initial_pose)
            self.rate.sleep()
        rospy.sleep(1)
        rospy.loginfo("[ClosedLoop] initialpose published")

    def adjust_pose(self, target_yaw):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            try:
                (current_position, current_rotation) = self.tf_listener.lookupTransform(
                    "map", "base_link", rospy.Time(0)
                )
                rate.sleep()
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        _, _, current_yaw = euler_from_quaternion(current_rotation)
        error = current_yaw - target_yaw
        error = math.atan2(math.sin(error), math.cos(error))
        if abs(error) > 0.025:
            # goal_rotation 需要为 -error，才能转到目标航向；
            # 之前传 abs(error) 会导致向相反方向多转。
            self.turn_ang(0.6, -error)

    def stop(self):
        if self.map_nav is not None:
            self.map_nav.stop()
        if self.move_base is not None:
            try:
                self.move_base.cancel_all_goals()
            except Exception:
                pass
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        vel_msg.linear.y = 0
        self.velocity_publisher.publish(vel_msg)

    def turn_ang(self, ang_speed, goal_rotation, tolerance=0.02, timeout=10.0):
        """闭环转向：以 ang_speed 为最大角速度，转过 goal_rotation 弧度。"""
        if abs(goal_rotation) < tolerance:
            return

        rate = rospy.Rate(20)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        max_speed = abs(ang_speed)
        direction = 1.0 if goal_rotation > 0 else -1.0
        Kp = 1.5

        # 获取起始航向
        start_yaw = None
        while not rospy.is_shutdown():
            try:
                (_, rotation) = self.tf_listener.lookupTransform(
                    "map", "base_link", rospy.Time(0)
                )
                start_yaw = euler_from_quaternion(rotation)[2]
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        if start_yaw is None:
            return

        target_yaw = start_yaw + direction * abs(goal_rotation)
        target_yaw = math.atan2(math.sin(target_yaw), math.cos(target_yaw))

        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            try:
                (_, rotation) = self.tf_listener.lookupTransform(
                    "map", "base_link", rospy.Time(0)
                )
                _, _, current_yaw = euler_from_quaternion(rotation)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rate.sleep()
                continue

            error = current_yaw - target_yaw
            error = math.atan2(math.sin(error), math.cos(error))
            if abs(error) < tolerance:
                break

            angular_z = -Kp * error
            angular_z = max(-max_speed, min(max_speed, angular_z))
            twist_ang = Twist()
            twist_ang.angular.z = angular_z
            twist_ang.linear.x = 0
            twist_ang.linear.y = 0
            self.velocity_publisher.publish(twist_ang)
            rate.sleep()

        self.stop()

    def go_linear_x(self, linear_speed, goal_distance, tolerance=0.02, timeout=10.0):
        """闭环直线：以 linear_speed 为最大线速度，沿当前 base_link X 方向移动 goal_distance 米。"""
        if abs(goal_distance) < tolerance:
            return

        rate = rospy.Rate(20)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        max_speed = abs(linear_speed)
        direction = 1.0 if linear_speed > 0 else -1.0
        Kp = 1.0
        avoid = self.avoidance_cfg if getattr(self, "avoidance_cfg", None) else {}

        # 获取起始位姿
        start_pos = None
        start_yaw = None
        while not rospy.is_shutdown():
            try:
                (start_pos, start_rot) = self.tf_listener.lookupTransform(
                    "map", "base_link", rospy.Time(0)
                )
                start_yaw = euler_from_quaternion(start_rot)[2]
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        if start_pos is None or start_yaw is None:
            return

        target_x = start_pos[0] + direction * abs(goal_distance) * math.cos(start_yaw)
        target_y = start_pos[1] + direction * abs(goal_distance) * math.sin(start_yaw)

        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            try:
                (current_pos, _) = self.tf_listener.lookupTransform(
                    "map", "base_link", rospy.Time(0)
                )
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rate.sleep()
                continue

            dx = target_x - current_pos[0]
            dy = target_y - current_pos[1]
            remaining = dx * math.cos(start_yaw) + dy * math.sin(start_yaw)
            if abs(remaining) < tolerance:
                break
            if direction > 0 and remaining < 0:
                break
            if direction < 0 and remaining > 0:
                break

            linear_x = Kp * remaining
            linear_x = max(-max_speed, min(max_speed, linear_x))
            if avoid.get("enabled", True) and is_emergency(
                self.laser_data, avoid, linear_x, 0.0, 0.0
            ):
                rospy.logwarn_throttle(
                    1.0, "[ClosedLoop] go_linear_x 紧急停车（动态障碍）"
                )
                self.stop()
                rate.sleep()
                continue
            twist_linear = Twist()
            twist_linear.linear.x = linear_x
            twist_linear.linear.y = 0
            twist_linear.angular.z = 0
            self.velocity_publisher.publish(twist_linear)
            rate.sleep()

        self.stop()

    def go_linear_y(self, linear_speed, goal_distance, tolerance=0.02, timeout=10.0):
        """闭环侧移：以 linear_speed 为最大速度，沿 base_link Y 方向移动 goal_distance 米。"""
        if abs(goal_distance) < tolerance:
            return

        rate = rospy.Rate(20)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        max_speed = abs(linear_speed)
        direction = 1.0 if goal_distance > 0 else -1.0
        Kp = 1.0

        start_pos = None
        start_yaw = None
        while not rospy.is_shutdown():
            try:
                (start_pos, start_rot) = self.tf_listener.lookupTransform(
                    "map", "base_link", rospy.Time(0)
                )
                start_yaw = euler_from_quaternion(start_rot)[2]
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        if start_pos is None or start_yaw is None:
            return

        target_projection = direction * abs(goal_distance)

        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            try:
                (current_pos, current_rot) = self.tf_listener.lookupTransform(
                    "map", "base_link", rospy.Time(0)
                )
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rate.sleep()
                continue

            current_yaw = euler_from_quaternion(current_rot)[2]
            dx = current_pos[0] - start_pos[0]
            dy = current_pos[1] - start_pos[1]
            # base_link Y 轴在 map 中的单位向量是 (-sin(yaw), cos(yaw))
            progress = -dx * math.sin(current_yaw) + dy * math.cos(current_yaw)
            remaining = target_projection - progress

            if abs(remaining) < tolerance:
                break
            if direction > 0 and remaining < 0:
                break
            if direction < 0 and remaining > 0:
                break

            linear_y = Kp * remaining
            linear_y = max(-max_speed, min(max_speed, linear_y))
            twist_linear = Twist()
            twist_linear.linear.x = 0
            twist_linear.linear.y = linear_y
            twist_linear.angular.z = 0
            self.velocity_publisher.publish(twist_linear)
            rate.sleep()

        self.stop()

    def move(self, goal, target_yaw, is_xy_fix, timeout=30, retries=3):
        """2025 年 move_base 导航 + TF 闭环修正。"""
        attempt = 0
        while attempt < retries:
            attempt += 1
            if attempt > 1:
                # 重试前用当前 TF 位姿重新初始化 AMCL
                rate = rospy.Rate(20)
                while not rospy.is_shutdown():
                    try:
                        (current_position, rotation) = self.tf_listener.lookupTransform(
                            "map", "base_link", rospy.Time(0)
                        )
                        rate.sleep()
                        break
                    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                        continue
                _, _, yaw = euler_from_quaternion(rotation)
                self.initial_pose_for_amcl(current_position[0], current_position[1], 0, yaw)

            self.move_base.send_goal(goal)
            success = self.move_base.wait_for_result(rospy.Duration(timeout))
            if success:
                state = self.move_base.get_state()
                if state == GoalStatus.SUCCEEDED:
                    rospy.loginfo("[ClosedLoop] Goal reached on attempt %d", attempt)
                    rospy.sleep(0.5)
                    self.adjust_pose(target_yaw)
                    rospy.sleep(0.5)
                    if is_xy_fix is False:
                        return True

                    # TF 查询当前位姿，做 XY 闭环修正
                    rate = rospy.Rate(20)
                    while not rospy.is_shutdown():
                        try:
                            (current_position, rotation) = self.tf_listener.lookupTransform(
                                "map", "base_link", rospy.Time(0)
                            )
                            rate.sleep()
                            break
                        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                            continue
                    current_position_x = current_position[0]
                    current_position_y = current_position[1]
                    error_x = current_position_x - goal.target_pose.pose.position.x
                    error_y = current_position_y - goal.target_pose.pose.position.y
                    if abs(error_x) > 0.025:
                        self.go_linear_x(-0.1 * abs(error_x) / error_x, abs(error_x))
                        rospy.sleep(0.5)
                    if abs(error_y) > 0.025:
                        y_distance = abs(error_y)
                        if current_position_y > goal.target_pose.pose.position.y:
                            # 当前在目标 Y 上方，左转 90° 后车头朝 +Y，
                            # 需要倒车（负速度）才能减小 Y。
                            self.turn_ang(0.8, math.pi / 2)
                            rospy.sleep(0.5)
                            self.go_linear_x(-0.1, y_distance)
                            rospy.sleep(0.5)
                            self.turn_ang(-0.8, math.pi / 2)
                            rospy.sleep(1)
                        else:
                            # 当前在目标 Y 下方，右转 90° 后车头朝 -Y，
                            # 需要倒车（负速度）才能增大 Y。
                            self.turn_ang(-0.8, math.pi / 2)
                            rospy.sleep(0.5)
                            self.go_linear_x(-0.1, y_distance)
                            rospy.sleep(0.5)
                            self.turn_ang(0.8, math.pi / 2)
                            rospy.sleep(1)
                    self.adjust_pose(target_yaw)
                    rospy.sleep(0.5)
                    return True
                else:
                    rospy.logwarn(
                        "[ClosedLoop] Attempt %d: Goal failed with state: %d", attempt, state
                    )
            else:
                rospy.logwarn(
                    "[ClosedLoop] Attempt %d: Goal timed out after %d seconds", attempt, timeout
                )
            rospy.loginfo("[ClosedLoop] Retrying... (%d/%d)", attempt, retries)

        rospy.logerr("[ClosedLoop] All attempts failed. Unable to reach the goal.")
        return False

    # ------------------------------------------------------------------
    # 2026 年任务流程
    # ------------------------------------------------------------------

    def run_mission(self):
        timeouts = self.config.get("timeouts", {})
        mission = self.config.get("mission", {})

        # 任务一：等待无人机救援目标
        self._set_state(RescueMission.WAIT_DRONE_CMD)
        if self.skip_drone:
            zone = self.rescue_zone or 2
            self.order = RescueOrder(zone=zone, level=1)
            rospy.loginfo("[Mission] 跳过无人机通信，使用指定救援区 zone=%d", self.order.zone)
        else:
            self.order = self.wait_for_rescue_level(
                timeout=timeouts.get("wait_drone_cmd", 120)
            )
            if self.order is None:
                return self._abort("未收到无人机救援目标点（rescue_target.flag）")
            rospy.loginfo("[Mission] 收到救援目标点: zone=%d", self.order.zone)

        # 任务二：取货
        self._set_state(RescueMission.GO_TO_PICKUP_AREA)
        if not self.goto_zone("pickup"):
            return self._abort("无法到达取货区")
        rospy.sleep(1.0)

        self._set_state(RescueMission.VISION_GRASP)
        if not self.vision_grasp():
            return self._abort("视觉抓取失败")

        self._set_state(RescueMission.GO_TO_LOADING_AREA)
        if not self.goto_zone("loading"):
            return self._abort("无法到达装货区")

        # 任务三：装货/通知起飞
        if self.skip_drone:
            self._loading_completed = True
            rospy.loginfo("[Mission] 跳过无人机：装货区保持 3.0s")
            rospy.sleep(3.0)
        else:
            self._set_state(RescueMission.PUT_ON_DRONE)
            if not self.put_on_drone():
                return self._abort("装货失败")

            self._set_state(RescueMission.NOTIFY_TAKEOFF)
            if not self.notify_drone_loading_done():
                return self._abort("未满足装货完成条件，拒绝发送 loading_done")

            if not self.wait_drone_delivery(self.order.zone):
                return self._abort("无人机投送超时")

        # 离开装货区前小退
        rospy.loginfo("[Mission] 前往救援区前，车体后退 0.05m")
        self.go_linear_x(-0.1, 0.05)
        self.stop()
        rospy.sleep(3.0)

        # 任务四：救援区卸货
        self._set_state(RescueMission.GO_TO_RESCUE_AREA)
        if not self.goto_zone("rescue", zone_id=self.order.zone):
            return self._abort("无法到达救援区 %d" % self.order.zone)

        if self.skip_drone:
            self._unload_completed = True
            rospy.loginfo("[Mission] 跳过无人机：救援区保持 3.0s")
            rospy.sleep(3.0)
        else:
            self._set_state(RescueMission.UNLOAD_GOODS)
            if not self.unload_to_target_zone():
                return self._abort("卸货失败")

            self._set_state(RescueMission.NOTIFY_UNLOAD_DONE)
            if not self.notify_drone_unload_done():
                return self._abort("未满足卸货完成条件，拒绝发送 unload_done")

        # 任务五：返航
        self._set_state(RescueMission.GO_TO_HOME_AREA)
        if not self.goto_zone("parking"):
            return self._abort("无法返回停车区")

        self._set_state(RescueMission.FINISH)
        self.stop()
        rospy.loginfo("[Mission] 全部任务完成")
        return True

    def wait_for_rescue_level(self, timeout=120):
        comm = self.config.get("comm", {})
        return wait_for_rescue_target(
            remote_path=comm.get("rescue_target_path", "/mnt/rescue_target.flag"),
            host=comm.get("drone_host", "192.168.31.110"),
            user=comm.get("drone_user", "root"),
            password=comm.get("drone_password", "123456"),
            timeout=timeout,
        )

    def goto_zone(self, zone_name, zone_id=None):
        zones = self.config.get("zones", {})
        if zone_name == "rescue":
            pose = zones["rescue"][zone_id]
        else:
            pose = zones[zone_name]
        pose = as_pose(pose)
        label = "rescue_%s" % zone_id if zone_name == "rescue" else zone_name

        if self.nav_mode == "map_laser" and self.map_nav is not None:
            rospy.loginfo(
                "[Nav] MapLaserNav -> %s (%.3f, %.3f, yaw=%.3f) map+scan 动态避障",
                label,
                pose["x"],
                pose["y"],
                pose["yaw"],
            )
            if not self.map_nav.go_pose(pose, label=label):
                return False
        else:
            rospy.loginfo(
                "[Nav] move_base -> %s (%.3f, %.3f, yaw=%.3f)",
                label,
                pose["x"],
                pose["y"],
                pose["yaw"],
            )
            goal = self.make_move_base_goal(pose["x"], pose["y"], pose["yaw"])
            if not self.move(goal, pose["yaw"], is_xy_fix=True):
                return False

        self._current_zone = zone_name
        self._current_zone_id = zone_id
        self.stop()
        rospy.sleep(0.5)
        return True

    def _init_vision(self):
        if self._vision_initialized or not _VISION_AVAILABLE:
            return
        camera_yaml = self.config.get("vision", {}).get("camera_yaml")
        if camera_yaml:
            camera_yaml = str((_HERE / camera_yaml).resolve())
        self._camera = CameraCapture(self.config, camera_yaml=camera_yaml)
        self._detector = PickupDetector(self.config)
        self._grasp = GraspController(self.config, cmd_vel_pub=self.velocity_publisher)
        self._vision_initialized = True
        rospy.loginfo("[ClosedLoop] vision modules initialized")

    def vision_grasp(self):
        mission = self.config.get("mission", {})
        if mission.get("allow_unimplemented_actions", True):
            rospy.logwarn("[Mission] vision_grasp stub bypass enabled")
            return True
        if not _VISION_AVAILABLE:
            rospy.logerr("[Mission] vision_grasp 需要视觉模块但未导入")
            return False

        self._init_vision()
        grasp_cfg = self.config.get("vision", {}).get("grasp", {})
        timeout = self.config.get("timeouts", {}).get("grasp", 30)
        max_retries = grasp_cfg.get("max_retries", 3)
        deadline = time.time() + timeout

        for attempt in range(1, max_retries + 1):
            if time.time() > deadline:
                break
            frame = self._camera.get_frame()
            if frame is None:
                continue
            target = self._detector.detect(frame)
            if target is None:
                continue
            target = self._grasp.align_to_target(
                target,
                frame.shape,
                get_frame=lambda: self._camera.get_frame(discard=2),
                detect_fn=self._detector.detect,
            )
            if target is None:
                continue
            if not self._grasp.require_hardware and not self._grasp.hardware_ready:
                return True
            if self._grasp.execute(target):
                return True
        return False

    def put_on_drone(self):
        self._loading_completed = False
        if self._current_zone != "loading":
            rospy.logerr("[Mission] put_on_drone 拒绝：当前不在装货区")
            return False
        mission = self.config.get("mission", {})
        hold_sec = float(mission.get("loading_hold_sec", 10.0))
        if hold_sec > 0:
            rospy.loginfo("[Mission] 装货保持 %.1fs", hold_sec)
            rospy.sleep(hold_sec)
        if mission.get("wait_operator_confirm", False):
            try:
                input("按回车确认装货完成...")
            except Exception as exc:
                rospy.logerr("[Mission] 人工确认失败: %s", exc)
                return False
        self._loading_completed = True
        rospy.loginfo("[Mission] 装货完成")
        return True

    def notify_drone_loading_done(self):
        if self._current_zone != "loading" or not self._loading_completed:
            rospy.logerr("[Comm] 拒绝发送 loading_done")
            return False
        comm = self.config.get("comm", {})
        rospy.loginfo("[Comm] 发送 loading_done.flag")
        notify_loading_done(
            remote_path=comm.get("loading_done_path", "/mnt/loading_done.flag"),
            host=comm.get("drone_host", "192.168.31.110"),
            user=comm.get("drone_user", "root"),
            password=comm.get("drone_password", "123456"),
        )
        return True

    def wait_drone_delivery(self, zone_id):
        comm = self.config.get("comm", {})
        return wait_for_delivery_done(
            remote_path=comm.get("delivery_done_path", "/mnt/delivery_done.flag"),
            host=comm.get("drone_host", "192.168.31.110"),
            user=comm.get("drone_user", "root"),
            password=comm.get("drone_password", "123456"),
            timeout=self.config.get("timeouts", {}).get("delivery", 300),
        )

    def unload_to_target_zone(self):
        self._unload_completed = False
        if self._current_zone != "rescue":
            rospy.logerr("[Mission] unload 拒绝：当前不在救援区")
            return False
        mission = self.config.get("mission", {})
        hold_sec = float(mission.get("unload_hold_sec", 10.0))
        if hold_sec > 0:
            rospy.loginfo("[Mission] 卸货保持 %.1fs", hold_sec)
            rospy.sleep(hold_sec)
        if mission.get("wait_operator_confirm", False):
            try:
                input("按回车确认卸货完成...")
            except Exception as exc:
                rospy.logerr("[Mission] 人工确认失败: %s", exc)
                return False
        self._unload_completed = True
        rospy.loginfo("[Mission] 卸货完成")
        return True

    def notify_drone_unload_done(self):
        if self._current_zone != "rescue" or not self._unload_completed:
            rospy.logerr("[Comm] 拒绝发送 unload_done")
            return False
        comm = self.config.get("comm", {})
        rospy.loginfo("[Comm] 发送 unload_done.flag")
        notify_unload_done(
            remote_path=comm.get("unload_done_path", "/mnt/unload_done.flag"),
            host=comm.get("drone_host", "192.168.31.110"),
            user=comm.get("drone_user", "root"),
            password=comm.get("drone_password", "123456"),
        )
        return True


def main():
    parser = argparse.ArgumentParser(
        description="nav_rescue_2026_closed_loop：2026 任务 + MapLaserNav 动态避障（可选 move_base）"
    )
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="只初始化节点，不自动跑任务",
    )
    parser.add_argument(
        "--skip-drone",
        action="store_true",
        help="跳过无人机通信，直接测试小车位置参数",
    )
    parser.add_argument(
        "--rescue",
        type=int,
        default=2,
        choices=(1, 2, 3, 4),
        help="跳过无人机时使用的救援区编号（默认 2）",
    )
    parser.add_argument(
        "--cmd-vel-topic",
        type=str,
        default="/cmd_vel",
        help="底盘速度话题，默认 /cmd_vel",
    )
    parser.add_argument(
        "--nav-mode",
        type=str,
        default="map_laser",
        choices=("map_laser", "move_base"),
        help="导航模式：map_laser=测试脚本同款动态避障（默认）；move_base=旧 TEB 栈",
    )
    args = parser.parse_args()
    NavRescue2026ClosedLoop(
        autostart=not args.no_autostart,
        skip_drone=args.skip_drone,
        rescue_zone=args.rescue,
        cmd_vel_topic=args.cmd_vel_topic,
        nav_mode=args.nav_mode,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
