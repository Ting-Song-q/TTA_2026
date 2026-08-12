#!/usr/bin/python3
# coding=UTF-8

import sys
import time

import rospy
import yaml
import math
import tf
import actionlib
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from vision.camera_capture import CameraCapture
from vision.pickup_detector import PickupDetector
from vision.grasp_controller import GraspController

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from rescue_protocol import (
    RescueOrder,
    wait_for_rescue_cmd,
    notify_loading_done,
    wait_for_delivery_done,
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
    GO_TO_HOME_AREA = 8
    FINISH = 9

    _NAMES = {
        0: "WAIT_DRONE_CMD",
        1: "GO_TO_PICKUP_AREA",
        2: "VISION_GRASP",
        3: "GO_TO_LOADING_AREA",
        4: "PUT_ON_DRONE",
        5: "NOTIFY_TAKEOFF",
        6: "GO_TO_RESCUE_AREA",
        7: "UNLOAD_GOODS",
        8: "GO_TO_HOME_AREA",
        9: "FINISH",
    }

    @classmethod
    def name(cls, state):
        return cls._NAMES.get(state, str(state))


class Nav:
    def __init__(self):
        rospy.init_node("nav_rescue_2026", anonymous=True)

        self.config = self._load_config()
        self.state = RescueMission.WAIT_DRONE_CMD
        self.order = None

        self.velocity_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.laser_subscriber = rospy.Subscriber(
            "/scan", LaserScan, self.laser_callback, queue_size=1
        )
        self.laser_data = None
        self._camera = None
        self._detector = None
        self._grasp = None

        self.rate = rospy.Rate(100)
        self.tf_listener = tf.TransformListener()
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        while not self.move_base.wait_for_server(rospy.Duration(1.0)):
            continue

        rospy.loginfo("nav_rescue_2026 ready")
        self.run_mission()

    def _load_config(self):
        config_path = Path(__file__).resolve().parent / "mission_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _set_state(self, state):
        self.state = state
        rospy.loginfo("[Mission] %s", RescueMission.name(state))

    def _abort(self, reason):
        rospy.logerr("[Mission] abort: %s", reason)
        self.stop()
        return False

    def run_mission(self):
        """
        主状态机：严格按 任务二 → 三 → 四 → 五 顺序执行。
        任务一（无人机巡检）在 wait_for_rescue_level 中等待其完成。
        """
        timeouts = self.config.get("timeouts", {})

        # --- 任务一：等无人机发救援指令 ---
        self._set_state(RescueMission.WAIT_DRONE_CMD)
        self.order = self.wait_for_rescue_level(
            timeout=timeouts.get("wait_drone_cmd", 120)
        )
        if self.order is None:
            return self._abort("未收到无人机救援指令")

        rospy.loginfo(
            "收到救援指令: zone=%d level=%d",
            self.order.zone,
            self.order.level,
        )

        # --- 任务二：取货 ---
        self._set_state(RescueMission.GO_TO_PICKUP_AREA)
        if not self.goto_zone("pickup"):
            return self._abort("无法到达取货区")
        if not self.assert_wheels_in_zone("pickup"):
            return self._abort("取货区驱动轮未到位")

        self._set_state(RescueMission.VISION_GRASP)
        if not self.vision_grasp():
            return self._abort("视觉抓取失败")

        self._set_state(RescueMission.GO_TO_LOADING_AREA)
        if not self.goto_zone("loading"):
            return self._abort("无法到达装货区")
        if not self.assert_wheels_in_zone("loading"):
            return self._abort("装货区驱动轮未到位")

        # --- 任务三：装到无人机 ---
        self._set_state(RescueMission.PUT_ON_DRONE)
        if not self.put_on_drone():
            return self._abort("装货失败")

        self._set_state(RescueMission.NOTIFY_TAKEOFF)
        self.notify_drone_loading_done()

        # --- 任务四：无人机投送 + 小车卸货 ---
        if not self.wait_drone_delivery(self.order.zone):
            return self._abort("无人机投送超时")

        self._set_state(RescueMission.GO_TO_RESCUE_AREA)
        if not self.goto_zone("rescue", zone_id=self.order.zone):
            return self._abort(f"无法到达救援区 {self.order.zone}")
        if not self.assert_wheels_in_zone("rescue", zone_id=self.order.zone):
            return self._abort("救援区驱动轮未到位")

        self._set_state(RescueMission.UNLOAD_GOODS)
        if not self.unload_to_target_zone():
            return self._abort("卸货失败")

        # --- 任务五：返航 ---
        self._set_state(RescueMission.GO_TO_HOME_AREA)
        if not self.goto_zone("parking"):
            return self._abort("无法返回停车区")

        self._set_state(RescueMission.FINISH)
        self.stop()
        rospy.loginfo("[Mission] 全部任务完成")
        return True

    # ---------- 各步骤实现（先占位，后续逐个填） ----------

    def wait_for_rescue_level(self, timeout=120):
        comm = self.config.get("comm", {})
        return wait_for_rescue_cmd(
            remote_path=comm.get("rescue_cmd_path", "/mnt/rescue_cmd.csv"),
            host=comm.get("drone_host", "192.168.31.110"),
            user=comm.get("drone_user", "root"),
            password=comm.get("drone_password", "123456"),
            timeout=timeout,
        )

    def goto_zone(self, zone_name, zone_id=None):
        zones = self.config["zones"]
        if zone_name == "rescue":
            pose = zones["rescue"][zone_id]
        else:
            pose = zones[zone_name]

        goal = self.make_move_base_goal(pose["x"], pose["y"], pose["yaw"])
        timeout = self.config.get("timeouts", {}).get("nav_goal", 60)
        return self.move(goal, pose["yaw"], True, timeout=timeout)

    def assert_wheels_in_zone(self, zone_name, zone_id=None):
        # TODO: 用 TF 判断至少 2 个驱动轮在目标区域内
        rospy.loginfo("assert_wheels_in_zone(%s) stub -> True", zone_name)
        return True

    def _init_vision(self):
        if self._camera is not None:
            return
        camera_yaml = self.config.get("vision", {}).get("camera_yaml")
        if camera_yaml:
            camera_yaml = str((_SCRIPT_DIR / camera_yaml).resolve())
        self._camera = CameraCapture(self.config, camera_yaml=camera_yaml)
        self._detector = PickupDetector(self.config)
        self._grasp = GraspController(self.config, cmd_vel_pub=self.velocity_publisher)
        rospy.loginfo("vision modules initialized")

    def _save_vision_debug(self, frame, name, target=None):
        if not self.config.get("vision", {}).get("save_debug", True):
            return
        debug_dir = self.config.get("vision", {}).get("debug_dir", "/tmp/rescue_vision")
        vis = self._detector.draw_debug(frame, target)
        self._camera.save_debug(debug_dir, name, vis)

    def vision_grasp(self):
        self._init_vision()

        vision_cfg = self.config.get("vision", {})
        grasp_cfg = vision_cfg.get("grasp", {})
        timeout = self.config.get("timeouts", {}).get("grasp", 30)
        max_retries = grasp_cfg.get("max_retries", 3)
        verify = grasp_cfg.get("verify_after_grasp", True)
        deadline = time.time() + timeout

        for attempt in range(1, max_retries + 1):
            if time.time() > deadline:
                break

            frame = self._camera.get_frame()
            if frame is None:
                rospy.logwarn("vision_grasp attempt %d: no frame", attempt)
                continue

            target = self._detector.detect(frame)
            if target is None:
                rospy.logwarn("vision_grasp attempt %d: no target", attempt)
                self._save_vision_debug(frame, f"grasp_fail_{attempt}.jpg")
                continue

            rospy.loginfo(
                "detected cell=%d (row=%d,col=%d) conf=%.2f",
                target.cell_index,
                target.row,
                target.col,
                target.confidence,
            )
            self._save_vision_debug(frame, f"grasp_detect_{attempt}.jpg", target)

            target = self._grasp.align_to_target(
                target,
                frame.shape,
                get_frame=lambda: self._camera.get_frame(discard=2),
                detect_fn=self._detector.detect,
            )
            if target is None:
                continue

            if not self._grasp.require_hardware and not self._grasp.hardware_ready:
                rospy.loginfo("vision_grasp detect-only success")
                return True

            if not self._grasp.execute(target):
                rospy.logwarn("vision_grasp attempt %d: execute failed", attempt)
                continue

            if verify:
                verify_frame = self._camera.get_frame(discard=3)
                if verify_frame is not None:
                    self._save_vision_debug(
                        verify_frame, f"grasp_verify_{attempt}.jpg"
                    )
                    leftover = self._detector.detect(verify_frame)
                    if leftover is None or leftover.cell_index != target.cell_index:
                        rospy.loginfo("vision_grasp success (verified)")
                        return True
                    rospy.logwarn("vision_grasp attempt %d: verify failed", attempt)
                    continue

            rospy.loginfo("vision_grasp success")
            return True

        rospy.logerr("vision_grasp failed after %d attempts", max_retries)
        return False

    def put_on_drone(self):
        # TODO: 机械臂/夹爪把物资放到无人机平台
        rospy.loginfo("put_on_drone stub -> True")
        return True

    def notify_drone_loading_done(self):
        comm = self.config.get("comm", {})
        notify_loading_done(
            remote_path=comm.get("loading_done_path", "/mnt/loading_done.flag"),
            host=comm.get("drone_host", "192.168.31.110"),
            user=comm.get("drone_user", "root"),
            password=comm.get("drone_password", "123456"),
        )

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
        # TODO: 从无人机平台取货，放入 100x100mm 放置区
        rospy.loginfo("unload_to_target_zone stub -> True")
        return True

    # ---------- 导航底层（可从 nav_2025.py 迁移） ----------

    def laser_callback(self, msg):
        self.laser_data = msg

    def stop(self):
        vel = Twist()
        self.velocity_publisher.publish(vel)

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

    def move(self, goal, target_yaw, is_xy_fix, timeout=60, retries=3):
        for attempt in range(1, retries + 1):
            self.move_base.send_goal(goal)
            finished = self.move_base.wait_for_result(rospy.Duration(timeout))
            if finished and self.move_base.get_state() == GoalStatus.SUCCEEDED:
                self.adjust_pose(target_yaw)
                return True
            rospy.logwarn("move attempt %d/%d failed", attempt, retries)
        return False

    def adjust_pose(self, target_yaw):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            try:
                _, rotation = self.tf_listener.lookupTransform(
                    "map", "base_link", rospy.Time(0)
                )
                break
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ):
                rate.sleep()
                continue

        _, _, current_yaw = euler_from_quaternion(rotation)
        error = math.atan2(
            math.sin(current_yaw - target_yaw),
            math.cos(current_yaw - target_yaw),
        )
        if abs(error) > 0.025:
            ang_speed = 0.6 * (1 if error > 0 else -1)
            self.turn_ang(ang_speed, abs(error))

    def turn_ang(self, ang_speed, goal_rotation):
        twist = Twist()
        twist.angular.z = -ang_speed if goal_rotation > 0 else ang_speed
        duration = abs(goal_rotation / ang_speed)
        start = rospy.Time.now()
        while (rospy.Time.now() - start).to_sec() < duration:
            self.velocity_publisher.publish(twist)
            self.rate.sleep()
        self.stop()


if __name__ == "__main__":
    nav = Nav()
    rospy.spin()
