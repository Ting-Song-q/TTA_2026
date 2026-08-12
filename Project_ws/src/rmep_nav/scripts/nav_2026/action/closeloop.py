#!/usr/bin/python3
# coding=UTF-8
"""closeloop：2026 救援任务流程 + YaLongR8 风格 move_base/TEB 闭环导航。

任务流程与 nav_rescue_2026_closed_loop.py 一致：
  等待无人机目标 → 取货区 → 视觉抓取 → 装货区 → 装货/通知起飞 →
  等待无人机投送 → 救援区 → 卸货/通知完成 → 返航停车区

闭环控制（对齐 YaLongR8 multi_goals_navigation + run_navigation）：
  - 定位反馈：AMCL + odom（由外部 launch 提供）
  - 规划/跟踪：move_base + TEB
  - 每个目标前 clear_costmaps
  - 仅用 ActionClient 发目标并 wait_for_result（勿兼发 /move_base_simple/goal，会抢占取消）
  - 等待 /move_base result：SUCCEEDED 成功；ABORTED 记失败（任务层不继续）
  - 无 TF/开环精修、无走廊 keep_yaw、无漂移 /initialpose reseed

依赖启动栈：
  1) roscore
  2) roslaunch rmep_base rmep_base.launch
  3) roslaunch rmep_nav map_amcl_move.launch
  4) python3 action/closeloop.py [--skip-drone --rescue 2]
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
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_srvs.srv import Empty as EmptySrv
from tf.transformations import quaternion_from_euler, euler_from_quaternion

_HERE = Path(__file__).resolve().parent
_NAV_DIR = _HERE.parent
if str(_NAV_DIR) not in sys.path:
    sys.path.insert(0, str(_NAV_DIR))

from rescue_protocol import (  # noqa: E402
    RescueOrder,
    wait_for_rescue_target,
    notify_loading_done,
    wait_for_delivery_done,
    notify_unload_done,
)
from config_loader import load_mission_config, as_pose  # noqa: E402

_VISION_AVAILABLE = False
try:
    from vision.camera_capture import CameraCapture  # noqa: E402
    from vision.pickup_detector import PickupDetector  # noqa: E402
    from vision.grasp_controller import GraspController  # noqa: E402

    _VISION_AVAILABLE = True
except Exception as _vision_err:  # pragma: no cover
    rospy.logwarn(
        "[CloseLoop] 视觉模块未导入: %s; vision_grasp 将使用占位实现", _vision_err
    )


class RescueMission(object):
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


class MoveBaseClosedLoopNav(object):
    """YaLong 风格：clear_costmaps → 发目标 → 等 move_base result。"""

    def __init__(
        self,
        cmd_vel_topic="/cmd_vel",
        move_base_name="move_base",
        frame_id="map",
        clear_costmaps=True,
        server_wait_timeout=60.0,
        log_prefix="[YaLongCL]",
    ):
        self._log = log_prefix
        self.frame_id = str(frame_id or "map")
        self.clear_costmaps_enabled = bool(clear_costmaps)
        self.cmd_vel_topic = str(cmd_vel_topic or "/cmd_vel")

        self.tf_listener = tf.TransformListener()
        self.velocity_publisher = rospy.Publisher(
            self.cmd_vel_topic, Twist, queue_size=10
        )
        # 可视化用独立话题；不要发 /move_base_simple/goal（会与 ActionClient 互抢占）
        self.goal_pub = rospy.Publisher(
            "/closeloop/goal_viz", PoseStamped, queue_size=1
        )

        self.clear_costmap = None
        service_name = "/%s/clear_costmaps" % move_base_name.strip("/")
        if self.clear_costmaps_enabled:
            try:
                rospy.wait_for_service(service_name, timeout=10.0)
                self.clear_costmap = rospy.ServiceProxy(service_name, EmptySrv)
                rospy.loginfo("%s clear_costmaps ready: %s", self._log, service_name)
            except rospy.ROSException:
                rospy.logwarn(
                    "%s 未找到 %s，将跳过清图（仍可导航）", self._log, service_name
                )

        rospy.loginfo("%s 等待 move_base action server: %s", self._log, move_base_name)
        self.move_base = actionlib.SimpleActionClient(move_base_name, MoveBaseAction)
        deadline = rospy.Time.now() + rospy.Duration(float(server_wait_timeout))
        while not self.move_base.wait_for_server(rospy.Duration(1.0)):
            if rospy.is_shutdown() or rospy.Time.now() > deadline:
                raise RuntimeError(
                    "move_base action server 未就绪: %s" % move_base_name
                )
        rospy.loginfo(
            "%s move_base ready (YaLong-style: clear → goal → wait result)",
            self._log,
        )

    def stop(self):
        """停车并取消 move_base 目标。"""
        try:
            self.move_base.cancel_all_goals()
        except Exception:
            pass
        vel_msg = Twist()
        self.velocity_publisher.publish(vel_msg)

    def clear_maps(self):
        if self.clear_costmap is None:
            return False
        try:
            self.clear_costmap()
            rospy.loginfo("%s costmaps cleared", self._log)
            return True
        except rospy.ServiceException as exc:
            rospy.logwarn("%s clear_costmaps 失败: %s", self._log, exc)
            return False

    def make_move_base_goal(self, x, y, yaw, frame_id=None):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = frame_id or self.frame_id
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = float(x)
        goal.target_pose.pose.position.y = float(y)
        goal.target_pose.pose.position.z = 0.0
        q = quaternion_from_euler(0.0, 0.0, float(yaw))
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        return goal

    def _publish_simple_goal(self, goal):
        pose = PoseStamped()
        pose.header = goal.target_pose.header
        pose.pose = goal.target_pose.pose
        self.goal_pub.publish(pose)

    def lookup_base_pose(self, rate_hz=20, max_wait=2.0):
        """读取 map→base_link，返回 (x, y, yaw) 或 None。"""
        rate = rospy.Rate(rate_hz)
        deadline = rospy.Time.now() + rospy.Duration(float(max_wait))
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            try:
                (trans, rot) = self.tf_listener.lookupTransform(
                    "map", "base_link", rospy.Time(0)
                )
                _, _, yaw = euler_from_quaternion(rot)
                return float(trans[0]), float(trans[1]), float(yaw)
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ):
                rate.sleep()
        return None

    def move(self, goal, timeout=90.0, clear=True):
        """YaLong 核心闭环：可选清图 → 发目标 → 等 SUCCEEDED。

        返回 True 仅当 GoalStatus.SUCCEEDED；ABORTED/超时为 False。
        （任务层不把 ABORT 当成功；与 YaLong 多点脚本“ABORT 也进下一点”不同。）
        """
        if clear and self.clear_costmaps_enabled:
            rospy.loginfo("%s Clearing costmaps before goal", self._log)
            self.clear_maps()
            rospy.sleep(0.3)

        goal.target_pose.header.stamp = rospy.Time.now()
        # 只走 ActionClient；viz 发到非 move_base 输入话题，避免 simple goal 抢占取消
        self.move_base.send_goal(goal)
        self._publish_simple_goal(goal)
        rospy.loginfo(
            "%s Navigating to (%.3f, %.3f)",
            self._log,
            goal.target_pose.pose.position.x,
            goal.target_pose.pose.position.y,
        )

        finished = self.move_base.wait_for_result(rospy.Duration(float(timeout)))
        if not finished:
            rospy.logwarn("%s Goal timed out after %.1fs", self._log, timeout)
            try:
                self.move_base.cancel_goal()
            except Exception:
                pass
            self.stop()
            return False

        state = self.move_base.get_state()
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo("%s Goal reached!", self._log)
            self.stop()
            return True

        text = ""
        try:
            text = self.move_base.get_goal_status_text()
        except Exception:
            pass
        if state == GoalStatus.ABORTED:
            rospy.logwarn("%s Goal aborted: %s", self._log, text or state)
        else:
            rospy.logwarn("%s Goal finished with state=%s %s", self._log, state, text)
        self.stop()
        return False

    def go_pose(self, pose, label="goal", timeout=90.0, clear=True):
        """单点导航：直接发 map 系位姿，交给 TEB 容差判定到位。"""
        pose = as_pose(pose)
        goal_x = float(pose["x"])
        goal_y = float(pose["y"])
        target_yaw = float(pose["yaw"])

        cur = self.lookup_base_pose()
        if cur is not None:
            dist = math.hypot(goal_x - cur[0], goal_y - cur[1])
            timeout = max(float(timeout), dist / 0.25 + 45.0)
        else:
            timeout = float(timeout)

        goal = self.make_move_base_goal(goal_x, goal_y, target_yaw)
        rospy.loginfo(
            "%s -> %s (%.3f, %.3f, yaw=%.3f) timeout=%.1fs clear=%s",
            self._log,
            label,
            goal_x,
            goal_y,
            target_yaw,
            timeout,
            clear,
        )
        return self.move(goal, timeout=timeout, clear=clear)


class NavRescue2026YaLongCloseLoop(object):
    """任务状态机 + YaLong 风格 move_base/TEB 闭环。"""

    def __init__(
        self,
        autostart=True,
        skip_drone=False,
        rescue_zone=None,
        cmd_vel_topic="/cmd_vel",
        config_path=None,
        move_base_name="move_base",
        clear_costmaps=True,
    ):
        rospy.init_node("nav_rescue_2026_yalong_closeloop", anonymous=True)

        self.skip_drone = bool(skip_drone)
        self.rescue_zone = int(rescue_zone) if rescue_zone is not None else None
        self.cmd_vel_topic = str(cmd_vel_topic or "/cmd_vel")
        self.config = self._load_config(config_path)
        self.state = RescueMission.WAIT_DRONE_CMD
        self.order = None

        self._loading_completed = False
        self._unload_completed = False
        self._current_zone = None
        self._current_zone_id = None

        self._camera = None
        self._detector = None
        self._grasp = None
        self._vision_initialized = False

        parking = (self.config.get("zones") or {}).get("parking") or {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
        }
        self._parking_pose = (
            float(parking.get("x", 0.0)),
            float(parking.get("y", 0.0)),
            float(parking.get("yaw", 0.0)),
        )
        # EP/AMCL 需要初值；YaLong 实车常靠 launch 初值，这里发一次停车区位姿
        self.publish_initial_pose(
            self._parking_pose[0],
            self._parking_pose[1],
            self._parking_pose[2],
        )

        self.nav = MoveBaseClosedLoopNav(
            cmd_vel_topic=self.cmd_vel_topic,
            move_base_name=move_base_name,
            frame_id="map",
            clear_costmaps=clear_costmaps,
            log_prefix="[YaLongCL]",
        )
        self.velocity_publisher = self.nav.velocity_publisher
        self._start_pose_monitor(period_sec=1.0)

        self._log_zones()
        rospy.loginfo(
            "[CloseLoop] ready: mission=nav_rescue_2026, "
            "control=YaLong clear→move_base/TEB→result"
        )

        if autostart:
            self.run_mission()

    def _start_pose_monitor(self, period_sec=1.0):
        """每 period_sec 秒在终端打印 map→base_link 位姿。"""
        self._pose_timer = rospy.Timer(
            rospy.Duration(float(period_sec)), self._on_pose_timer
        )
        rospy.loginfo(
            "[CloseLoop] pose monitor: every %.1fs (map→base_link)", period_sec
        )

    def _on_pose_timer(self, _event):
        pose = self.nav.lookup_base_pose(rate_hz=50, max_wait=0.15)
        if pose is None:
            rospy.logwarn_throttle(5.0, "[Pose] map→base_link 暂不可用")
            return
        x, y, yaw = pose
        rospy.loginfo(
            "[Pose] x=%.3f y=%.3f yaw=%.3f deg=%.1f",
            x,
            y,
            yaw,
            math.degrees(yaw),
        )

    @staticmethod
    def _load_config(config_path=None):
        if config_path:
            path = Path(config_path)
        else:
            path = _NAV_DIR / "mission_config.yaml"
        return load_mission_config(path)

    def _log_zones(self):
        zones = self.config.get("zones") or {}
        for name in ("parking", "pickup", "loading"):
            pose = zones.get(name) or {}
            rospy.loginfo(
                "[CloseLoop] zones.%s = (%.3f, %.3f, yaw=%.3f)",
                name,
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("yaw", 0.0)),
            )
        for zid in sorted((zones.get("rescue") or {}).keys(), key=lambda x: int(x)):
            pose = zones["rescue"][zid]
            rospy.loginfo(
                "[CloseLoop] zones.rescue_%s = (%.3f, %.3f, yaw=%.3f)",
                zid,
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("yaw", 0.0)),
            )

    def publish_initial_pose(self, x, y, yaw, hold_sec=0.8):
        pub = rospy.Publisher(
            "/initialpose", PoseWithCovarianceStamped, queue_size=10
        )
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        q = quaternion_from_euler(0.0, 0.0, float(yaw))
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        msg.pose.covariance = [
            0.25, 0, 0, 0, 0, 0,
            0, 0.25, 0, 0, 0, 0,
            0, 0, 0.25, 0, 0, 0,
            0, 0, 0, 0.068, 0, 0,
            0, 0, 0, 0, 0.068, 0,
            0, 0, 0, 0, 0, 0.068,
        ]
        rate = rospy.Rate(20)
        start = rospy.Time.now()
        while (rospy.Time.now() - start).to_sec() < float(hold_sec):
            msg.header.stamp = rospy.Time.now()
            pub.publish(msg)
            rate.sleep()
        rospy.sleep(1.0)
        rospy.loginfo(
            "[CloseLoop] /initialpose published (%.3f, %.3f, yaw=%.3f)", x, y, yaw
        )

    def _set_state(self, state):
        self.state = state
        rospy.loginfo("[Mission] %s", RescueMission.name(state))

    def _abort(self, reason):
        rospy.logerr("[Mission] abort: %s", reason)
        self.stop()
        return False

    def stop(self):
        if self.nav is not None:
            self.nav.stop()

    def run_mission(self):
        timeouts = self.config.get("timeouts", {})

        self._set_state(RescueMission.WAIT_DRONE_CMD)
        if self.skip_drone:
            zone = self.rescue_zone or 2
            self.order = RescueOrder(zone=zone, level=1)
            rospy.loginfo(
                "[Mission] 跳过无人机通信，使用指定救援区 zone=%d", self.order.zone
            )
        else:
            self.order = self.wait_for_rescue_level(
                timeout=timeouts.get("wait_drone_cmd", 120)
            )
            if self.order is None:
                return self._abort("未收到无人机救援目标点（rescue_target.flag）")
            rospy.loginfo("[Mission] 收到救援目标点: zone=%d", self.order.zone)

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
        """YaLong 风格到点：无 forward/reverse_keep_yaw、无 twophase。"""
        zones = self.config.get("zones", {})
        if zone_name == "rescue":
            pose = zones["rescue"][zone_id]
        else:
            pose = zones[zone_name]
        pose = as_pose(pose)
        label = "rescue_%s" % zone_id if zone_name == "rescue" else zone_name
        base_timeout = float((self.config.get("timeouts") or {}).get("nav_goal", 90))

        rospy.loginfo(
            "[Nav] YaLong move_base/TEB -> %s (%.3f, %.3f, yaw=%.3f)",
            label,
            pose["x"],
            pose["y"],
            pose["yaw"],
        )
        if not self.nav.go_pose(pose, label=label, timeout=base_timeout, clear=True):
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
            camera_yaml = str((_NAV_DIR / camera_yaml).resolve())
        self._camera = CameraCapture(self.config, camera_yaml=camera_yaml)
        self._detector = PickupDetector(self.config)
        self._grasp = GraspController(self.config, cmd_vel_pub=self.velocity_publisher)
        self._vision_initialized = True
        rospy.loginfo("[CloseLoop] vision modules initialized")

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

        for _attempt in range(1, max_retries + 1):
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
        description=(
            "action/closeloop：2026 任务 + YaLong 风格 move_base/TEB 闭环"
        )
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
        "--move-base-name",
        type=str,
        default="move_base",
        help="move_base action 名称，默认 move_base",
    )
    parser.add_argument(
        "--no-clear-costmaps",
        action="store_true",
        help="关闭发目标前 clear_costmaps（YaLong 默认开启）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="mission_config.yaml 路径（默认 nav_2026/mission_config.yaml）",
    )
    args = parser.parse_args()

    NavRescue2026YaLongCloseLoop(
        autostart=not args.no_autostart,
        skip_drone=args.skip_drone,
        rescue_zone=args.rescue,
        cmd_vel_topic=args.cmd_vel_topic,
        config_path=args.config or None,
        move_base_name=args.move_base_name,
        clear_costmaps=not args.no_clear_costmaps,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
