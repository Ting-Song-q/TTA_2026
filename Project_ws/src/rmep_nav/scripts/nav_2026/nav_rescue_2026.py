#!/usr/bin/python3
# coding=UTF-8

import sys
import time

import rospy
import math
import tf
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from vision.camera_capture import CameraCapture
from vision.pickup_detector import PickupDetector
from vision.grasp_controller import GraspController
from vision.zone_boundary_detector import ZoneBoundaryDetector


from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion

from rescue_protocol import (
    RescueOrder,
    wait_for_rescue_target,
    notify_loading_done,
    wait_for_delivery_done,
    notify_unload_done,
)
from laser_avoidance import (
    guard_twist,
    is_emergency,
)

from zone_guard import (
    clamp_vector,
    count_points_in_polygon,
    map_vector_to_base,
    polygon_center,
    rectangle_polygon,
    resolve_zone_polygon,
    safe_pose_error,
    transform_points,
    wheel_outer_points_base,
)
from config_loader import calibration_has_zone, load_mission_config
from zone_health import covariance_std, evaluate_localization_health
from openloop_duo import LaserOpenLoopNav, build_zone_whitelist, normalize_yaw



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


class Nav:
    def __init__(
        self,
        autostart=True,
        wait_for_move_base=None,
        skip_drone=False,
        rescue_zone=None,
        cmd_vel_topic="/cmd_vel",
    ):
        # wait_for_move_base 仅为兼容旧调用；运动唯一实现为开环+白名单避障
        del wait_for_move_base
        rospy.init_node("nav_rescue_2026", anonymous=True)

        self.skip_drone = bool(skip_drone)
        self.rescue_zone = int(rescue_zone) if rescue_zone is not None else None
        self.cmd_vel_topic = str(cmd_vel_topic) if cmd_vel_topic else "/cmd_vel"

        self.config = self._load_config()
        self.state = RescueMission.WAIT_DRONE_CMD
        self.order = None

        car_cfg = dict(self.config.get("car") or {})
        self.speed = float(car_cfg.get("speed", 0.3))
        self.turn_speed = float(car_cfg.get("turn_speed", 0.5))
        self.align_speed = float(car_cfg.get("align_speed", 0.5))
        if self.turn_speed < 0.5:
            rospy.logwarn(
                "[Nav] car.turn_speed=%.2f 过低，可能导致底盘无法有效旋转；"
                "建议改为 0.5 或更高",
                self.turn_speed,
            )

        self.zone_boxes, duo_avoid = build_zone_whitelist(self.config)
        merged_avoid = dict(duo_avoid)
        merged_avoid.update(self.config.get("obstacle_avoidance") or {})
        self.config["obstacle_avoidance"] = merged_avoid

        parking = (self.config.get("zones") or {}).get("parking") or {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
        }

        self._last_scan_stamp = None
        self._amcl_pose = None
        self._last_amcl_stamp = None
        self._camera = None
        self._detector = None
        self._grasp = None
        self._zone_detector = None
        self._last_zone_check = None
        self._loading_completed = False
        self._unload_completed = False
        self._current_zone = None
        self._current_zone_id = None

        self.rate = rospy.Rate(100)
        self.tf_listener = tf.TransformListener()
        self.amcl_subscriber = rospy.Subscriber(
            "/amcl_pose",
            PoseWithCovarianceStamped,
            self.amcl_pose_callback,
            queue_size=1,
        )

        # 唯一底盘运动：与 run_car_duo 相同的 LaserOpenLoopNav
        self.ol = LaserOpenLoopNav(
            speed=self.speed,
            turn_speed=self.turn_speed,
            align_speed=self.align_speed,
            avoidance_cfg=merged_avoid,
            zone_boxes=self.zone_boxes,
            log_prefix="[Nav]",
            node_name="nav_rescue_2026",
            wait_laser=True,
            on_laser=self._on_laser,
            cmd_vel_topic=self.cmd_vel_topic,
            odom_topic="/odom",
        )
        self.velocity_publisher = self.ol.velocity_publisher
        rospy.sleep(0.5)
        self._check_cmd_vel_conflicts()
        self.ol.set_est_pose(
            {
                "x": float(parking.get("x", 0.0)),
                "y": float(parking.get("y", 0.0)),
                "yaw": float(parking.get("yaw", 0.0)),
            }
        )

        rospy.loginfo(
            "nav_rescue_2026 ready (open_loop + zone whitelist only, cmd_vel=%s)",
            self.cmd_vel_topic,
        )
        cal = self.config.get("_calibration") or {}
        zones = self.config.get("zones") or {}
        rospy.loginfo(
            "[Nav] 航点来源: %s (calibration applied=%s path=%s)",
            "zone_calibration.yaml" if cal.get("applied") else "mission_config.yaml zones",
            cal.get("applied", False),
            cal.get("path"),
        )
        for name in ("parking", "pickup", "loading"):
            pose = zones.get(name) or {}
            rospy.loginfo(
                "[Nav]  zones.%s = (%.3f, %.3f, yaw=%.3f)",
                name,
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("yaw", 0.0)),
            )
        for zid in sorted((zones.get("rescue") or {}).keys(), key=lambda x: int(x)):
            pose = zones["rescue"][zid]
            rospy.loginfo(
                "[Nav]  zones.rescue_%s = (%.3f, %.3f, yaw=%.3f)",
                zid,
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("yaw", 0.0)),
            )
        if autostart:
            self.run_mission()

    @property
    def laser_data(self):
        return self.ol.laser_data

    @property
    def _est_pose(self):
        return self.ol.get_est_pose()

    @_est_pose.setter
    def _est_pose(self, pose):
        self.ol.set_est_pose(pose)

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

    def run_mission(self):
        timeouts = self.config.get("timeouts", {})

        # --- 任务一：等无人机写最终目标点 flag ---
        self._set_state(RescueMission.WAIT_DRONE_CMD)
        if self.skip_drone:
            zone = self.rescue_zone or 2
            self.order = RescueOrder(zone=zone, level=1)
            rospy.loginfo(
                "[Mission] 跳过无人机通信，使用指定救援区 zone=%d",
                self.order.zone,
            )
        else:
            self.order = self.wait_for_rescue_level(
                timeout=timeouts.get("wait_drone_cmd", 120)
            )
            if self.order is None:
                return self._abort("未收到无人机救援目标点（rescue_target.flag）")
            rospy.loginfo(
                "收到救援目标点: zone=%d（来自 rescue_target.flag）",
                self.order.zone,
            )

        # --- 任务二：取货 ---
        self._set_state(RescueMission.GO_TO_PICKUP_AREA)
        if not self.goto_zone("pickup"):
            return self._abort("无法到达取货区")
        if not self.ensure_wheels_in_zone("pickup"):
            return self._abort("取货区驱动轮未到位")
        rospy.loginfo("[Mission] 已到达取货区，等待 1.0s")
        rospy.sleep(1.0)

        self._set_state(RescueMission.VISION_GRASP)
        if not self.vision_grasp():
            return self._abort("视觉抓取失败")

        self._set_state(RescueMission.GO_TO_LOADING_AREA)
        if not self.goto_zone("loading"):
            return self._abort("无法到达装货区")
        if not self.ensure_wheels_in_zone("loading"):
            return self._abort("装货区驱动轮未到位")

        # --- 任务三：装到无人机（必须到位并装货完成后，才发 ②） ---
        if self.skip_drone:
            self._loading_completed = True
            rospy.loginfo("[Mission] 跳过无人机：装货区保持 3.0s（对齐完整流程停顿）")
            rospy.sleep(3.0)
        else:
            self._set_state(RescueMission.PUT_ON_DRONE)
            if not self.put_on_drone():
                return self._abort("装货失败")

            self._set_state(RescueMission.NOTIFY_TAKEOFF)
            if not self.notify_drone_loading_done():
                return self._abort("未满足装货完成条件，拒绝发送 loading_done")

            # --- 任务四：无人机投送 + 小车卸货（必须到救援区并卸货完成后，才发 ④） ---
            if not self.wait_drone_delivery(self.order.zone):
                return self._abort("无人机投送超时")

        # 前往救援区前先车体后退 5cm，离开装货区/无人机附近
        backoff_m = 0.05
        rospy.loginfo("[Mission] 前往救援区前，车体后退 %.2fm", backoff_m)
        self.ol.move_body_delta(-backoff_m, 0.0)
        self.stop()
        rospy.sleep(3.0)

        self._set_state(RescueMission.GO_TO_RESCUE_AREA)
        if not self.goto_zone("rescue", zone_id=self.order.zone):
            return self._abort(f"无法到达救援区 {self.order.zone}")
        if not self.ensure_wheels_in_zone("rescue", zone_id=self.order.zone):
            return self._abort("救援区驱动轮未到位")

        if self.skip_drone:
            self._unload_completed = True
            rospy.loginfo("[Mission] 跳过无人机：救援区保持 3.0s（对齐完整流程停顿）")
            rospy.sleep(3.0)
        else:
            self._set_state(RescueMission.UNLOAD_GOODS)
            if not self.unload_to_target_zone():
                return self._abort("卸货失败")

            self._set_state(RescueMission.NOTIFY_UNLOAD_DONE)
            if not self.notify_drone_unload_done():
                return self._abort("未满足卸货完成条件，拒绝发送 unload_done")

        # --- 任务五：返航 ---
        self._set_state(RescueMission.GO_TO_HOME_AREA)
        if not self.goto_zone("parking"):
            return self._abort("无法返回停车区")

        self._set_state(RescueMission.FINISH)
        self.stop()
        rospy.loginfo("[Mission] 全部任务完成")
        return True

    # ---------- 各步骤实现 ----------

    def wait_for_rescue_level(self, timeout=120):
        """等待 /mnt/rescue_target.flag，直接得到最终目标区号 1/2/3/4。"""
        comm = self.config.get("comm", {})
        return wait_for_rescue_target(
            remote_path=comm.get(
                "rescue_target_path", "/mnt/rescue_target.flag"
            ),
            host=comm.get("drone_host", "192.168.31.110"),
            user=comm.get("drone_user", "root"),
            password=comm.get("drone_password", "123456"),
            timeout=timeout,
        )

    def goto_zone(self, zone_name, zone_id=None, guarded=None):
        del guarded  # 兼容旧调用；无 move_base 分支
        if not self._zone_is_calibrated(zone_name, zone_id):
            rospy.logerr(
                "[Mission] refuse navigation to uncalibrated zone=%s id=%s",
                zone_name,
                zone_id,
            )
            return False
        zones = self.config["zones"]
        if zone_name == "rescue":
            pose = zones["rescue"][zone_id]
        else:
            pose = zones[zone_name]

        label = "rescue_%s" % zone_id if zone_name == "rescue" else zone_name
        from_pose = self._est_pose
        dx = float(pose["x"]) - float(from_pose["x"])
        dy = float(pose["y"]) - float(from_pose["y"])
        yaw = float(from_pose.get("yaw", 0.0))
        body_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        body_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        rospy.loginfo(
            "[Nav] open_loop -> %s (%.3f, %.3f, yaw=%.3f) "
            "mapΔ=(%.3f, %.3f) body=(%.3f, %.3f) |Δ|=%.3f",
            label,
            pose["x"],
            pose["y"],
            pose["yaw"],
            dx,
            dy,
            body_x,
            body_y,
            math.hypot(dx, dy),
        )
        self._est_pose = self.ol.go_pose_open_loop(self._est_pose, pose)
        self._verify_and_fix_yaw(float(pose.get("yaw", 0.0)))
        self._current_zone = zone_name
        self._current_zone_id = zone_id
        self.stop()
        rospy.sleep(0.5)
        return True

    # BEGIN added: wheel-in-zone validation entrypoint
    def assert_wheels_in_zone(self, zone_name, zone_id=None):
        return bool(self.check_wheels_in_zone(zone_name, zone_id)["passed"])

    def ensure_wheels_in_zone(self, zone_name, zone_id=None):
        zone_cfg = self.config.get("zone_entry", {})
        # 开环默认不强制入区校验，视为驱动轮已正确进入
        if not zone_cfg.get("enabled", False):
            label = (
                "%s_%s" % (zone_name, zone_id)
                if zone_id is not None
                else zone_name
            )
            rospy.loginfo("[Zone] %s 默认视为驱动轮已正确入区", label)
            return True

        result = self.check_wheels_in_zone(zone_name, zone_id)
        if result["passed"]:
            self._log_zone_entry_quality(result)
            return True

        adjustment = zone_cfg.get("adjustment", {})
        if not adjustment.get("enabled", True):
            return False

        max_attempts = max(0, int(adjustment.get("max_attempts", 3)))
        max_step = max(0.0, float(adjustment.get("max_step", 0.08)))
        max_yaw_step = max(0.0, float(adjustment.get("max_yaw_step", 0.10)))
        max_total = max(0.0, float(adjustment.get("max_total_distance", 0.20)))
        min_distance = max(0.0, float(adjustment.get("min_distance", 0.015)))
        min_yaw = max(0.0, float(adjustment.get("min_yaw", 0.02)))
        settle_time = max(0.0, float(adjustment.get("settle_time", 0.50)))
        total_distance = 0.0

        for attempt in range(1, max_attempts + 1):
            correction = result.get("correction_base")
            correction_yaw = float(result.get("correction_yaw", 0.0))
            if not result.get("available") or correction is None:
                rospy.logwarn(
                    "[ZoneAdjust] %s unavailable reason=%s",
                    zone_name,
                    result.get("reason"),
                )
                break

            remaining = max_total - total_distance
            step = clamp_vector(correction, min(max_step, max(0.0, remaining)))
            step_distance = math.hypot(step[0], step[1])
            yaw_step = max(-max_yaw_step, min(max_yaw_step, correction_yaw))
            if step_distance < min_distance and abs(yaw_step) < min_yaw:
                rospy.logwarn(
                    "[ZoneAdjust] %s correction too small %.3fm/%.3frad but still outside",
                    zone_name,
                    step_distance,
                    abs(yaw_step),
                )
                break

            rospy.loginfo(
                "[ZoneAdjust] %s attempt=%d/%d step=(%.3f, %.3f)m yaw=%.3frad",
                zone_name,
                attempt,
                max_attempts,
                step[0],
                step[1],
                yaw_step,
            )
            if not self._drive_zone_adjustment(step, adjustment, yaw_step=yaw_step):
                rospy.logwarn("[ZoneAdjust] %s attempt blocked", zone_name)
                break
            total_distance += step_distance
            if settle_time:
                rospy.sleep(settle_time)

            result = self.check_wheels_in_zone(zone_name, zone_id)
            if result["passed"]:
                self._log_zone_entry_quality(result)
                return True

        rospy.logerr(
            "[ZoneAdjust] %s failed source=%s reason=%s inside=%s",
            zone_name,
            result.get("source"),
            result.get("reason"),
            result.get("inside_count"),
        )
        return False

    def _drive_zone_adjustment(self, step, config, yaw_step=0.0):
        """入区微调：同样走 LaserOpenLoopNav（开环+白名单避障）。"""
        distance = math.hypot(step[0], step[1])
        yaw_distance = abs(float(yaw_step))
        if distance <= 0.0 and yaw_distance <= 0.0:
            return True
        linear_speed = max(0.01, float(config.get("linear_speed", 0.05)))
        angular_speed = max(0.05, float(config.get("angular_speed", 0.15)))
        old_speed = self.ol.speed
        old_turn = self.ol.turn_speed
        try:
            self.ol.speed = linear_speed
            self.ol.turn_speed = angular_speed
            if yaw_distance > 0.0:
                self.ol.turn_ang(angular_speed, float(yaw_step), guard=False)
            if distance > 0.0:
                self.ol.move_body_delta(step[0], step[1])
        finally:
            self.ol.speed = old_speed
            self.ol.turn_speed = old_turn
            self.ol.stop()
        return True

    def _log_zone_entry_quality(self, result):
        ideal = int(self.config.get("zone_entry", {}).get("ideal_wheels", 4))
        inside = int(result.get("inside_count", 0))
        if inside >= ideal:
            rospy.loginfo("[Zone] %s ideal entry: %d wheels", result["zone"], inside)
        else:
            rospy.logwarn(
                "[Zone] %s legal entry but not ideal: %d/%d wheels",
                result["zone"],
                inside,
                ideal,
            )

    def check_wheels_in_zone(self, zone_name, zone_id=None):
        vision_result = self._assert_wheels_in_zone_by_vision(zone_name, zone_id)
        if vision_result is not None:
            self._last_zone_check = vision_result
            return vision_result

        zone_cfg = self.config.get("zone_entry", {})
        if not zone_cfg.get("fallback_to_map", True):
            result = self._zone_result(
                "none",
                False,
                False,
                "vision_unavailable_map_fallback_disabled",
                zone_name,
                zone_id,
            )
        else:
            result = self._assert_wheels_in_zone_by_map(zone_name, zone_id)
        self._last_zone_check = result
        return result

    def _zone_result(
        self,
        source_name,
        available,
        passed,
        reason,
        zone_name,
        zone_id=None,
        **details,
    ):
        result = {
            "source": source_name,
            "available": bool(available),
            "passed": bool(passed),
            "reason": reason,
            "zone": zone_name,
            "zone_id": zone_id,
        }
        result.update(details)
        return result
    # END added: wheel-in-zone validation entrypoint

    def _init_vision(self):
        if self._camera is not None:
            return
        camera_yaml = self.config.get("vision", {}).get("camera_yaml")
        if camera_yaml:
            camera_yaml = str((_SCRIPT_DIR / camera_yaml).resolve())
        self._camera = CameraCapture(self.config, camera_yaml=camera_yaml)
        self._detector = PickupDetector(self.config)
        self._grasp = GraspController(self.config, cmd_vel_pub=self.velocity_publisher)
        # BEGIN added: wheel-in-zone boundary detector init
        self._zone_detector = ZoneBoundaryDetector(self.config)
        # END added: wheel-in-zone boundary detector init
        rospy.loginfo("vision modules initialized")

    # BEGIN added: wheel-in-zone validation helpers
    def _zone_key(self, zone_name, zone_id=None):
        if zone_name == "rescue":
            return zone_id
        return zone_name

    def _zone_entry(self, zone_name, zone_id=None):
        zone_bounds = self.config.get("zone_bounds", {})
        key = self._zone_key(zone_name, zone_id)
        if zone_name == "rescue":
            rescue = zone_bounds.get("rescue", {})
            return rescue.get(key) or rescue.get(str(key), {})
        return zone_bounds.get(zone_name, {})

    def _zone_is_calibrated(self, zone_name, zone_id=None):
        zone_cfg = self.config.get("zone_entry", {})
        if not zone_cfg.get("require_calibration", True):
            return True
        return calibration_has_zone(
            self.config.get("_calibration", {}), zone_name, zone_id
        )

    def _required_wheels(self, zone_name, zone_id=None):
        entry = self._zone_entry(zone_name, zone_id)
        default_required = 4 if zone_name == "parking" else 2
        return int(entry.get("required_wheels", default_required))

    @staticmethod
    def _stamp_age(stamp):
        if stamp is None or stamp.to_sec() <= 0.0:
            return None
        return max(0.0, (rospy.Time.now() - stamp).to_sec())

    def _localization_metrics(self):
        metrics = {
            "scan_age": self._stamp_age(self._last_scan_stamp),
            "amcl_pose_age": self._stamp_age(self._last_amcl_stamp),
            "tf_age": None,
            "position_std": None,
            "yaw_std": None,
        }
        if self._amcl_pose is not None:
            position_std, yaw_std = covariance_std(
                self._amcl_pose.pose.covariance
            )
            metrics["position_std"] = position_std
            metrics["yaw_std"] = yaw_std
        try:
            latest_tf = self.tf_listener.getLatestCommonTime("map", "base_link")
            metrics["tf_age"] = self._stamp_age(latest_tf)
        except (
            tf.Exception,
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            pass
        return metrics

    def _localization_health(self):
        metrics = self._localization_metrics()
        healthy, reason = evaluate_localization_health(
            metrics, self.config.get("zone_entry", {})
        )
        return healthy, reason, metrics

    def _current_pose_map(self):
        try:
            trans, rot = self.tf_listener.lookupTransform(
                "map", "base_link", rospy.Time(0)
            )
            _, _, yaw = euler_from_quaternion(rot)
            return float(trans[0]), float(trans[1]), float(yaw)
        except (
            tf.Exception,
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            return None

    def _current_yaw_map(self):
        pose = self._current_pose_map()
        return None if pose is None else float(pose[2])

    def _verify_and_fix_yaw(self, target_yaw, tolerance=0.12, max_retries=1):
        """用 TF 校验实际航向，未对齐则再转一次（解决开环假成功）。"""
        for attempt in range(max_retries + 1):
            actual = self._current_yaw_map()
            if actual is None:
                rospy.loginfo(
                    "[Nav] TF 航向不可用（rescue 2026 默认不开 AMCL），跳过航向校验"
                )
                return
            dyaw = normalize_yaw(target_yaw - actual)
            if abs(dyaw) <= tolerance:
                rospy.loginfo(
                    "[Nav] 航向校验通过: %.1f° (目标 %.1f°)",
                    math.degrees(actual),
                    math.degrees(target_yaw),
                )
                self._est_pose["yaw"] = actual
                return
            rospy.logwarn(
                "[Nav] 航向偏差 %.1f°，第 %d/%d 次重试对齐",
                math.degrees(dyaw),
                attempt + 1,
                max_retries + 1,
            )
            self.ol.face_yaw(actual, target_yaw)
        rospy.logerr("[Nav] 航向多次重试仍未对齐，继续任务")

    def _wheel_points_base(self):
        return wheel_outer_points_base(self.config.get("vehicle", {}))

    def _zone_polygon_map(self, zone_name, zone_id=None):
        zone_bounds = self.config.get("zone_bounds", {})
        entry = self._zone_entry(zone_name, zone_id)
        if entry:
            return resolve_zone_polygon(
                zone_bounds,
                zone_name,
                zone_id=zone_id,
                fallback_pose=entry.get("center"),
                default_size=entry.get("size", [0.8, 0.8]),
            )
        pose = self.config.get("zones", {}).get(zone_name)
        if zone_name == "rescue":
            rescue = self.config.get("zones", {}).get("rescue", {})
            pose = rescue.get(zone_id) or rescue.get(str(zone_id))
        if pose is None:
            return None
        return rectangle_polygon(pose, [0.8, 0.8])

    def _save_zone_vision_debug(self, frame, observation, zone_name):
        if not self.config.get("vision", {}).get("save_debug", True):
            return
        debug_dir = self.config.get("vision", {}).get(
            "debug_dir", "/tmp/rescue_vision"
        )
        visual = self._zone_detector.draw_debug(frame, observation, None)
        self._camera.save_debug(
            debug_dir, "zone_%s_vision.jpg" % zone_name, visual
        )

    def _assert_wheels_in_zone_by_vision(self, zone_name, zone_id=None):
        zone_cfg = self.config.get("zone_entry", {})
        if not zone_cfg.get("enabled", True) or not zone_cfg.get(
            "prefer_vision", True
        ):
            return None

        self._init_vision()
        if not self._zone_detector.homography_available:
            rospy.logwarn(
                "[ZoneVision] %s unavailable: %s",
                zone_name,
                self._zone_detector.homography_reason,
            )
            return None

        sample_count = max(1, int(zone_cfg.get("vision_sample_count", 5)))
        consensus_count = max(1, int(zone_cfg.get("vision_consensus_count", 3)))
        min_valid_samples = max(
            consensus_count, int(zone_cfg.get("vision_min_valid_samples", 3))
        )
        sample_interval = float(zone_cfg.get("vision_sample_interval", 0.08))
        confidence_limit = float(zone_cfg.get("vision_confidence", 0.65))
        margin = float(self.config.get("vehicle", {}).get("zone_margin", 0.04))
        required = self._required_wheels(zone_name, zone_id)
        wheels = self._wheel_points_base()
        samples = []
        last_frame = None
        last_observation = None

        for sample_index in range(sample_count):
            frame = self._camera.get_frame(discard=1)
            if frame is None:
                continue
            observation = self._zone_detector.detect(frame, zone_name)
            last_frame = frame
            last_observation = observation
            if observation is None:
                continue
            if observation.confidence < confidence_limit:
                continue
            if not observation.geometry_valid or observation.polygon_base is None:
                continue
            inside_count = count_points_in_polygon(
                wheels, observation.polygon_base, margin=margin
            )
            center_base = polygon_center(observation.polygon_base)
            samples.append(
                {
                    "passed": inside_count >= required,
                    "inside_count": inside_count,
                    "confidence": observation.confidence,
                    "correction_base": center_base,
                }
            )
            if sample_index + 1 < sample_count:
                rospy.sleep(sample_interval)

        if last_frame is not None and last_observation is not None:
            self._save_zone_vision_debug(
                last_frame, last_observation, zone_name
            )

        if len(samples) < min_valid_samples:
            rospy.logwarn(
                "[ZoneVision] %s valid samples=%d required=%d, fallback to map",
                zone_name,
                len(samples),
                min_valid_samples,
            )
            return None

        pass_count = sum(1 for sample in samples if sample["passed"])
        fail_count = len(samples) - pass_count
        if pass_count >= consensus_count:
            passed = True
            reason = "vision_consensus_inside"
        elif fail_count >= consensus_count:
            passed = False
            reason = "vision_consensus_outside"
        else:
            rospy.logwarn("[ZoneVision] %s no consensus, fallback to map", zone_name)
            return None

        inside_counts = sorted(sample["inside_count"] for sample in samples)
        median_inside = inside_counts[len(inside_counts) // 2]
        mean_confidence = sum(
            sample["confidence"] for sample in samples
        ) / float(len(samples))
        correction_x = sorted(sample["correction_base"][0] for sample in samples)
        correction_y = sorted(sample["correction_base"][1] for sample in samples)
        middle = len(samples) // 2
        correction_base = (correction_x[middle], correction_y[middle])
        rospy.loginfo(
            "[ZoneVision] %s passed=%s inside=%d required=%d votes=%d/%d",
            zone_name,
            passed,
            median_inside,
            required,
            pass_count,
            len(samples),
        )
        return self._zone_result(
            "vision",
            True,
            passed,
            reason,
            zone_name,
            zone_id,
            inside_count=median_inside,
            required_wheels=required,
            confidence=mean_confidence,
            valid_samples=len(samples),
            pass_votes=pass_count,
            correction_base=correction_base,
        )

    def _assert_wheels_in_zone_by_map(self, zone_name, zone_id=None):
        if not self._zone_is_calibrated(zone_name, zone_id):
            return self._zone_result(
                "map",
                False,
                False,
                "zone_not_calibrated",
                zone_name,
                zone_id,
                calibration=self.config.get("_calibration", {}),
            )

        healthy, health_reason, metrics = self._localization_health()
        if not healthy:
            rospy.logwarn("[ZoneMap] %s unavailable: %s", zone_name, health_reason)
            return self._zone_result(
                "map",
                False,
                False,
                health_reason,
                zone_name,
                zone_id,
                health=metrics,
            )

        pose = self._current_pose_map()
        if pose is None:
            return self._zone_result(
                "map",
                False,
                False,
                "tf_lookup_failed",
                zone_name,
                zone_id,
                health=metrics,
            )

        target = self.config.get("zones", {}).get(zone_name)
        if zone_name == "rescue":
            rescue = self.config.get("zones", {}).get("rescue", {})
            target = rescue.get(zone_id) or rescue.get(str(zone_id))
        if target is None:
            return self._zone_result(
                "map",
                False,
                False,
                "zone_target_missing",
                zone_name,
                zone_id,
                health=metrics,
            )

        x, y, yaw = pose
        pose_error = safe_pose_error(pose, target)
        correction_base = pose_error["correction_base"]
        validation = self.config.get("zone_entry", {}).get("map_validation", {})
        mode = validation.get("mode", "safe_pose")
        required = self._required_wheels(zone_name, zone_id)

        if mode == "safe_pose":
            position_error = pose_error["position_error"]
            yaw_error = pose_error["yaw_error"]
            position_tolerance = float(validation.get("position_tolerance", 0.04))
            yaw_tolerance = float(validation.get("yaw_tolerance", 0.12))
            passed = (
                position_error <= position_tolerance
                and abs(yaw_error) <= yaw_tolerance
            )
            estimated_wheels = int(
                self.config.get("zone_entry", {}).get("ideal_wheels", 4)
            ) if passed else 0
            rospy.loginfo(
                "[ZoneMap] %s safe_pose passed=%s position_error=%.3f/%.3f "
                "yaw_error=%.3f/%.3f",
                zone_name,
                passed,
                position_error,
                position_tolerance,
                abs(yaw_error),
                yaw_tolerance,
            )
            return self._zone_result(
                "map",
                True,
                passed,
                "map_safe_pose_inside" if passed else "map_safe_pose_outside",
                zone_name,
                zone_id,
                inside_count=estimated_wheels,
                required_wheels=required,
                wheel_count_estimated=True,
                position_error=position_error,
                position_tolerance=position_tolerance,
                yaw_error=yaw_error,
                yaw_tolerance=yaw_tolerance,
                pose={"x": x, "y": y, "yaw": yaw},
                target=target,
                correction_base=correction_base,
                correction_yaw=yaw_error,
                health=metrics,
            )

        polygon = self._zone_polygon_map(zone_name, zone_id)
        if polygon is None:
            return self._zone_result(
                "map",
                False,
                False,
                "zone_polygon_missing",
                zone_name,
                zone_id,
                health=metrics,
            )

        wheels_map = transform_points(x, y, yaw, self._wheel_points_base())
        center_map = polygon_center(polygon)
        if center_map is not None:
            correction_base = map_vector_to_base(
                center_map[0] - x, center_map[1] - y, yaw
            )
        margin = float(self.config.get("vehicle", {}).get("zone_margin", 0.04))
        inside_count = count_points_in_polygon(
            wheels_map, polygon, margin=margin
        )
        passed = inside_count >= required
        rospy.loginfo(
            "[ZoneMap] %s passed=%s inside=%d required=%d pose=(%.2f, %.2f, %.2f)",
            zone_name,
            passed,
            inside_count,
            required,
            x,
            y,
            yaw,
        )
        return self._zone_result(
            "map",
            True,
            passed,
            "map_geometry_inside" if passed else "map_geometry_outside",
            zone_name,
            zone_id,
            inside_count=inside_count,
            required_wheels=required,
            pose={"x": x, "y": y, "yaw": yaw},
            wheels=wheels_map,
            correction_base=correction_base,
            health=metrics,
        )

    def _save_vision_debug(self, frame, name, target=None):
        if not self.config.get("vision", {}).get("save_debug", True):
            return
        if self._camera is None or self._detector is None:
            return
        debug_dir = self.config.get("vision", {}).get(
            "debug_dir", "/tmp/rescue_vision"
        )
        visual = self._detector.draw_debug(frame, target)
        self._camera.save_debug(debug_dir, name, visual)

    def vision_grasp(self):
        # 默认跳过真实抓取，联调/无机械臂时直接成功
        if self.config.get("mission", {}).get("allow_unimplemented_actions", True):
            rospy.logwarn("vision_grasp stub bypass enabled")
            return True

        self._init_vision()
        grasp_cfg = self.config.get("vision", {}).get("grasp", {})
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
                twist_guard_fn=self._align_twist_guard,
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

    def _mission_cfg(self):
        return self.config.get("mission") or {}

    def _wait_action_hold(self, action_name, hold_key, default_hold):
        """装货/卸货保持或人工确认；返回是否允许视为动作完成。"""
        mission = self._mission_cfg()
        hold_sec = float(mission.get(hold_key, default_hold))
        if hold_sec > 0:
            rospy.loginfo(
                "[Mission] %s：到位后保持 %.1fs（完成真实操作）",
                action_name,
                hold_sec,
            )
            rospy.sleep(hold_sec)
        if mission.get("wait_operator_confirm", False):
            rospy.logwarn(
                "[Mission] %s：请确认已完成后，在终端按回车继续发信号",
                action_name,
            )
            try:
                input("按回车确认 %s 完成..." % action_name)
            except Exception as exc:
                rospy.logerr("[Mission] 人工确认失败: %s", exc)
                return False
        return True

    def put_on_drone(self):
        """到达装货区并完成装货后才返回 True；之后才允许发 loading_done。"""
        self._loading_completed = False
        if self._current_zone != "loading":
            rospy.logerr(
                "[Mission] put_on_drone 拒绝：当前不在装货区 (zone=%s)",
                self._current_zone,
            )
            return False

        rospy.loginfo("[Mission] 已到达装货区，开始装货到无人机平台")
        if self._mission_cfg().get("allow_unimplemented_actions", True):
            rospy.logwarn(
                "put_on_drone：机械臂未接入，使用到位保持/确认代替真实装货"
            )
            if not self._wait_action_hold("装货", "loading_hold_sec", 10.0):
                return False
            self._loading_completed = True
            rospy.loginfo("[Mission] 装货完成（可发送 loading_done）")
            return True

        rospy.logerr("put_on_drone is not implemented; failing closed")
        return False

    def notify_drone_loading_done(self):
        """仅在装货区完成装货后发送 ② loading_done.flag。"""
        if self._current_zone != "loading" or not self._loading_completed:
            rospy.logerr(
                "[Comm] 拒绝发送 loading_done：zone=%s loading_completed=%s",
                self._current_zone,
                self._loading_completed,
            )
            return False
        comm = self.config.get("comm", {})
        rospy.loginfo("[Comm] 发送 ② loading_done.flag（装货完成，通知起飞）")
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
        """到达救援区并完成卸货后才返回 True；之后才允许发 unload_done。"""
        self._unload_completed = False
        if self._current_zone != "rescue":
            rospy.logerr(
                "[Mission] unload 拒绝：当前不在救援区 (zone=%s)",
                self._current_zone,
            )
            return False

        rospy.loginfo(
            "[Mission] 已到达救援区 %s，开始卸货",
            self._current_zone_id,
        )
        if self._mission_cfg().get("allow_unimplemented_actions", True):
            rospy.logwarn(
                "unload_to_target_zone：机械臂未接入，使用到位保持/确认代替真实卸货"
            )
            if not self._wait_action_hold("卸货", "unload_hold_sec", 10.0):
                return False
            self._unload_completed = True
            rospy.loginfo("[Mission] 卸货完成（可发送 unload_done）")
            return True

        rospy.logerr("unload_to_target_zone is not implemented; failing closed")
        return False

    def notify_drone_unload_done(self):
        """仅在救援区完成卸货后发送 ④ unload_done.flag。"""
        if self._current_zone != "rescue" or not self._unload_completed:
            rospy.logerr(
                "[Comm] 拒绝发送 unload_done：zone=%s unload_completed=%s",
                self._current_zone,
                self._unload_completed,
            )
            return False
        comm = self.config.get("comm", {})
        rospy.loginfo("[Comm] 发送 ④ unload_done.flag（救援区卸货完成）")
        notify_unload_done(
            remote_path=comm.get("unload_done_path", "/mnt/unload_done.flag"),
            host=comm.get("drone_host", "192.168.31.110"),
            user=comm.get("drone_user", "root"),
            password=comm.get("drone_password", "123456"),
        )
        return True

    def _check_cmd_vel_conflicts(self):
        """检查输出话题是否存在多个发布者，避免被其他节点的零速度覆盖。"""
        try:
            master = rospy.get_master()
            code, _msg, state = master.getSystemState()
            if code != 1:
                return
            publishers = state[0] if state else []
            for topic, nodes in publishers:
                if topic == self.cmd_vel_topic:
                    other_nodes = [
                        n for n in nodes if not n.startswith("/nav_rescue_2026")
                    ]
                    if len(other_nodes) > 0:
                        rospy.logerr(
                            "[Nav] %s 存在多个发布者，当前节点指令可能被覆盖: %s",
                            self.cmd_vel_topic,
                            other_nodes,
                        )
                        if self.cmd_vel_topic == "/cmd_vel":
                            rospy.logerr(
                                "[Nav] 如需配合 avoidance_controller，请使用 "
                                "--cmd-vel-topic /cmd_vel_nav"
                            )
                    return
        except Exception as exc:
            rospy.logwarn("[Nav] 检查 %s 发布者失败: %s", self.cmd_vel_topic, exc)

    # ---------- 导航底层：委托 LaserOpenLoopNav（唯一实现） ----------

    def _on_laser(self, msg):
        self._last_scan_stamp = (
            msg.header.stamp
            if msg.header.stamp.to_sec() > 0.0
            else rospy.Time.now()
        )

    def amcl_pose_callback(self, msg):
        self._amcl_pose = msg
        self._last_amcl_stamp = (
            msg.header.stamp
            if msg.header.stamp.to_sec() > 0.0
            else rospy.Time.now()
        )

    def _avoidance_cfg(self):
        return self.ol.avoidance_cfg

    def _align_twist_guard(self, twist):
        """视觉微对位用：原始激光急停/限速（非导航主路径）。"""
        cfg = self._avoidance_cfg()
        if not cfg.get("enabled", True):
            return twist
        if is_emergency(
            self.laser_data,
            cfg,
            twist.linear.x,
            twist.linear.y,
            twist.angular.z,
        ):
            rospy.logwarn("[Avoid] vision align 紧急停车")
            return Twist()
        return guard_twist(
            self.laser_data,
            twist.linear.x,
            twist.linear.y,
            twist.angular.z,
            cfg,
        )

    def stop(self):
        self.ol.stop()

    def turn_ang(self, ang_speed, goal_rotation, guard=False, target_yaw=None):
        self.ol.turn_ang(ang_speed, goal_rotation, guard=guard, target_yaw=target_yaw)

    def go_linear_x(self, linear_speed, goal_distance):
        self.ol.go_linear_x(linear_speed, goal_distance)

    def go_linear_y(self, linear_speed, goal_distance):
        self.ol.go_linear_y(linear_speed, goal_distance)

    def go_pose_open_loop(self, from_pose, to_pose):
        return self.ol.go_pose_open_loop(from_pose, to_pose)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="nav_rescue_2026 救援任务（开环+白名单避障）"
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
        help="底盘速度话题，默认 /cmd_vel；配合 avoidance_controller 时用 /cmd_vel_nav",
    )
    args = parser.parse_args()
    nav = Nav(
        autostart=not args.no_autostart,
        skip_drone=args.skip_drone,
        rescue_zone=args.rescue,
        cmd_vel_topic=args.cmd_vel_topic,
    )
    rospy.spin()
