#!/usr/bin/python3
# coding=UTF-8
"""lar：开环救援任务 + 应用层激光决策绕障。

参考：
  - nav_rescue_2026_1.py：转速×时间航段 + 救援任务流程
  - closeloop.py：救援状态机/航点语义（不依赖 move_base/TEB）
  - laser_avoidance.py：扇区净空、急停、侧移方向选择

运动与避障（应用层，非 TEB）：
  1) 按 YAML 航段以 cmd_vel 开环运动（时间→距离）
  2) 订阅 /scan，主轴遇障时：侧移 → 通过 → 回车道
  3) 紧急净空过近则停车并反向拉开

依赖：
  1) roscore
  2) roslaunch rmep_base rmep_base.launch   # /scan + /cmd_vel
  3) python3 action/lar.py --skip-drone --rescue 2

配置默认：同目录上级 nav_rescue_2026_1.yaml；
避障参数优先读 mission_config.yaml 的 obstacle_avoidance。
"""

from __future__ import print_function

import argparse
import math
import sys
from pathlib import Path

import rospy
import yaml
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

_HERE = Path(__file__).resolve().parent
_NAV_DIR = _HERE.parent
_DEFAULT_TIMED_CONFIG = _NAV_DIR / "nav_rescue_2026_1.yaml"
_DEFAULT_MISSION_CONFIG = _NAV_DIR / "mission_config.yaml"
if str(_NAV_DIR) not in sys.path:
    sys.path.insert(0, str(_NAV_DIR))

from rescue_protocol import (  # noqa: E402
    RescueOrder,
    notify_loading_done,
    notify_unload_done,
    wait_for_delivery_done,
    wait_for_rescue_target,
)
from laser_avoidance import (  # noqa: E402
    get_clearances,
    is_emergency,
    is_path_clear,
    pick_bypass_direction,
)


def load_yaml(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("配置文件不存在: %s" % path)
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError("配置文件格式错误: %s" % path)
    return data


def normalize_yaw(yaw):
    while yaw > math.pi:
        yaw -= 2.0 * math.pi
    while yaw < -math.pi:
        yaw += 2.0 * math.pi
    return yaw


def default_avoidance_cfg():
    return {
        "enabled": True,
        "fail_closed": True,
        "scan_topic": "/scan",
        "lidar_mount": "rear",
        "lidar_to_body": {
            "front": 0.12,
            "back": 0.31,
            "left": 0.25,
            "right": 0.25,
        },
        "safe_distance": 0.20,
        "side_safe_distance": 0.15,
        "critical_distance": 0.10,
        "emergency_stop_distance": 0.05,
        "sector_half_width": 25,
        "side_sector_half_width": 55,
        "bypass_speed": 0.20,
        "creep_ratio": 0.0,
        "sidestep_distance": 0.20,
        "max_sidestep": 0.30,
        "pass_distance": 0.40,
        "max_linear_speed": 0.30,
        "max_angular_speed": 0.40,
        "control_rate": 20,
        "max_bypass_time": 15.0,
        "move_timeout": 90.0,
        "retreat_speed_ratio": 0.5,
        "turn_duration_scale": 0.98,
    }


class LaserDecisionNav(object):
    """开环运动 + 激光决策侧移绕障（应用层）。"""

    def __init__(
        self,
        speed=0.3,
        turn_speed=0.5,
        cmd_vel_topic="/cmd_vel",
        control_rate=20,
        segment_pause_sec=0.30,
        avoidance_cfg=None,
        log_prefix="[LarNav]",
    ):
        self.speed = float(speed)
        self.turn_speed = float(turn_speed)
        self.segment_pause_sec = max(0.0, float(segment_pause_sec))
        self.log_prefix = log_prefix
        self._est = {"x": 0.0, "y": 0.0, "yaw": 0.0}

        self.avoidance_cfg = dict(default_avoidance_cfg())
        if avoidance_cfg:
            self.avoidance_cfg.update(avoidance_cfg)

        scan_topic = str(self.avoidance_cfg.get("scan_topic", "/scan"))
        rate_hz = int(self.avoidance_cfg.get("control_rate", control_rate))
        self.rate = rospy.Rate(max(5, rate_hz))
        self.dt = 1.0 / float(max(5, rate_hz))

        self.pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        self.laser_data = None
        self._laser_stamp = None
        rospy.Subscriber(scan_topic, LaserScan, self._on_scan, queue_size=1)
        rospy.sleep(0.3)
        rospy.loginfo(
            "%s ready cmd=%s scan=%s avoid=%s mount=%s speed=%.2f",
            self.log_prefix,
            cmd_vel_topic,
            scan_topic,
            self.avoidance_cfg.get("enabled", True),
            self.avoidance_cfg.get("lidar_mount", "rear"),
            self.speed,
        )

    def _on_scan(self, msg):
        self.laser_data = msg
        self._laser_stamp = rospy.Time.now()

    def get_est_pose(self):
        return dict(self._est)

    def set_est_pose(self, pose):
        self._est = {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "yaw": float(pose.get("yaw", 0.0)),
        }

    def stop(self):
        self.pub.publish(Twist())

    def _scan_ok(self):
        cfg = self.avoidance_cfg
        if self.laser_data is None:
            return None
        max_age = float(cfg.get("max_scan_age", 0.35))
        if self._laser_stamp is not None:
            age = (rospy.Time.now() - self._laser_stamp).to_sec()
            if age > max_age:
                return None
        return self.laser_data

    def _publish_dt(self, vx, vy, wz=0.0):
        max_v = float(self.avoidance_cfg.get("max_linear_speed", 0.30))
        max_w = float(self.avoidance_cfg.get("max_angular_speed", 0.40))
        twist = Twist()
        twist.linear.x = max(-max_v, min(max_v, float(vx)))
        twist.linear.y = max(-max_v, min(max_v, float(vy)))
        twist.angular.z = max(-max_w, min(max_w, float(wz)))
        self.pub.publish(twist)
        self.rate.sleep()

    def _obstacle_side_clear(self, scan, side):
        if scan is None:
            return True
        cfg = self.avoidance_cfg
        side_safe = float(cfg.get("side_safe_distance", 0.15))
        c = get_clearances(scan, cfg)
        if side == "left":
            return c["left"] >= side_safe
        if side == "right":
            return c["right"] >= side_safe
        return True

    def _move_axis_with_laser(self, axis, signed_speed, goal_distance):
        """沿车体 x/y 轴走 goal_distance；遇障侧移绕行。"""
        cfg = self.avoidance_cfg
        speed = abs(float(signed_speed))
        goal_distance = abs(float(goal_distance))
        if speed < 1e-6 or goal_distance < 1e-4:
            return

        if not cfg.get("enabled", True):
            # 纯定时：无激光决策
            duration = goal_distance / speed
            twist = Twist()
            if axis == "x":
                twist.linear.x = signed_speed
            else:
                twist.linear.y = signed_speed
            t0 = rospy.Time.now().to_sec()
            while not rospy.is_shutdown():
                if rospy.Time.now().to_sec() - t0 >= duration:
                    break
                self.pub.publish(twist)
                self.rate.sleep()
            self.stop()
            return

        vx = signed_speed if axis == "x" else 0.0
        vy = signed_speed if axis == "y" else 0.0
        main_v = signed_speed
        dt = self.dt
        progress = 0.0
        deadline = rospy.Time.now().to_sec() + float(cfg.get("move_timeout", 90.0))
        max_bypass = float(cfg.get("max_bypass_time", 15.0))
        max_sidestep = float(cfg.get("max_sidestep", cfg.get("sidestep_distance", 0.30)))
        pass_need = float(cfg.get("pass_distance", 0.40))
        bypass_speed = float(cfg.get("bypass_speed", 0.20))
        creep_ratio = float(cfg.get("creep_ratio", 0.0))
        retreat_ratio = float(cfg.get("retreat_speed_ratio", 0.5))

        phase = "normal"
        lateral_offset = 0.0
        pass_progress = 0.0
        bypass_vy = 0.0
        bypass_side = ""
        phase_start = rospy.Time.now().to_sec()

        rospy.loginfo(
            "%s move_%s dist=%.2fm speed=%.2f (laser decision ON)",
            self.log_prefix,
            axis,
            goal_distance,
            signed_speed,
        )

        while progress < goal_distance and not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            if now > deadline:
                rospy.logwarn(
                    "%s 超时 %.2f/%.2fm phase=%s",
                    self.log_prefix,
                    progress,
                    goal_distance,
                    phase,
                )
                break

            scan = self._scan_ok()
            if scan is None and cfg.get("fail_closed", True):
                self.stop()
                rospy.logwarn_throttle(1.0, "%s 无激光 fail-closed", self.log_prefix)
                rospy.sleep(dt)
                continue

            if is_emergency(scan, cfg, vx, vy, 0.0):
                self.stop()
                retreat = speed * retreat_ratio
                if axis == "x":
                    self._publish_dt(-retreat if main_v > 0 else retreat, 0.0)
                else:
                    self._publish_dt(0.0, -retreat if main_v > 0 else retreat)
                continue

            cmd_vx = 0.0
            cmd_vy = 0.0

            if phase == "normal":
                if is_path_clear(scan, vx, vy, cfg):
                    if axis == "x":
                        cmd_vx = main_v
                    else:
                        cmd_vy = main_v
                    self._publish_dt(cmd_vx, cmd_vy)
                    progress += speed * dt
                else:
                    vy_pick, side = pick_bypass_direction(scan, cfg)
                    if side == "backward":
                        phase = "retreat"
                        phase_start = now
                        rospy.logwarn("%s 两侧受阻，先后退", self.log_prefix)
                    else:
                        phase = "step_out"
                        phase_start = now
                        bypass_vy = vy_pick
                        bypass_side = side
                        pass_progress = 0.0
                        lateral_offset = 0.0
                        rospy.loginfo(
                            "%s 侧移绕障 side=%s", self.log_prefix, bypass_side
                        )
                    rospy.sleep(dt)
                    continue

            elif phase == "step_out":
                if now - phase_start > max_bypass:
                    phase = "retreat"
                    phase_start = now
                    rospy.sleep(dt)
                    continue
                creep = speed * creep_ratio
                if axis == "x":
                    cmd_vx = (creep if main_v > 0 else -creep) if creep > 1e-6 else 0.0
                    cmd_vy = bypass_vy
                else:
                    cmd_vy = (creep if main_v > 0 else -creep) if creep > 1e-6 else 0.0
                    cmd_vx = bypass_vy
                if is_emergency(scan, cfg, cmd_vx, cmd_vy, 0.0):
                    phase = "retreat"
                    phase_start = now
                    self.stop()
                    rospy.sleep(dt)
                    continue
                self._publish_dt(cmd_vx, cmd_vy)
                lateral_offset += abs(bypass_speed) * dt
                if is_path_clear(scan, vx, vy, cfg) or lateral_offset >= max_sidestep:
                    phase = "pass"
                    pass_progress = 0.0
                    phase_start = now
                    rospy.loginfo(
                        "%s 侧移 %.2fm → 通过", self.log_prefix, lateral_offset
                    )

            elif phase == "pass":
                if now - phase_start > max_bypass:
                    phase = "step_out"
                    phase_start = now
                    rospy.sleep(dt)
                    continue
                if not is_path_clear(scan, vx, vy, cfg):
                    phase = "step_out"
                    phase_start = now
                    rospy.sleep(dt)
                    continue
                if axis == "x":
                    cmd_vx = main_v
                else:
                    cmd_vy = main_v
                self._publish_dt(cmd_vx, cmd_vy)
                progress += speed * dt
                pass_progress += speed * dt
                if pass_progress >= pass_need and self._obstacle_side_clear(
                    scan, bypass_side
                ):
                    phase = "step_back"
                    phase_start = now

            elif phase == "step_back":
                if axis == "x":
                    cmd_vy = -bypass_vy
                    cmd_vx = 0.0
                else:
                    cmd_vx = -bypass_vy
                    cmd_vy = 0.0
                if is_emergency(scan, cfg, cmd_vx, cmd_vy, 0.0):
                    phase = "normal"
                    lateral_offset = 0.0
                    rospy.sleep(dt)
                    continue
                self._publish_dt(cmd_vx, cmd_vy)
                lateral_offset -= abs(bypass_speed) * dt
                if lateral_offset <= 0.0:
                    lateral_offset = 0.0
                    phase = "normal"
                    rospy.loginfo("%s 回车道完成", self.log_prefix)

            elif phase == "retreat":
                if now - phase_start > max_bypass * 0.5:
                    phase = "normal"
                    phase_start = now
                    rospy.sleep(dt)
                    continue
                retreat = speed * retreat_ratio
                if axis == "x":
                    self._publish_dt(-retreat if main_v > 0 else retreat, 0.0)
                else:
                    self._publish_dt(0.0, -retreat if main_v > 0 else retreat)
                if is_path_clear(scan, vx, vy, cfg):
                    phase = "normal"

            else:
                phase = "normal"
                rospy.sleep(dt)

        self.stop()

    def go_linear_x_timed(self, linear_speed, duration):
        spd = float(linear_speed)
        duration = float(duration)
        if abs(spd) < 1e-6 or duration < 1e-4:
            return
        dist = abs(spd) * duration
        self._move_axis_with_laser("x", spd, dist)
        signed = math.copysign(dist, spd)
        yaw = self._est["yaw"]
        self._est["x"] += math.cos(yaw) * signed
        self._est["y"] += math.sin(yaw) * signed

    def go_linear_y_timed(self, linear_speed, duration):
        spd = float(linear_speed)
        duration = float(duration)
        if abs(spd) < 1e-6 or duration < 1e-4:
            return
        dist = abs(spd) * duration
        self._move_axis_with_laser("y", spd, dist)
        signed = math.copysign(dist, spd)
        yaw = self._est["yaw"]
        self._est["x"] += -math.sin(yaw) * signed
        self._est["y"] += math.cos(yaw) * signed

    def turn_ang_timed(self, ang_speed, duration, direction=1):
        """转向：开环时间，激光仅做急停监护（不侧移）。"""
        spd = abs(float(ang_speed))
        duration = float(duration)
        if spd < 1e-6 or duration < 1e-4:
            return
        direction = 1 if direction >= 0 else -1
        scale = float(self.avoidance_cfg.get("turn_duration_scale", 0.98))
        duration *= scale
        signed_w = -spd if direction > 0 else spd
        cfg = self.avoidance_cfg
        t0 = rospy.Time.now().to_sec()
        rospy.loginfo(
            "%s turn wz=%.3f duration=%.2fs", self.log_prefix, signed_w, duration
        )
        while not rospy.is_shutdown():
            if rospy.Time.now().to_sec() - t0 >= duration:
                break
            scan = self._scan_ok()
            if cfg.get("enabled", True):
                if scan is None and cfg.get("fail_closed", True):
                    self.stop()
                    rospy.sleep(self.dt)
                    continue
                if is_emergency(scan, cfg, 0.0, 0.0, signed_w):
                    self.stop()
                    rospy.logwarn_throttle(1.0, "%s 转向急停", self.log_prefix)
                    rospy.sleep(self.dt)
                    continue
            self._publish_dt(0.0, 0.0, signed_w)
        self.stop()
        elapsed = min(rospy.Time.now().to_sec() - t0, duration)
        self._est["yaw"] = normalize_yaw(
            self._est["yaw"] + direction * spd * elapsed
        )

    def run_segment(self, body_x_sign, body_y_sign, yaw_delta_sign, seg):
        speed = float(seg.get("speed", self.speed))
        turn_speed = float(seg.get("turn_speed", self.turn_speed))
        dx_t = float(seg.get("duration_x", 0.0))
        dy_t = float(seg.get("duration_y", 0.0))
        dt_t = float(seg.get("duration_turn", 0.0))

        rospy.loginfo(
            "%s segment speed=%.2f turn=%.2f "
            "dur_x=%.2fs dur_y=%.2fs dur_turn=%.2fs signs=(x=%+.0f,y=%+.0f,yaw=%+.0f)",
            self.log_prefix,
            speed,
            turn_speed,
            dx_t,
            dy_t,
            dt_t,
            body_x_sign,
            body_y_sign,
            yaw_delta_sign,
        )

        if dy_t > 1e-4 and abs(body_y_sign) > 1e-9:
            self.go_linear_y_timed(speed if body_y_sign > 0 else -speed, dy_t)
            if self.segment_pause_sec > 0:
                rospy.sleep(self.segment_pause_sec)

        if dx_t > 1e-4 and abs(body_x_sign) > 1e-9:
            self.go_linear_x_timed(speed if body_x_sign > 0 else -speed, dx_t)
            if self.segment_pause_sec > 0:
                rospy.sleep(self.segment_pause_sec)

        if dt_t > 1e-4 and abs(yaw_delta_sign) > 1e-9:
            self.turn_ang_timed(
                turn_speed, dt_t, direction=1 if yaw_delta_sign > 0 else -1
            )
            if self.segment_pause_sec > 0:
                rospy.sleep(self.segment_pause_sec)

    def run_backoff_x(self, seg):
        speed = abs(float(seg.get("speed", self.speed)))
        dx_t = float(seg.get("duration_x", 0.0))
        if dx_t < 1e-4:
            return
        self.go_linear_x_timed(-speed, dx_t)
        if self.segment_pause_sec > 0:
            rospy.sleep(self.segment_pause_sec)


class NavLar(object):
    """救援任务（closeloop 流程）+ 开环航段（nav_rescue_2026_1）+ 激光绕障。"""

    def __init__(
        self,
        config_path=None,
        mission_config_path=None,
        skip_drone=False,
        rescue_zone=2,
        cmd_vel_topic=None,
        autostart=True,
        no_avoid=False,
    ):
        rospy.init_node("nav_rescue_lar_laser", anonymous=True)
        self.skip_drone = bool(skip_drone)
        self.rescue_zone = int(rescue_zone)
        self.config_path = Path(config_path or _DEFAULT_TIMED_CONFIG)
        self.config = load_yaml(self.config_path)

        # 避障：mission_config.obstacle_avoidance 优先，再被 timed 配置覆盖
        avoid = dict(default_avoidance_cfg())
        mission_cfg_path = Path(mission_config_path or _DEFAULT_MISSION_CONFIG)
        if mission_cfg_path.exists():
            try:
                mc = load_yaml(mission_cfg_path)
                avoid.update(mc.get("obstacle_avoidance") or {})
            except Exception as exc:
                rospy.logwarn("[Lar] 读取 mission_config 避障失败: %s", exc)
        avoid.update(self.config.get("obstacle_avoidance") or {})
        if no_avoid:
            avoid["enabled"] = False

        timed = dict(self.config.get("timed") or {})
        self.mission_cfg = dict(self.config.get("mission") or {})
        self.segment_motion = dict(self.config.get("segment_motion") or {})
        self.arrive_pause_sec = float(timed.get("arrive_pause_sec", 0.50))

        speed = float(timed.get("speed", 0.3))
        turn_speed = float(timed.get("turn_speed", 0.5))
        topic = str(
            cmd_vel_topic
            if cmd_vel_topic
            else timed.get("cmd_vel_topic", "/cmd_vel")
        )

        self.nav = LaserDecisionNav(
            speed=speed,
            turn_speed=turn_speed,
            cmd_vel_topic=topic,
            control_rate=int(
                avoid.get("control_rate", timed.get("control_rate", 20))
            ),
            segment_pause_sec=float(timed.get("segment_pause_sec", 0.30)),
            avoidance_cfg=avoid,
        )
        parking = (self.config.get("zones") or {}).get("parking") or {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
        }
        self.nav.set_est_pose(parking)
        self.order = None
        self._loading_completed = False
        self._unload_completed = False
        self._current_zone = None
        self._current_zone_id = None

        rospy.loginfo(
            "[Lar] config=%s skip_drone=%s rescue=%s avoid=%s",
            self.config_path,
            self.skip_drone,
            self.rescue_zone,
            avoid.get("enabled", True),
        )
        if autostart:
            ok = self.run_mission()
            rospy.loginfo("[Lar] 结束 ok=%s", ok)

    def stop(self):
        self.nav.stop()

    def _abort(self, reason):
        rospy.logerr("[Lar] abort: %s", reason)
        self.stop()
        return False

    def _zone_pose(self, zone_name, zone_id=None):
        zones = self.config["zones"]
        if zone_name == "rescue":
            return zones["rescue"][int(zone_id)]
        return zones[zone_name]

    def _segment_key(self, zone_name, zone_id=None):
        if zone_name == "pickup":
            return "to_pickup"
        if zone_name == "loading":
            return "to_loading"
        if zone_name == "rescue":
            return "to_rescue_%d" % int(zone_id)
        if zone_name == "parking":
            rid = zone_id
            if rid is None:
                rid = self._current_zone_id
            if rid is None and self.order is not None:
                rid = self.order.zone
            if rid is None:
                return "to_parking"
            return "to_parking_%d" % int(rid)
        return "to_%s" % zone_name

    def _signs_from_poses(self, from_pose, to_pose):
        dx = float(to_pose["x"]) - float(from_pose["x"])
        dy = float(to_pose["y"]) - float(from_pose["y"])
        yaw = float(from_pose.get("yaw", 0.0))
        body_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        body_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        dyaw = normalize_yaw(
            float(to_pose.get("yaw", yaw)) - float(from_pose.get("yaw", yaw))
        )
        sx = 0.0 if abs(body_x) < 1e-4 else (1.0 if body_x > 0 else -1.0)
        sy = 0.0 if abs(body_y) < 1e-4 else (1.0 if body_y > 0 else -1.0)
        st = 0.0 if abs(dyaw) < 1e-4 else (1.0 if dyaw > 0 else -1.0)
        return sx, sy, st, body_x, body_y, dyaw

    def goto_zone(self, zone_name, zone_id=None):
        pose = self._zone_pose(zone_name, zone_id)
        label = "rescue_%s" % zone_id if zone_name == "rescue" else zone_name
        key_zone_id = zone_id
        if zone_name == "parking":
            key_zone_id = (
                self._current_zone_id
                if self._current_zone_id is not None
                else (self.order.zone if self.order is not None else None)
            )
            if key_zone_id is not None:
                label = "parking(from_rescue_%s)" % key_zone_id
        key = self._segment_key(zone_name, key_zone_id)
        seg = self.segment_motion.get(key)
        if not seg:
            return self._abort("配置缺少 segment_motion.%s" % key)

        from_pose = self.nav.get_est_pose()
        sx, sy, st, body_x, body_y, dyaw = self._signs_from_poses(from_pose, pose)
        rospy.loginfo(
            "[Lar] goto %s via %s body≈(%.3f, %.3f) dyaw=%.1f°",
            label,
            key,
            body_x,
            body_y,
            math.degrees(dyaw),
        )
        self.nav.run_segment(sx, sy, st, seg)
        self.nav.set_est_pose(pose)
        self._current_zone = zone_name
        self._current_zone_id = zone_id
        self.stop()
        if self.arrive_pause_sec > 0:
            rospy.sleep(self.arrive_pause_sec)
        return True

    def run_mission(self):
        timeouts = self.config.get("timeouts", {})
        mission = self.mission_cfg

        if not self.goto_zone("pickup"):
            return self._abort("无法到达取货区")
        pickup_hold = float(mission.get("pickup_hold_sec", 1.0))
        if pickup_hold > 0:
            rospy.sleep(pickup_hold)
        rospy.logwarn("[Lar] vision_grasp stub bypass")

        if not self.goto_zone("loading"):
            return self._abort("无法到达装货区")

        if self.skip_drone:
            self.order = RescueOrder(zone=self.rescue_zone, level=1)
            rospy.loginfo("[Lar] 跳过无人机，救援区 zone=%d", self.order.zone)
        else:
            comm = self.config.get("comm", {})
            self.order = wait_for_rescue_target(
                remote_path=comm.get(
                    "rescue_target_path", "/mnt/rescue_target.flag"
                ),
                host=comm.get("drone_host", "192.168.10.66"),
                user=comm.get("drone_user", "forlinx"),
                password=comm.get("drone_password", "forlinx"),
                timeout=timeouts.get("wait_drone_cmd", 600),
            )
            if self.order is None:
                return self._abort("未收到无人机救援目标点")

        if self.skip_drone:
            self._loading_completed = True
            hold = float(mission.get("skip_drone_loading_hold_sec", 3.0))
            if hold > 0:
                rospy.sleep(hold)
        else:
            hold = float(mission.get("loading_hold_sec", 10.0))
            if hold > 0:
                rospy.sleep(hold)
            self._loading_completed = True
            comm = self.config.get("comm", {})
            notify_loading_done(
                remote_path=comm.get(
                    "loading_done_path", "/mnt/loading_done.flag"
                ),
                host=comm.get("drone_host", "192.168.31.110"),
                user=comm.get("drone_user", "root"),
                password=comm.get("drone_password", "123456"),
            )
            if not wait_for_delivery_done(
                remote_path=comm.get(
                    "delivery_done_path", "/mnt/delivery_done.flag"
                ),
                host=comm.get("drone_host", "192.168.31.110"),
                user=comm.get("drone_user", "root"),
                password=comm.get("drone_password", "123456"),
                timeout=timeouts.get("delivery", 300),
            ):
                return self._abort("无人机投送超时")

        if mission.get("pre_rescue_backoff_enable", True):
            seg = self.segment_motion.get("pre_rescue_backoff")
            if seg:
                self.nav.run_backoff_x(seg)
                self.stop()
                rospy.sleep(0.3)

        if not self.goto_zone("rescue", zone_id=self.order.zone):
            return self._abort("无法到达救援区")

        if self.skip_drone:
            self._unload_completed = True
            hold = float(mission.get("skip_drone_unload_hold_sec", 3.0))
            if hold > 0:
                rospy.sleep(hold)
        else:
            hold = float(mission.get("unload_hold_sec", 10.0))
            if hold > 0:
                rospy.sleep(hold)
            self._unload_completed = True
            comm = self.config.get("comm", {})
            notify_unload_done(
                remote_path=comm.get(
                    "unload_done_path", "/mnt/unload_done.flag"
                ),
                host=comm.get("drone_host", "192.168.31.110"),
                user=comm.get("drone_user", "root"),
                password=comm.get("drone_password", "123456"),
            )

        if not self.goto_zone("parking"):
            return self._abort("无法返回停车区")

        self.stop()
        rospy.loginfo("[Lar] 全部任务完成（开环+激光决策绕障）")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="lar：开环救援 + 应用层激光侧移绕障"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_TIMED_CONFIG),
        help="航段 YAML（默认 nav_rescue_2026_1.yaml）",
    )
    parser.add_argument(
        "--mission-config",
        type=str,
        default=str(_DEFAULT_MISSION_CONFIG),
        help="读取 obstacle_avoidance 的 mission_config.yaml",
    )
    parser.add_argument("--no-autostart", action="store_true")
    parser.add_argument("--skip-drone", action="store_true")
    parser.add_argument("--rescue", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument("--cmd-vel-topic", type=str, default=None)
    parser.add_argument(
        "--no-avoid",
        action="store_true",
        help="关闭激光决策，退化为纯开环计时",
    )
    args = parser.parse_args()
    NavLar(
        config_path=args.config,
        mission_config_path=args.mission_config,
        skip_drone=args.skip_drone,
        rescue_zone=args.rescue,
        cmd_vel_topic=args.cmd_vel_topic,
        autostart=not args.no_autostart,
        no_avoid=args.no_avoid,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
