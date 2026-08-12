#!/usr/bin/python3
# coding=UTF-8
"""nav_rescue_2026_1：转速（cmd_vel）× 时间 的纯开环控制版。

平移速度、旋转角速度、各航段运动时间均在 nav_rescue_2026_1.yaml 中由你填写：
  - timed.speed / timed.turn_speed
  - segment_motion.<段名>.duration_x / duration_y / duration_turn

用法：
  python3 nav_rescue_2026_1.py --skip-drone --rescue 3
  python3 nav_rescue_2026_1.py --config nav_rescue_2026_1.yaml --skip-drone --rescue 3
"""

from __future__ import print_function

import argparse
import math
import sys
from pathlib import Path

import rospy
import yaml
from geometry_msgs.msg import Twist

_HERE = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _HERE / "nav_rescue_2026_1.yaml"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from rescue_protocol import (  # noqa: E402
    RescueOrder,
    notify_loading_done,
    notify_unload_done,
    wait_for_delivery_done,
    wait_for_rescue_target,
)


def load_timed_config(path):
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


class TimedSpeedNav(object):
    """纯开环：你设的转速 × 你设的时间。"""

    def __init__(
        self,
        speed=0.3,
        turn_speed=0.5,
        cmd_vel_topic="/cmd_vel",
        control_rate=100,
        yaw_skip_rad=0.03,
        segment_pause_sec=0.30,
        log_prefix="[TimedNav]",
    ):
        self.speed = float(speed)
        self.turn_speed = float(turn_speed)
        self.yaw_skip_rad = float(yaw_skip_rad)
        self.segment_pause_sec = max(0.0, float(segment_pause_sec))
        self.log_prefix = log_prefix
        self._est = {"x": 0.0, "y": 0.0, "yaw": 0.0}

        self.pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        self.rate = rospy.Rate(int(control_rate))
        rospy.sleep(0.3)
        rospy.loginfo(
            "%s ready cmd_vel=%s speed=%.2f turn=%.2f (转速×时间，时间由配置指定)",
            self.log_prefix,
            cmd_vel_topic,
            self.speed,
            self.turn_speed,
        )

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

    def _drive_timed(self, twist, duration, label):
        duration = max(0.0, float(duration))
        if duration < 1e-4:
            return 0.0
        rospy.loginfo(
            "%s %s: vx=%.3f vy=%.3f wz=%.3f duration=%.3fs",
            self.log_prefix,
            label,
            twist.linear.x,
            twist.linear.y,
            twist.angular.z,
            duration,
        )
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            if rospy.Time.now().to_sec() - t0 >= duration:
                break
            self.pub.publish(twist)
            self.rate.sleep()
        self.stop()
        return min(rospy.Time.now().to_sec() - t0, duration)

    def go_linear_x_timed(self, linear_speed, duration):
        spd = float(linear_speed)
        duration = float(duration)
        if abs(spd) < 1e-6 or duration < 1e-4:
            return
        twist = Twist()
        twist.linear.x = spd
        self._drive_timed(twist, duration, "go_linear_x")
        signed = spd * duration
        yaw = self._est["yaw"]
        self._est["x"] += math.cos(yaw) * signed
        self._est["y"] += math.sin(yaw) * signed

    def go_linear_y_timed(self, linear_speed, duration):
        spd = float(linear_speed)
        duration = float(duration)
        if abs(spd) < 1e-6 or duration < 1e-4:
            return
        twist = Twist()
        twist.linear.y = spd
        self._drive_timed(twist, duration, "go_linear_y")
        signed = spd * duration
        yaw = self._est["yaw"]
        self._est["x"] += -math.sin(yaw) * signed
        self._est["y"] += math.cos(yaw) * signed

    def turn_ang_timed(self, ang_speed, duration, direction=1):
        spd = abs(float(ang_speed))
        duration = float(duration)
        if spd < 1e-6 or duration < 1e-4:
            return
        direction = 1 if direction >= 0 else -1
        twist = Twist()
        twist.angular.z = -spd if direction > 0 else spd
        self._drive_timed(twist, duration, "turn_ang")
        self._est["yaw"] = normalize_yaw(
            self._est["yaw"] + direction * spd * duration
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
            self.go_linear_y_timed(
                speed if body_y_sign > 0 else -speed, dy_t
            )
            if self.segment_pause_sec > 0:
                rospy.sleep(self.segment_pause_sec)

        if dx_t > 1e-4 and abs(body_x_sign) > 1e-9:
            self.go_linear_x_timed(
                speed if body_x_sign > 0 else -speed, dx_t
            )
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


class NavTimed(object):
    """救援任务：运动由 YAML 中的 speed + duration 驱动。"""

    def __init__(
        self,
        config_path=None,
        skip_drone=False,
        rescue_zone=2,
        cmd_vel_topic=None,
        autostart=True,
    ):
        rospy.init_node("nav_rescue_2026_timed", anonymous=True)
        self.skip_drone = bool(skip_drone)
        self.rescue_zone = int(rescue_zone)
        self.config_path = Path(config_path or _DEFAULT_CONFIG)
        self.config = load_timed_config(self.config_path)

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

        self.nav = TimedSpeedNav(
            speed=speed,
            turn_speed=turn_speed,
            cmd_vel_topic=topic,
            control_rate=int(timed.get("control_rate", 100)),
            yaw_skip_rad=float(timed.get("yaw_skip_rad", 0.03)),
            segment_pause_sec=float(timed.get("segment_pause_sec", 0.30)),
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
            "[Mission] nav_rescue_2026_1 | config=%s | skip_drone=%s "
            "rescue=%s | speed=%.2f turn=%.2f",
            self.config_path,
            self.skip_drone,
            self.rescue_zone,
            speed,
            turn_speed,
        )
        if autostart:
            ok = self.run_mission()
            rospy.loginfo("[Mission] 结束 ok=%s", ok)

    def stop(self):
        self.nav.stop()

    def _abort(self, reason):
        rospy.logerr("[Mission] abort: %s", reason)
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
            # 从哪个救援区返回，用对应的 to_parking_N
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
        # 返回停车区时带上当前救援区号，选用 to_parking_N
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
            "[Mission] goto %s via %s (%.3f, %.3f, yaw=%.3f) "
            "body≈(%.3f, %.3f) dyaw=%.1f°",
            label,
            key,
            pose["x"],
            pose["y"],
            pose["yaw"],
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
            rospy.loginfo("[Mission] 取货区等待 %.1fs", pickup_hold)
            rospy.sleep(pickup_hold)
        rospy.logwarn("[Mission] vision_grasp stub bypass")


        if not self.goto_zone("loading"):
            return self._abort("无法到达装货区")

        if self.skip_drone:
            self.order = RescueOrder(zone=self.rescue_zone, level=1)
            rospy.loginfo(
                "[Mission] 跳过无人机，救援区 zone=%d", self.order.zone
            )
        else:
            rospy.loginfo("等待无人机信号")
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
            rospy.loginfo("[Mission] 救援区 zone=%d", self.order.zone)

        if self.skip_drone:
            self._loading_completed = True
            hold = float(mission.get("skip_drone_loading_hold_sec", 3.0))
            if hold > 0:
                rospy.loginfo("[Mission] 跳过无人机：装货区保持 %.1fs", hold)
                rospy.sleep(hold)
        else:
            hold = float(mission.get("loading_hold_sec", 10.0))
            if hold > 0:
                rospy.loginfo("[Mission] 装货保持 %.1fs", hold)
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
                rospy.loginfo(
                    "[Mission] 前往救援区前后退 duration_x=%.2fs speed=%.2f",
                    float(seg.get("duration_x", 0.0)),
                    float(seg.get("speed", self.nav.speed)),
                )
                self.nav.run_backoff_x(seg)
                self.stop()
                rospy.sleep(0.3)

        if not self.goto_zone("rescue", zone_id=self.order.zone):
            return self._abort("无法到达救援区")

        if self.skip_drone:
            self._unload_completed = True
            hold = float(mission.get("skip_drone_unload_hold_sec", 3.0))
            if hold > 0:
                rospy.loginfo("[Mission] 跳过无人机：救援区保持 %.1fs", hold)
                rospy.sleep(hold)
        else:
            hold = float(mission.get("unload_hold_sec", 10.0))
            if hold > 0:
                rospy.loginfo("[Mission] 卸货保持 %.1fs", hold)
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
        rospy.loginfo("[Mission] 全部任务完成（转速×时间版）")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="nav_rescue_2026_1：转速×时间纯开环救援导航"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="参数配置 YAML（默认同目录 nav_rescue_2026_1.yaml）",
    )
    parser.add_argument("--no-autostart", action="store_true")
    parser.add_argument("--skip-drone", action="store_true")
    parser.add_argument(
        "--rescue", type=int, default=2, choices=(1, 2, 3, 4)
    )
    parser.add_argument("--cmd-vel-topic", type=str, default=None)
    args = parser.parse_args()
    NavTimed(
        config_path=args.config,
        skip_drone=args.skip_drone,
        rescue_zone=args.rescue,
        cmd_vel_topic=args.cmd_vel_topic,
        autostart=not args.no_autostart,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
