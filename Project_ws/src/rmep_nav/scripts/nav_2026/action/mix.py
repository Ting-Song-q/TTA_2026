#!/usr/bin/python3
# coding=UTF-8
"""mix：开环（转速×时间）为主，遇障切 YaLong 风格 move_base/TEB 闭环到点。

参考：
  - nav_rescue_2026_1.py：TimedSpeedNav + segment_motion 开环任务
  - action/closeloop.py：clear_costmaps → move_base/TEB → 等 SUCCEEDED

策略：
  1) 默认按 nav_rescue_2026_1.yaml 做开环段运动
  2) 开环发速时监视 /scan；触发紧急净空则停车
  3) 随后用 mission_config.yaml 的 map 系航点，走 move_base/TEB 到目标区

依赖启动栈：
  1) roscore
  2) roslaunch rmep_base rmep_base.launch
  3) roslaunch rmep_nav map_amcl_move.launch   # 闭环回退需要
  4) python3 action/mix.py --skip-drone --rescue 2

用法：
  python3 action/mix.py --skip-drone --rescue 2
  python3 action/mix.py --timed-config ../nav_rescue_2026_1.yaml --skip-drone
  python3 action/mix.py --openloop-only --skip-drone   # 强制纯开环（不切闭环）
  python3 action/mix.py --closedloop-only --skip-drone # 强制全程闭环
"""

from __future__ import print_function

import argparse
import math
import sys
from pathlib import Path

import rospy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from tf.transformations import quaternion_from_euler

_HERE = Path(__file__).resolve().parent
_NAV_DIR = _HERE.parent
if str(_NAV_DIR) not in sys.path:
    sys.path.insert(0, str(_NAV_DIR))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from rescue_protocol import (  # noqa: E402
    RescueOrder,
    wait_for_rescue_target,
    notify_loading_done,
    wait_for_delivery_done,
    notify_unload_done,
)
from config_loader import load_mission_config, as_pose  # noqa: E402
from laser_avoidance import is_emergency  # noqa: E402
from nav_rescue_2026_1 import (  # noqa: E402
    TimedSpeedNav,
    load_timed_config,
    normalize_yaw,
)
from closeloop import MoveBaseClosedLoopNav  # noqa: E402


class GuardedTimedNav(TimedSpeedNav):
    """开环 TimedSpeedNav + 激光急停监护。"""

    def __init__(self, avoidance_cfg=None, **kwargs):
        super(GuardedTimedNav, self).__init__(**kwargs)
        self.avoidance_cfg = dict(avoidance_cfg or {})
        self.laser_data = None
        self.obstacle_hit = False
        scan_topic = str(self.avoidance_cfg.get("scan_topic", "/scan"))
        rospy.Subscriber(scan_topic, LaserScan, self._on_scan, queue_size=1)
        rospy.loginfo(
            "%s laser guard on %s emergency=%.3f",
            self.log_prefix,
            scan_topic,
            float(self.avoidance_cfg.get("emergency_stop_distance", 0.05)),
        )

    def _on_scan(self, msg):
        self.laser_data = msg

    def reset_obstacle_flag(self):
        self.obstacle_hit = False

    def _drive_timed(self, twist, duration, label):
        """与 TimedSpeedNav 相同，但周期检查激光急停。"""
        if self.obstacle_hit:
            return 0.0
        duration = max(0.0, float(duration))
        if duration < 1e-4:
            return 0.0
        rospy.loginfo(
            "%s %s: vx=%.3f vy=%.3f wz=%.3f duration=%.3fs (guarded)",
            self.log_prefix,
            label,
            twist.linear.x,
            twist.linear.y,
            twist.angular.z,
            duration,
        )
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            elapsed = rospy.Time.now().to_sec() - t0
            if elapsed >= duration:
                break
            if is_emergency(
                self.laser_data,
                self.avoidance_cfg,
                vx=twist.linear.x,
                vy=twist.linear.y,
                wz=twist.angular.z,
            ):
                self.obstacle_hit = True
                rospy.logwarn(
                    "%s %s: laser emergency -> stop open-loop (t=%.2fs/%.2fs)",
                    self.log_prefix,
                    label,
                    elapsed,
                    duration,
                )
                self.stop()
                return elapsed
            self.pub.publish(twist)
            self.rate.sleep()
        self.stop()
        return min(rospy.Time.now().to_sec() - t0, duration)

    def go_linear_x_timed(self, linear_speed, duration):
        if self.obstacle_hit:
            return
        super(GuardedTimedNav, self).go_linear_x_timed(linear_speed, duration)

    def go_linear_y_timed(self, linear_speed, duration):
        if self.obstacle_hit:
            return
        super(GuardedTimedNav, self).go_linear_y_timed(linear_speed, duration)

    def turn_ang_timed(self, ang_speed, duration, direction=1):
        if self.obstacle_hit:
            return
        super(GuardedTimedNav, self).turn_ang_timed(
            ang_speed, duration, direction=direction
        )

    def run_segment(self, body_x_sign, body_y_sign, yaw_delta_sign, seg):
        if self.obstacle_hit:
            return
        super(GuardedTimedNav, self).run_segment(
            body_x_sign, body_y_sign, yaw_delta_sign, seg
        )

    def run_backoff_x(self, seg):
        if self.obstacle_hit:
            return
        super(GuardedTimedNav, self).run_backoff_x(seg)


class MixRescueNav(object):
    """救援任务：开环优先，遇障回退 move_base/TEB。"""

    def __init__(
        self,
        timed_config_path=None,
        mission_config_path=None,
        skip_drone=False,
        rescue_zone=2,
        cmd_vel_topic=None,
        move_base_name="move_base",
        clear_costmaps=True,
        mode="mix",
        autostart=True,
    ):
        """
        mode:
          - mix: 开环 + 遇障切闭环（默认）
          - openloop: 仅开环（仍激光急停，但不切闭环）
          - closedloop: 全程闭环（忽略 segment_motion）
        """
        rospy.init_node("nav_rescue_2026_mix", anonymous=True)

        self.mode = str(mode or "mix").strip().lower()
        if self.mode not in ("mix", "openloop", "closedloop"):
            rospy.logwarn("[Mix] unknown mode=%s, fallback mix", self.mode)
            self.mode = "mix"

        self.skip_drone = bool(skip_drone)
        self.rescue_zone = int(rescue_zone)
        self.timed_path = Path(timed_config_path or (_NAV_DIR / "nav_rescue_2026_1.yaml"))
        self.mission_path = Path(
            mission_config_path or (_NAV_DIR / "mission_config.yaml")
        )

        self.timed_cfg = load_timed_config(self.timed_path)
        self.mission_cfg_full = load_mission_config(self.mission_path)

        timed = dict(self.timed_cfg.get("timed") or {})
        self.mission_cfg = dict(self.timed_cfg.get("mission") or {})
        self.segment_motion = dict(self.timed_cfg.get("segment_motion") or {})
        self.arrive_pause_sec = float(timed.get("arrive_pause_sec", 0.50))
        self.avoidance_cfg = dict(
            self.mission_cfg_full.get("obstacle_avoidance") or {}
        )
        # 地图系航点：闭环回退用 mission_config.zones
        self.map_zones = dict(self.mission_cfg_full.get("zones") or {})
        # 开环方向航点：用 timed yaml 的 zones（与 1.py 一致）
        self.open_zones = dict(self.timed_cfg.get("zones") or {})

        topic = str(
            cmd_vel_topic
            if cmd_vel_topic
            else timed.get("cmd_vel_topic", "/cmd_vel")
        )
        self.cmd_vel_topic = topic

        speed = float(timed.get("speed", 0.3))
        turn_speed = float(timed.get("turn_speed", 0.5))
        self.open_nav = GuardedTimedNav(
            avoidance_cfg=self.avoidance_cfg,
            speed=speed,
            turn_speed=turn_speed,
            cmd_vel_topic=topic,
            control_rate=int(timed.get("control_rate", 100)),
            yaw_skip_rad=float(timed.get("yaw_skip_rad", 0.03)),
            segment_pause_sec=float(timed.get("segment_pause_sec", 0.30)),
            log_prefix="[MixOpen]",
        )

        parking_open = self.open_zones.get("parking") or {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
        }
        self.open_nav.set_est_pose(parking_open)

        self.closed_nav = None
        if self.mode in ("mix", "closedloop"):
            map_parking = as_pose(
                (self.map_zones.get("parking") or {"x": 0.0, "y": 0.0, "yaw": 0.0})
            )
            self._publish_initial_pose(
                map_parking["x"], map_parking["y"], map_parking["yaw"]
            )
            self.closed_nav = MoveBaseClosedLoopNav(
                cmd_vel_topic=topic,
                move_base_name=move_base_name,
                frame_id="map",
                clear_costmaps=clear_costmaps,
                log_prefix="[MixTEB]",
            )

        self.order = None
        self._loading_completed = False
        self._unload_completed = False
        self._current_zone = None
        self._current_zone_id = None
        self._fallback_count = 0

        rospy.loginfo(
            "[Mix] mode=%s timed=%s mission=%s skip_drone=%s rescue=%s",
            self.mode,
            self.timed_path.name,
            self.mission_path.name,
            self.skip_drone,
            self.rescue_zone,
        )
        if autostart:
            ok = self.run_mission()
            rospy.loginfo(
                "[Mix] 结束 ok=%s fallback_count=%d", ok, self._fallback_count
            )

    @staticmethod
    def _publish_initial_pose(x, y, yaw, hold_sec=0.6):
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
        t0 = rospy.Time.now()
        while (rospy.Time.now() - t0).to_sec() < float(hold_sec):
            msg.header.stamp = rospy.Time.now()
            pub.publish(msg)
            rate.sleep()
        rospy.sleep(0.5)
        rospy.loginfo(
            "[Mix] /initialpose (%.3f, %.3f, yaw=%.3f)", x, y, yaw
        )

    def stop(self):
        self.open_nav.stop()
        if self.closed_nav is not None:
            self.closed_nav.stop()

    def _abort(self, reason):
        rospy.logerr("[Mix] abort: %s", reason)
        self.stop()
        return False

    def _open_zone_pose(self, zone_name, zone_id=None):
        zones = self.open_zones
        if zone_name == "rescue":
            return zones["rescue"][int(zone_id)]
        return zones[zone_name]

    def _map_zone_pose(self, zone_name, zone_id=None):
        zones = self.map_zones
        if zone_name == "rescue":
            return as_pose(zones["rescue"][int(zone_id)])
        return as_pose(zones[zone_name])

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

    def _closed_fallback(self, zone_name, zone_id, label):
        if self.closed_nav is None:
            rospy.logerr("[Mix] 需闭环回退但 closed_nav 未初始化")
            return False
        self._fallback_count += 1
        map_pose = self._map_zone_pose(zone_name, zone_id)
        rospy.logwarn(
            "[Mix] #%d 开环遇障 -> TEB 闭环到 %s (%.3f, %.3f, yaw=%.3f)",
            self._fallback_count,
            label,
            map_pose["x"],
            map_pose["y"],
            map_pose["yaw"],
        )
        # 避免与开环抢 /cmd_vel
        self.open_nav.stop()
        ok = self.closed_nav.go_pose(map_pose, label="mix_" + label, clear=True)
        if ok:
            # 闭环成功后，用开环估计航点同步（方向一致性）
            self.open_nav.set_est_pose(self._open_zone_pose(zone_name, zone_id))
        return ok

    def goto_zone(self, zone_name, zone_id=None):
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

        # 全程闭环
        if self.mode == "closedloop":
            if self.closed_nav is None:
                return self._abort("closedloop 模式需要 map_amcl_move")
            map_pose = self._map_zone_pose(zone_name, zone_id)
            rospy.loginfo("[Mix] closedloop -> %s", label)
            if not self.closed_nav.go_pose(map_pose, label=label, clear=True):
                return False
            self.open_nav.set_est_pose(self._open_zone_pose(zone_name, zone_id))
            self._current_zone = zone_name
            self._current_zone_id = zone_id
            self.stop()
            if self.arrive_pause_sec > 0:
                rospy.sleep(self.arrive_pause_sec)
            return True

        # 开环（mix / openloop）
        key = self._segment_key(zone_name, key_zone_id)
        seg = self.segment_motion.get(key)
        if not seg:
            return self._abort("配置缺少 segment_motion.%s" % key)

        open_pose = self._open_zone_pose(zone_name, zone_id)
        from_pose = self.open_nav.get_est_pose()
        sx, sy, st, body_x, body_y, dyaw = self._signs_from_poses(
            from_pose, open_pose
        )
        rospy.loginfo(
            "[Mix] openloop goto %s via %s body≈(%.3f, %.3f) dyaw=%.1f°",
            label,
            key,
            body_x,
            body_y,
            math.degrees(dyaw),
        )

        # 开环前取消可能残留的 move_base 目标
        if self.closed_nav is not None:
            try:
                self.closed_nav.move_base.cancel_all_goals()
            except Exception:
                pass
            self.closed_nav.stop()

        self.open_nav.reset_obstacle_flag()
        self.open_nav.run_segment(sx, sy, st, seg)

        if self.open_nav.obstacle_hit:
            if self.mode == "openloop":
                return self._abort(
                    "开环遇障且 --openloop-only，放弃 %s" % label
                )
            if not self._closed_fallback(zone_name, zone_id, label):
                return self._abort("闭环回退失败: %s" % label)
        else:
            self.open_nav.set_est_pose(open_pose)

        self._current_zone = zone_name
        self._current_zone_id = zone_id
        self.stop()
        if self.arrive_pause_sec > 0:
            rospy.sleep(self.arrive_pause_sec)
        return True

    def run_mission(self):
        timeouts = self.timed_cfg.get("timeouts", {})
        mission = self.mission_cfg
        comm = self.timed_cfg.get("comm", {})

        if not self.goto_zone("pickup"):
            return self._abort("无法到达取货区")
        pickup_hold = float(mission.get("pickup_hold_sec", 1.0))
        if pickup_hold > 0:
            rospy.loginfo("[Mix] 取货区等待 %.1fs", pickup_hold)
            rospy.sleep(pickup_hold)
        rospy.logwarn("[Mix] vision_grasp stub bypass")

        if not self.goto_zone("loading"):
            return self._abort("无法到达装货区")

        if self.skip_drone:
            self.order = RescueOrder(zone=self.rescue_zone, level=1)
            rospy.loginfo("[Mix] 跳过无人机，救援区 zone=%d", self.order.zone)
        else:
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
            rospy.loginfo("[Mix] 救援区 zone=%d", self.order.zone)

        if self.skip_drone:
            self._loading_completed = True
            hold = float(mission.get("skip_drone_loading_hold_sec", 3.0))
            if hold > 0:
                rospy.loginfo("[Mix] 跳过无人机：装货区保持 %.1fs", hold)
                rospy.sleep(hold)
        else:
            hold = float(mission.get("loading_hold_sec", 10.0))
            if hold > 0:
                rospy.loginfo("[Mix] 装货保持 %.1fs", hold)
                rospy.sleep(hold)
            self._loading_completed = True
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
            if seg and self.mode != "closedloop":
                self.open_nav.reset_obstacle_flag()
                self.open_nav.run_backoff_x(seg)
                self.stop()
                if self.open_nav.obstacle_hit and self.mode == "mix":
                    rospy.logwarn("[Mix] backoff 遇障，忽略并继续去救援区")
                rospy.sleep(0.3)

        if not self.goto_zone("rescue", zone_id=self.order.zone):
            return self._abort("无法到达救援区")

        if self.skip_drone:
            self._unload_completed = True
            hold = float(mission.get("skip_drone_unload_hold_sec", 3.0))
            if hold > 0:
                rospy.loginfo("[Mix] 跳过无人机：救援区保持 %.1fs", hold)
                rospy.sleep(hold)
        else:
            hold = float(mission.get("unload_hold_sec", 10.0))
            if hold > 0:
                rospy.loginfo("[Mix] 卸货保持 %.1fs", hold)
                rospy.sleep(hold)
            self._unload_completed = True
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
        rospy.loginfo(
            "[Mix] 全部任务完成 mode=%s fallback=%d",
            self.mode,
            self._fallback_count,
        )
        return True


def main():
    parser = argparse.ArgumentParser(
        description="mix：开环优先，遇障切 move_base/TEB 闭环"
    )
    parser.add_argument(
        "--timed-config",
        type=str,
        default="",
        help="开环 YAML（默认 nav_2026/nav_rescue_2026_1.yaml）",
    )
    parser.add_argument(
        "--mission-config",
        type=str,
        default="",
        help="地图航点/避障 YAML（默认 nav_2026/mission_config.yaml）",
    )
    parser.add_argument("--no-autostart", action="store_true")
    parser.add_argument("--skip-drone", action="store_true")
    parser.add_argument(
        "--rescue", type=int, default=2, choices=(1, 2, 3, 4)
    )
    parser.add_argument("--cmd-vel-topic", type=str, default=None)
    parser.add_argument("--move-base-name", type=str, default="move_base")
    parser.add_argument(
        "--no-clear-costmaps",
        action="store_true",
        help="闭环回退时不 clear_costmaps",
    )
    parser.add_argument(
        "--openloop-only",
        action="store_true",
        help="仅开环（遇障则失败，不切 TEB）",
    )
    parser.add_argument(
        "--closedloop-only",
        action="store_true",
        help="全程闭环（忽略 segment_motion）",
    )
    args = parser.parse_args()

    mode = "mix"
    if args.openloop_only and args.closedloop_only:
        rospy.logerr("不能同时 --openloop-only 与 --closedloop-only")
        sys.exit(2)
    if args.openloop_only:
        mode = "openloop"
    elif args.closedloop_only:
        mode = "closedloop"

    MixRescueNav(
        timed_config_path=args.timed_config or None,
        mission_config_path=args.mission_config or None,
        skip_drone=args.skip_drone,
        rescue_zone=args.rescue,
        cmd_vel_topic=args.cmd_vel_topic,
        move_base_name=args.move_base_name,
        clear_costmaps=not args.no_clear_costmaps,
        mode=mode,
        autostart=not args.no_autostart,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
