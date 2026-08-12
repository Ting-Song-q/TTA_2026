#!/usr/bin/python3
# coding=UTF-8
"""stop.py：开环指令的激光守门（急停/失效闭锁，不侧移绕障）。

参考：
  - nav_rescue_2026_1.py：上游发转速×时间开环速度
  - closeloop.py：安全侧先清状态再放行；开环对应为 fail-closed 守门
  - laser_avoidance：车体净空急停判定

数据流：
  nav_rescue_2026_1.py --cmd-vel-topic /cmd_vel_nav
       ↓
  stop.py  （订 /cmd_vel_nav + /scan）
       ↓
  /cmd_vel → rmep_base

策略（不做 TEB/侧移绕障）：
  - 激光缺失/过期且 fail_closed → 零速
  - 运动方向净空 < emergency_stop_distance → 零速
  - 否则透传上游 Twist

用法：
  roslaunch rmep_base rmep_base.launch
  python3 action/stop.py
  python3 nav_rescue_2026_1.py --skip-drone --rescue 2 --cmd-vel-topic /cmd_vel_nav
"""

from __future__ import print_function

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

_HERE = Path(__file__).resolve().parent
_NAV_DIR = _HERE.parent
if str(_NAV_DIR) not in sys.path:
    sys.path.insert(0, str(_NAV_DIR))

from config_loader import load_mission_config  # noqa: E402
from laser_avoidance import get_clearances, is_emergency  # noqa: E402


def _make_twist(vx=0.0, vy=0.0, wz=0.0):
    tw = Twist()
    tw.linear.x = float(vx)
    tw.linear.y = float(vy)
    tw.angular.z = float(wz)
    return tw


def _scan_validity(scan, cfg):
    """轻量激光有效性检查（与 avoidance fail-closed 同思路）。"""
    if scan is None:
        return False, "scan_unavailable"
    ranges = getattr(scan, "ranges", None) or []
    if not ranges:
        return False, "scan_empty"
    if abs(float(getattr(scan, "angle_increment", 0.0))) < 1e-9:
        return False, "angle_increment_invalid"
    min_pts = int(cfg.get("min_valid_scan_points", 30))
    valid = 0
    rmin = float(getattr(scan, "range_min", 0.0))
    rmax = float(getattr(scan, "range_max", 100.0))
    for r in ranges:
        if not math.isfinite(r):
            continue
        if r < rmin or r > rmax:
            continue
        valid += 1
        if valid >= min_pts:
            return True, "ok"
    return False, "valid_points_insufficient"


class _ClearanceMedian(object):
    """最近若干帧四向净空中值，抗雷达尖峰。"""

    def __init__(self, cfg):
        self.cfg = cfg
        n = max(1, int(cfg.get("distance_filter_window", 5)))
        self._bufs = {
            k: deque(maxlen=n) for k in ("front", "back", "left", "right")
        }

    def reset(self):
        for buf in self._bufs.values():
            buf.clear()

    def update(self, scan):
        if scan is None:
            return False
        c = get_clearances(scan, self.cfg)
        for k, buf in self._bufs.items():
            v = float(c[k])
            if math.isfinite(v):
                buf.append(v)
        return True

    def ready(self):
        need = int(self.cfg.get("distance_filter_min_samples", 3))
        return all(len(b) >= need for b in self._bufs.values())

    def clearances(self):
        out = {}
        for k, buf in self._bufs.items():
            if not buf:
                out[k] = float("inf")
            else:
                s = sorted(buf)
                out[k] = s[len(s) // 2]
        return out


class OpenLoopStopGate(object):
    """开环速度守门：透传或急停，不改写绕障轨迹。"""

    def __init__(
        self,
        config=None,
        scan_topic="/scan",
        nav_cmd_topic="/cmd_vel_nav",
        output_cmd_topic="/cmd_vel",
        control_rate=20.0,
        log_prefix="[StopGate]",
    ):
        self._log = log_prefix
        self.config = dict(config or {})
        self.config.setdefault("enabled", True)
        self.config.setdefault("fail_closed", True)
        self.config.setdefault("max_scan_age", 0.30)
        self.config.setdefault("max_cmd_age", 0.50)
        self.config.setdefault("lidar_mount", "rear")

        self.history = _ClearanceMedian(self.config)
        self.scan = None
        self.scan_received_at = None
        self.latest_command = Twist()
        self.command_received_at = None
        self.last_mode = None

        self.cmd_pub = rospy.Publisher(output_cmd_topic, Twist, queue_size=1)
        self.scan_sub = rospy.Subscriber(
            scan_topic, LaserScan, self._on_scan, queue_size=1
        )
        self.nav_sub = rospy.Subscriber(
            nav_cmd_topic, Twist, self._on_nav_cmd, queue_size=1
        )

        rate = max(1.0, float(control_rate))
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self._on_timer)
        rospy.on_shutdown(self.shutdown_stop)

        rospy.loginfo(
            "%s ready: %s -> %s | scan=%s | fail_closed=%s | emergency=%.3fm",
            self._log,
            nav_cmd_topic,
            output_cmd_topic,
            scan_topic,
            self.config.get("fail_closed", True),
            float(self.config.get("emergency_stop_distance", 0.05)),
        )

    def _on_scan(self, msg):
        now = time.monotonic()
        max_age = float(self.config.get("max_scan_age", 0.30))
        if self.scan_received_at is not None:
            gap = now - self.scan_received_at
            if gap < 0.0 or gap > max_age:
                self.history.reset()
        self.scan = msg
        self.scan_received_at = now
        if not self.history.update(msg):
            self.history.reset()

    def _on_nav_cmd(self, msg):
        self.latest_command = msg
        self.command_received_at = time.monotonic()

    def _scan_ok(self, now):
        valid, reason = _scan_validity(self.scan, self.config)
        if not valid:
            return False, reason
        max_age = float(self.config.get("max_scan_age", 0.30))
        if self.scan_received_at is None:
            return False, "scan_timestamp_unavailable"
        if now - self.scan_received_at > max_age:
            return False, "scan_stale"
        if not self.history.ready():
            return False, "scan_history_warming"
        return True, "ok"

    def gate(self, command, now=None):
        now = time.monotonic() if now is None else now
        vx = float(command.linear.x)
        vy = float(command.linear.y)
        wz = float(command.angular.z)
        moving = abs(vx) > 1e-4 or abs(vy) > 1e-4 or abs(wz) > 1e-4

        if not self.config.get("enabled", True):
            return _make_twist(vx, vy, wz), "disabled"

        healthy, reason = self._scan_ok(now)
        if not healthy:
            if self.config.get("fail_closed", True) and moving:
                return _make_twist(), reason
            return _make_twist(vx, vy, wz), reason

        # 当前帧急停
        if is_emergency(self.scan, self.config, vx=vx, vy=vy, wz=wz):
            return _make_twist(), "emergency_stop"

        # 多帧中值急停（抗尖峰）：用历史净空临时替换单帧判定
        hist = self.history.clearances()
        emerg = float(self.config.get("emergency_stop_distance", 0.05))
        checks = []
        if vx > 0.01:
            checks.append(hist["front"])
        elif vx < -0.01:
            checks.append(hist["back"])
        if vy > 0.01:
            checks.append(hist["left"])
        elif vy < -0.01:
            checks.append(hist["right"])
        if abs(wz) > 0.01:
            checks.append(hist["right"] if wz > 0 else hist["left"])
        if checks and min(checks) < emerg:
            return _make_twist(), "emergency_stop"

        return _make_twist(vx, vy, wz), "pass"

    def _on_timer(self, _event):
        now = time.monotonic()
        max_cmd_age = float(self.config.get("max_cmd_age", 0.50))

        if self.command_received_at is None:
            out, mode = _make_twist(), "cmd_unavailable"
        elif now - self.command_received_at > max_cmd_age:
            out, mode = _make_twist(), "cmd_stale"
        else:
            out, mode = self.gate(self.latest_command, now=now)

        self.cmd_pub.publish(out)
        self._report_mode(mode)

        if mode == "pass" and self.scan is not None:
            c = get_clearances(self.scan, self.config)
            rospy.logdebug_throttle(
                2.0,
                "%s pass clearances F=%.2f B=%.2f L=%.2f R=%.2f",
                self._log,
                c["front"],
                c["back"],
                c["left"],
                c["right"],
            )

    def _report_mode(self, mode):
        if mode == self.last_mode:
            return
        self.last_mode = mode
        stop_modes = {
            "scan_unavailable",
            "scan_empty",
            "angle_increment_invalid",
            "valid_points_insufficient",
            "scan_timestamp_unavailable",
            "scan_stale",
            "scan_history_warming",
            "emergency_stop",
            "cmd_unavailable",
            "cmd_stale",
        }
        if mode in stop_modes:
            rospy.logwarn("%s STOP output, reason=%s", self._log, mode)
        else:
            rospy.loginfo("%s mode=%s", self._log, mode)

    def shutdown_stop(self):
        for _ in range(5):
            self.cmd_pub.publish(_make_twist())
            time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser(
        description="开环守门：/cmd_vel_nav + /scan → 安全 /cmd_vel（急停透传）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="mission_config.yaml；默认 nav_2026/mission_config.yaml",
    )
    parser.add_argument("--scan-topic", type=str, default="")
    parser.add_argument("--nav-cmd-topic", type=str, default="")
    parser.add_argument("--output-cmd-topic", type=str, default="")
    parser.add_argument("--control-rate", type=float, default=0.0)
    parser.add_argument(
        "--no-fail-closed",
        action="store_true",
        help="激光异常时仍透传（不推荐）",
    )
    args = parser.parse_args()

    rospy.init_node("openloop_stop_gate", anonymous=True)

    config_path = Path(args.config) if args.config else (_NAV_DIR / "mission_config.yaml")
    mission = load_mission_config(config_path)
    oa = dict(mission.get("obstacle_avoidance") or {})
    if args.no_fail_closed:
        oa["fail_closed"] = False

    scan_topic = args.scan_topic or oa.get("scan_topic") or "/scan"
    nav_cmd = args.nav_cmd_topic or oa.get("nav_cmd_topic") or "/cmd_vel_nav"
    out_cmd = args.output_cmd_topic or oa.get("output_cmd_topic") or "/cmd_vel"
    rate = args.control_rate or float(oa.get("control_rate", 20))

    OpenLoopStopGate(
        config=oa,
        scan_topic=scan_topic,
        nav_cmd_topic=nav_cmd,
        output_cmd_topic=out_cmd,
        control_rate=rate,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
