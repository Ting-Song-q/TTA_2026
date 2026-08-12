#!/usr/bin/python3
# coding=utf8
"""nav_rescue_armpi：ArmPi Pro 适配版（编码器里程计闭环 + 可选开环回退）。

相对开环版：
  - 默认按 zones 位姿误差闭环：发 SetVelocity → 读里程计 → 误差进阈值再停
  - 里程计优先订阅 /odom；若无则用指令速度积分（编码器电机跟速后的软里程计）
  - closed_loop.enable=false 时可回退到 duration_* 开环

前置：
  1) 已 source armpi_pro 工作空间
  2) chassis_control 节点已启动
  3) 若有独立 odom 节点，在 YAML 中配置 odom_topic

用法：
  python3 nav_rescue_armpi.py --skip-drone --rescue 3
  python3 nav_rescue_armpi.py --config nav_rescue_armpi.yaml --skip-drone --rescue 2
"""

from __future__ import print_function

import argparse
import math
import sys
import threading
from pathlib import Path

import rospy
import yaml
from chassis_control.msg import SetVelocity
from nav_msgs.msg import Odometry

# ---------------------------------------------------------------------------
# 路径与依赖：保证同目录 yaml / rescue_protocol 可被找到
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _HERE / "nav_rescue_armpi.yaml"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# 无人机 SSH 握手（--skip-drone 时主流程不会真正用到 fabric2）
from rescue_protocol import (  # noqa: E402
    RescueOrder,
    notify_loading_done,
    notify_unload_done,
    wait_for_delivery_done,
    wait_for_rescue_target,
)


def load_timed_config(path):
    """读取 YAML 配置，返回 dict。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("配置文件不存在: %s" % path)
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError("配置文件格式错误: %s" % path)
    return data


def normalize_yaw(yaw):
    """把航向角归一化到 (-pi, pi]。"""
    while yaw > math.pi:
        yaw -= 2.0 * math.pi
    while yaw < -math.pi:
        yaw += 2.0 * math.pi
    return yaw


def yaw_from_quaternion(q):
    """geometry_msgs/Quaternion → yaw (rad)。"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def body_xy_to_set_velocity(vx_mps, vy_mps):
    """车体 vx/vy (m/s) → SetVelocity(velocity_mm/s, direction_deg, angular=0)。

    车体坐标（导航 zones / 段运动用）：
      +X = 前进，+Y = 左移

    ArmPi SetVelocity 方向角（与 basic_movement 一致）：
      90°  → 前进      180° → 左移
      270° → 后退        0° → 右移

    换算：direction = (90 + atan2(vy, vx)_deg) mod 360
    注意：切勿直接用 atan2(vy,vx) 当 direction，否则前进会变成侧移。
    底层 SetVelocity.velocity 单位是 mm/s，故这里 ×1000。
    """
    vx = float(vx_mps)
    vy = float(vy_mps)
    speed_mps = math.hypot(vx, vy)
    if speed_mps < 1e-9:
        return 0.0, 0.0, 0.0
    direction = (90.0 + math.degrees(math.atan2(vy, vx))) % 360.0
    return speed_mps * 1000.0, direction, 0.0


class TimedSpeedNavArmPi(object):
    """底盘控制：默认里程计闭环；可回退到速度×时间开环。

    单位：
      speed      — 平移 m/s（YAML/逻辑用；发底盘时转成 mm/s）
      turn_speed — ArmPi SetVelocity.angular（非 rad/s，demo 常用约 0.3~2）
    """

    def __init__(
        self,
        speed=0.06,
        turn_speed=2.0,
        set_velocity_topic="/chassis_control/set_velocity",
        segment_pause_sec=0.30,
        closed_loop=None,
        log_prefix="[TimedNavArmPi]",
    ):
        self.speed = float(speed)  # m/s
        self.turn_speed = float(turn_speed)
        self.segment_pause_sec = max(0.0, float(segment_pause_sec))
        self.log_prefix = log_prefix

        cl = dict(closed_loop or {})
        self.closed_loop_enable = bool(cl.get("enable", True))
        self.odom_topic = str(cl.get("odom_topic", "/odom"))
        self.xy_tolerance = float(cl.get("xy_tolerance", 0.03))  # m
        self.yaw_tolerance = float(cl.get("yaw_tolerance", 0.08))  # rad ≈4.5°
        self.control_period = float(cl.get("control_period", 0.05))
        self.timeout_margin = float(cl.get("timeout_margin", 1.8))
        # 指令积分标定：实际位移 ≈ cmd * scale * dt
        self.linear_scale = float(cl.get("linear_scale", 1.0))
        # ArmPi angular → 估计 rad/s（官方约 angular=0.3 转半圈≈7.5s → ~1.4）
        self.angular_to_radps = float(cl.get("angular_to_radps", 1.4))
        self.min_cmd_speed = float(cl.get("min_cmd_speed", 0.04))
        self.odom_stale_sec = float(cl.get("odom_stale_sec", 0.5))

        self._lock = threading.Lock()
        self._est = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self._cmd_vx = 0.0
        self._cmd_vy = 0.0
        self._cmd_wz = 0.0  # rad/s 估计，供无 /odom 时积分
        self._last_integ_time = None
        self._odom_from_topic = False
        self._last_odom_stamp = None

        self.pub = rospy.Publisher(set_velocity_topic, SetVelocity, queue_size=1)
        self._odom_sub = None
        if self.closed_loop_enable:
            self._odom_sub = rospy.Subscriber(
                self.odom_topic, Odometry, self._odom_cb, queue_size=1
            )
            self._integ_timer = rospy.Timer(
                rospy.Duration(self.control_period), self._integ_timer_cb
            )
        else:
            self._integ_timer = None

        # 等待 publisher 与 chassis_control 建连
        rospy.sleep(0.5)
        rospy.loginfo(
            "%s ready topic=%s speed=%.3fm/s (≈%.0fmm/s) turn=%.2f "
            "closed_loop=%s odom=%s",
            self.log_prefix,
            set_velocity_topic,
            self.speed,
            self.speed * 1000.0,
            self.turn_speed,
            self.closed_loop_enable,
            self.odom_topic if self.closed_loop_enable else "n/a",
        )

    def _odom_cb(self, msg):
        """外部里程计优先：用编码器/融合 odom 覆盖估计位姿。"""
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        with self._lock:
            self._est = {"x": x, "y": y, "yaw": normalize_yaw(yaw)}
            self._odom_from_topic = True
            self._last_odom_stamp = rospy.Time.now()
            self._last_integ_time = rospy.Time.now()

    def _integ_timer_cb(self, _event):
        """无新鲜 /odom 时，用指令速度积分（软里程计）。"""
        if not self.closed_loop_enable:
            return
        now = rospy.Time.now()
        with self._lock:
            if self._odom_from_topic and self._last_odom_stamp is not None:
                age = (now - self._last_odom_stamp).to_sec()
                if age <= self.odom_stale_sec:
                    self._last_integ_time = now
                    return
            if self._last_integ_time is None:
                self._last_integ_time = now
                return
            dt = (now - self._last_integ_time).to_sec()
            self._last_integ_time = now
            if dt <= 0.0 or dt > 0.5:
                return
            vx = self._cmd_vx * self.linear_scale
            vy = self._cmd_vy * self.linear_scale
            wz = self._cmd_wz
            yaw = self._est["yaw"]
            self._est["x"] += (math.cos(yaw) * vx - math.sin(yaw) * vy) * dt
            self._est["y"] += (math.sin(yaw) * vx + math.cos(yaw) * vy) * dt
            self._est["yaw"] = normalize_yaw(yaw + wz * dt)

    def get_est_pose(self):
        """返回当前估计位姿副本（米 / rad）。"""
        with self._lock:
            return dict(self._est)

    def set_est_pose(self, pose):
        """强制设置估计位姿（到点后对齐航点，便于下一段算符号）。"""
        with self._lock:
            self._est = {
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "yaw": float(pose.get("yaw", 0.0)),
            }
            self._last_integ_time = rospy.Time.now()

    def stop(self):
        """停车。chassis_control 缓变线程忙时可能丢包，故连发多次零速度。"""
        with self._lock:
            self._cmd_vx = 0.0
            self._cmd_vy = 0.0
            self._cmd_wz = 0.0
        for _ in range(3):
            self.pub.publish(SetVelocity(velocity=0.0, direction=0.0, angular=0.0))
            rospy.sleep(0.05)

    def _publish_drive(self, velocity, direction, angular, label):
        """发布一次底盘指令，并更新积分用的车体速度估计。"""
        # ArmPi：angular>0 约顺时针；逻辑 yaw 增大方向对应发负 angular
        with self._lock:
            if abs(velocity) < 1e-6:
                self._cmd_vx = 0.0
                self._cmd_vy = 0.0
            else:
                # direction: 90=前(+x), 180=左(+y), 270=后(-x), 0=右(-y)
                rad = math.radians(float(direction))
                # SetVelocity 方向角相对「前进=90°」
                body_angle = rad - math.pi / 2.0
                spd = float(velocity) / 1000.0  # mm/s → m/s
                self._cmd_vx = spd * math.cos(body_angle)
                self._cmd_vy = spd * math.sin(body_angle)
            # 发负 angular 时 yaw 增大 → wz = -angular * scale
            self._cmd_wz = -float(angular) * self.angular_to_radps
        rospy.loginfo(
            "%s %s: v=%.1fmm/s dir=%.1f ang=%.2f",
            self.log_prefix,
            label,
            velocity,
            direction,
            angular,
        )
        self.pub.publish(
            SetVelocity(
                velocity=float(velocity),
                direction=float(direction),
                angular=float(angular),
            )
        )

    def _drive_timed(self, velocity, direction, angular, duration, label):
        """开环：发一次指令 → 睡满 duration → 停车。"""
        duration = max(0.0, float(duration))
        if duration < 1e-4:
            return 0.0
        rospy.loginfo(
            "%s %s timed: v=%.1fmm/s dir=%.1f ang=%.2f duration=%.3fs",
            self.log_prefix,
            label,
            velocity,
            direction,
            angular,
            duration,
        )
        self._publish_drive(velocity, direction, angular, label)
        rospy.sleep(duration)
        self.stop()
        return duration

    def go_linear_x_timed(self, linear_speed_mps, duration):
        """车体前后平移（开环）。"""
        spd = float(linear_speed_mps)
        duration = float(duration)
        if abs(spd) < 1e-9 or duration < 1e-4:
            return
        velocity, direction, angular = body_xy_to_set_velocity(spd, 0.0)
        self._drive_timed(velocity, direction, angular, duration, "go_linear_x")
        signed_m = spd * duration
        pose = self.get_est_pose()
        yaw = pose["yaw"]
        self.set_est_pose(
            {
                "x": pose["x"] + math.cos(yaw) * signed_m,
                "y": pose["y"] + math.sin(yaw) * signed_m,
                "yaw": yaw,
            }
        )

    def go_linear_y_timed(self, linear_speed_mps, duration):
        """车体左右平移（开环）。"""
        spd = float(linear_speed_mps)
        duration = float(duration)
        if abs(spd) < 1e-9 or duration < 1e-4:
            return
        velocity, direction, angular = body_xy_to_set_velocity(0.0, spd)
        self._drive_timed(velocity, direction, angular, duration, "go_linear_y")
        signed_m = spd * duration
        pose = self.get_est_pose()
        yaw = pose["yaw"]
        self.set_est_pose(
            {
                "x": pose["x"] + -math.sin(yaw) * signed_m,
                "y": pose["y"] + math.cos(yaw) * signed_m,
                "yaw": yaw,
            }
        )

    def turn_ang_timed(self, ang_speed, duration, direction=1):
        """原地转向（开环）。direction>0 表示朝目标 yaw 增大的一侧。"""
        spd = abs(float(ang_speed))
        duration = float(duration)
        if spd < 1e-6 or duration < 1e-4:
            return
        direction = 1 if direction >= 0 else -1
        angular = -spd if direction > 0 else spd
        self._drive_timed(0.0, 0.0, angular, duration, "turn_ang")
        pose = self.get_est_pose()
        self.set_est_pose(
            {
                "x": pose["x"],
                "y": pose["y"],
                "yaw": normalize_yaw(
                    pose["yaw"] + direction * abs(angular) * duration
                ),
            }
        )

    def _wait_body_axis(self, axis, target_m, speed_mps, timeout, label):
        """沿车体单轴闭环平移，直到位移达到目标或超时。"""
        target_m = float(target_m)
        if abs(target_m) < self.xy_tolerance:
            return True
        speed = abs(float(speed_mps))
        if speed < 1e-9:
            return False
        sign = 1.0 if target_m > 0 else -1.0
        cmd = max(speed, self.min_cmd_speed) * sign
        start = self.get_est_pose()
        yaw0 = start["yaw"]
        if axis == "x":
            velocity, direction, _ = body_xy_to_set_velocity(cmd, 0.0)
        else:
            velocity, direction, _ = body_xy_to_set_velocity(0.0, cmd)
        self._publish_drive(velocity, direction, 0.0, label)
        t0 = rospy.Time.now()
        ok = False
        while not rospy.is_shutdown():
            if (rospy.Time.now() - t0).to_sec() > timeout:
                rospy.logwarn("%s %s timeout (%.1fs)", self.log_prefix, label, timeout)
                break
            pose = self.get_est_pose()
            dx = pose["x"] - start["x"]
            dy = pose["y"] - start["y"]
            # 世界位移投影回出发时车体轴
            if axis == "x":
                traveled = math.cos(yaw0) * dx + math.sin(yaw0) * dy
            else:
                traveled = -math.sin(yaw0) * dx + math.cos(yaw0) * dy
            remain = target_m - traveled
            if abs(remain) <= self.xy_tolerance or remain * target_m <= 0.0:
                ok = True
                break
            rospy.sleep(self.control_period)
        self.stop()
        return ok

    def _wait_turn_delta(self, delta_yaw, turn_speed, timeout, label):
        """原地闭环转向，直到转过 delta_yaw（最短方向）或超时。"""
        delta_yaw = normalize_yaw(float(delta_yaw))
        if abs(delta_yaw) < self.yaw_tolerance:
            return True
        spd = abs(float(turn_speed))
        if spd < 1e-6:
            return False
        # direction>0 → yaw 增大 → 发负 angular
        direction = 1 if delta_yaw > 0 else -1
        angular = -spd if direction > 0 else spd
        start_yaw = self.get_est_pose()["yaw"]
        target_yaw = normalize_yaw(start_yaw + delta_yaw)
        self._publish_drive(0.0, 0.0, angular, label)
        t0 = rospy.Time.now()
        ok = False
        while not rospy.is_shutdown():
            if (rospy.Time.now() - t0).to_sec() > timeout:
                rospy.logwarn("%s %s timeout (%.1fs)", self.log_prefix, label, timeout)
                break
            err = normalize_yaw(target_yaw - self.get_est_pose()["yaw"])
            if abs(err) <= self.yaw_tolerance or err * delta_yaw <= 0.0:
                ok = True
                break
            rospy.sleep(self.control_period)
        self.stop()
        return ok

    def run_segment(self, body_x_sign, body_y_sign, yaw_delta_sign, seg,
                    body_x=0.0, body_y=0.0, dyaw=0.0):
        """执行一段航程：先横移(Y) → 再前后(X) → 再转向。

        闭环：按 body_x/body_y/dyaw 实际位移量走，直到误差进阈值。
        开环：仍用 seg 的 duration_*。
        """
        speed = float(seg.get("speed", self.speed))
        turn_speed = float(seg.get("turn_speed", self.turn_speed))
        if abs(turn_speed) < 1e-6:
            turn_speed = self.turn_speed

        if self.closed_loop_enable:
            rospy.loginfo(
                "%s segment(CL) speed=%.3fm/s turn=%.2f "
                "body=(%.3f, %.3f)m dyaw=%.1f°",
                self.log_prefix,
                speed,
                turn_speed,
                body_x,
                body_y,
                math.degrees(dyaw),
            )
            # 1) Y
            if abs(body_y) > self.xy_tolerance:
                timeout = max(
                    2.0,
                    abs(body_y) / max(speed, 1e-3) * self.timeout_margin,
                )
                self._wait_body_axis("y", body_y, speed, timeout, "go_linear_y_cl")
                if self.segment_pause_sec > 0:
                    rospy.sleep(self.segment_pause_sec)
            # 2) X
            if abs(body_x) > self.xy_tolerance:
                timeout = max(
                    2.0,
                    abs(body_x) / max(speed, 1e-3) * self.timeout_margin,
                )
                self._wait_body_axis("x", body_x, speed, timeout, "go_linear_x_cl")
                if self.segment_pause_sec > 0:
                    rospy.sleep(self.segment_pause_sec)
            # 3) 转向
            if abs(dyaw) > self.yaw_tolerance and abs(turn_speed) > 1e-6:
                # 用标定把 angular 估成 rad/s，再算超时
                est_w = max(abs(turn_speed) * self.angular_to_radps, 0.05)
                timeout = max(2.0, abs(dyaw) / est_w * self.timeout_margin)
                self._wait_turn_delta(dyaw, turn_speed, timeout, "turn_ang_cl")
                if self.segment_pause_sec > 0:
                    rospy.sleep(self.segment_pause_sec)
            return

        # ----- 开环回退 -----
        dx_t = float(seg.get("duration_x", 0.0))
        dy_t = float(seg.get("duration_y", 0.0))
        dt_t = float(seg.get("duration_turn", 0.0))
        rospy.loginfo(
            "%s segment(OL) speed=%.3fm/s turn=%.2f "
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
        """沿车体 -X 短后退（去救援区前的 pre_rescue_backoff）。"""
        speed = abs(float(seg.get("speed", self.speed)))
        if self.closed_loop_enable:
            # 优先用 distance_x；否则用 duration_x * speed 估距离
            dist = float(seg.get("distance_x", 0.0))
            if abs(dist) < 1e-4:
                dist = abs(float(seg.get("duration_x", 0.0))) * max(speed, 1e-3)
            if dist < 1e-4:
                return
            timeout = max(1.0, dist / max(speed, 1e-3) * self.timeout_margin)
            self._wait_body_axis("x", -dist, speed, timeout, "backoff_x_cl")
        else:
            dx_t = float(seg.get("duration_x", 0.0))
            if dx_t < 1e-4:
                return
            self.go_linear_x_timed(-speed, dx_t)
        if self.segment_pause_sec > 0:
            rospy.sleep(self.segment_pause_sec)


class NavTimedArmPi(object):
    """救援任务状态机：运动编排 + 可选无人机通讯。

    流程：
      parking → pickup → loading → [无人机握手/跳过]
              → (可选后退) → rescue_N → parking
    """

    def __init__(
        self,
        config_path=None,
        skip_drone=False,
        rescue_zone=2,
        set_velocity_topic=None,
        autostart=True,
    ):
        rospy.init_node("nav_rescue_armpi", anonymous=True)
        self.skip_drone = bool(skip_drone)
        # --skip-drone 时用命令行指定的救援区号 1~4
        self.rescue_zone = int(rescue_zone)
        self.config_path = Path(config_path or _DEFAULT_CONFIG)
        self.config = load_timed_config(self.config_path)

        # 从 YAML 拆出常用块
        timed = dict(self.config.get("timed") or {})
        self.mission_cfg = dict(self.config.get("mission") or {})
        self.segment_motion = dict(self.config.get("segment_motion") or {})
        self.arrive_pause_sec = float(timed.get("arrive_pause_sec", 0.50))
        closed_loop = dict(self.config.get("closed_loop") or {})

        speed = float(timed.get("speed", 0.06))  # m/s
        turn_speed = float(timed.get("turn_speed", 2.0))
        topic = str(
            set_velocity_topic
            if set_velocity_topic
            else timed.get(
                "set_velocity_topic", "/chassis_control/set_velocity"
            )
        )

        self.nav = TimedSpeedNavArmPi(
            speed=speed,
            turn_speed=turn_speed,
            set_velocity_topic=topic,
            segment_pause_sec=float(timed.get("segment_pause_sec", 0.30)),
            closed_loop=closed_loop,
        )
        # 起始位姿设为停车区
        parking = (self.config.get("zones") or {}).get("parking") or {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
        }
        self.nav.set_est_pose(parking)
        self.order = None  # RescueOrder，决定去哪个救援区
        self._current_zone = None
        self._current_zone_id = None

        rospy.on_shutdown(self.stop)
        rospy.loginfo(
            "[Mission] nav_rescue_armpi | config=%s | skip_drone=%s "
            "rescue=%s | speed=%.3fm/s turn=%.2f | closed_loop=%s",
            self.config_path,
            self.skip_drone,
            self.rescue_zone,
            speed,
            turn_speed,
            closed_loop.get("enable", True),
        )
        if autostart:
            ok = self.run_mission()
            rospy.loginfo("[Mission] 结束 ok=%s", ok)

    def stop(self):
        self.nav.stop()

    def _abort(self, reason):
        """任务中止：停车并返回 False。"""
        rospy.logerr("[Mission] abort: %s", reason)
        self.stop()
        return False

    def _zone_pose(self, zone_name, zone_id=None):
        """取 YAML zones 中某点的 {x,y,yaw}。rescue 需带区号。"""
        zones = self.config["zones"]
        if zone_name == "rescue":
            return zones["rescue"][int(zone_id)]
        return zones[zone_name]

    def _segment_key(self, zone_name, zone_id=None):
        """目标区域名 → segment_motion 键名（如 to_rescue_2 / to_parking_3）。"""
        if zone_name == "pickup":
            return "to_pickup"
        if zone_name == "loading":
            return "to_loading"
        if zone_name == "rescue":
            return "to_rescue_%d" % int(zone_id)
        if zone_name == "parking":
            # 从哪个救援区返回，选用对应的 to_parking_N
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
        """世界系航点差 → 车体系位移/转角，再取正负号。

        返回：sx, sy, st, body_x, body_y, dyaw
          sx/sy/st ∈ {-1, 0, +1}；
          闭环时 body_x/body_y/dyaw 直接作为目标位移量。
        """
        dx = float(to_pose["x"]) - float(from_pose["x"])
        dy = float(to_pose["y"]) - float(from_pose["y"])
        yaw = float(from_pose.get("yaw", 0.0))
        # 世界系差旋到当前车体坐标系
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
        """走到指定区域：闭环按位姿误差，开环按 duration_*。"""
        pose = self._zone_pose(zone_name, zone_id)
        label = "rescue_%s" % zone_id if zone_name == "rescue" else zone_name
        key_zone_id = zone_id
        if zone_name == "parking":
            # 回停车区时带上当前救援区号，选用 to_parking_N
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
        mode = "CL" if self.nav.closed_loop_enable else "OL"
        rospy.loginfo(
            "[Mission] goto %s via %s [%s] (%.3f, %.3f, yaw=%.3f) "
            "body≈(%.3f, %.3f)m dyaw=%.1f°",
            label,
            key,
            mode,
            pose["x"],
            pose["y"],
            pose["yaw"],
            body_x,
            body_y,
            math.degrees(dyaw),
        )
        self.nav.run_segment(
            sx, sy, st, seg, body_x=body_x, body_y=body_y, dyaw=dyaw
        )
        # 对齐到目标航点，方便下一段算符号（里程计漂移时减少累积误差）
        self.nav.set_est_pose(pose)
        self._current_zone = zone_name
        self._current_zone_id = zone_id
        self.stop()
        if self.arrive_pause_sec > 0:
            rospy.sleep(self.arrive_pause_sec)
        return True

    def _comm_kwargs(self):
        """YAML comm 块副本（主机/账号/远程 flag 路径）。"""
        return dict(self.config.get("comm") or {})

    def run_mission(self):
        """执行完整救援任务；成功返回 True，中止返回 False。"""
        timeouts = self.config.get("timeouts", {}) or {}
        mission = self.mission_cfg
        comm = self._comm_kwargs()

        # ----- 1. 取货区 -----
        if not self.goto_zone("pickup"):
            return self._abort("无法到达取货区")
        pickup_hold = float(mission.get("pickup_hold_sec", 5.0))
        if pickup_hold > 0:
            rospy.loginfo("[Mission] 取货区等待 %.1fs", pickup_hold)
            rospy.sleep(pickup_hold)

        # ----- 2. 装货区 -----
        if not self.goto_zone("loading"):
            return self._abort("无法到达装货区")

        # ----- 3. 确定救援区 + 装货/投送握手 -----
        if self.skip_drone:
            # 不连无人机：用命令行 --rescue
            self.order = RescueOrder(zone=self.rescue_zone, level=1)
            rospy.loginfo(
                "[Mission] 跳过无人机，救援区 zone=%d", self.order.zone
            )
            hold = float(mission.get("skip_drone_loading_hold_sec", 5.0))
            if hold > 0:
                rospy.loginfo("[Mission] 跳过无人机：装货区保持 %.1fs", hold)
                rospy.sleep(hold)
        else:
            # 联机：等无人机写 rescue_target.flag（内容为 1~4）
            rospy.loginfo("[Mission] 等待无人机救援目标")
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

            hold = float(mission.get("loading_hold_sec", 5.0))
            if hold > 0:
                rospy.loginfo("[Mission] 装货保持 %.1fs", hold)
                rospy.sleep(hold)
            # 通知无人机：装货完成，可起飞
            notify_loading_done(
                remote_path=comm.get(
                    "loading_done_path", "/mnt/loading_done.flag"
                ),
                host=comm.get("drone_host", "192.168.10.66"),
                user=comm.get("drone_user", "forlinx"),
                password=comm.get("drone_password", "forlinx"),
            )
            # 等待无人机投送完成
            if not wait_for_delivery_done(
                remote_path=comm.get(
                    "delivery_done_path", "/mnt/delivery_done.flag"
                ),
                host=comm.get("drone_host", "192.168.10.66"),
                user=comm.get("drone_user", "forlinx"),
                password=comm.get("drone_password", "forlinx"),
                timeout=timeouts.get("delivery", 300),
            ):
                return self._abort("无人机投送超时")

        # ----- 4. 去救援区前可选短后退 -----
        if mission.get("pre_rescue_backoff_enable", True):
            seg = self.segment_motion.get("pre_rescue_backoff")
            if seg:
                rospy.loginfo(
                    "[Mission] 前往救援区前后退 duration_x=%.2fs speed=%.3fm/s",
                    float(seg.get("duration_x", 0.0)),
                    float(seg.get("speed", self.nav.speed)),
                )
                self.nav.run_backoff_x(seg)
                self.stop()
                rospy.sleep(0.3)

        # ----- 5. 救援区 -----
        if not self.goto_zone("rescue", zone_id=self.order.zone):
            return self._abort("无法到达救援区")

        if self.skip_drone:
            hold = float(mission.get("skip_drone_unload_hold_sec", 5.0))
            if hold > 0:
                rospy.loginfo("[Mission] 跳过无人机：救援区保持 %.1fs", hold)
                rospy.sleep(hold)
        else:
            hold = float(mission.get("unload_hold_sec", 5.0))
            if hold > 0:
                rospy.loginfo("[Mission] 卸货保持 %.1fs", hold)
                rospy.sleep(hold)
            # 通知无人机：小车卸货完成
            notify_unload_done(
                remote_path=comm.get(
                    "unload_done_path", "/mnt/unload_done.flag"
                ),
                host=comm.get("drone_host", "192.168.10.66"),
                user=comm.get("drone_user", "forlinx"),
                password=comm.get("drone_password", "forlinx"),
            )

        # ----- 6. 回停车区 -----
        if not self.goto_zone("parking"):
            return self._abort("无法返回停车区")

        self.stop()
        rospy.loginfo(
            "[Mission] 全部任务完成（ArmPi %s）",
            "里程计闭环" if self.nav.closed_loop_enable else "SetVelocity×时间",
        )
        return True


def main():
    """命令行入口。"""
    if sys.version_info.major == 2:
        print("Please run this program with python3!")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="nav_rescue_armpi：ArmPi Pro 里程计闭环救援导航"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="参数配置 YAML（默认同目录 nav_rescue_armpi.yaml）",
    )
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="只初始化节点，不自动跑任务（调试用）",
    )
    parser.add_argument(
        "--skip-drone",
        action="store_true",
        help="跳过无人机 SSH 通讯，救援区用 --rescue",
    )
    parser.add_argument(
        "--rescue",
        type=int,
        default=2,
        choices=(1, 2, 3, 4),
        help="--skip-drone 时的目标救援区号",
    )
    parser.add_argument(
        "--set-velocity-topic",
        type=str,
        default=None,
        help="默认 /chassis_control/set_velocity",
    )
    args = parser.parse_args()
    NavTimedArmPi(
        config_path=args.config,
        skip_drone=args.skip_drone,
        rescue_zone=args.rescue,
        set_velocity_topic=args.set_velocity_topic,
        autostart=not args.no_autostart,
    )
    # 任务结束后保持节点存活，便于 Ctrl+C / 看日志
    rospy.spin()


if __name__ == "__main__":
    main()
