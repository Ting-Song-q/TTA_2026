#!/usr/bin/python3
# coding=UTF-8
"""开环位移 + 方案二区域白名单避障（与 run_car_duo 一致，唯一运动实现）。"""

from __future__ import print_function

import copy
import math

from laser_avoidance import (
    get_clearances,
    is_emergency,
    is_path_clear,
    pick_bypass_direction,
)


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
        "lidar_mount": "rear",
        "lidar_to_body": {
            "front": 0.12,
            "back": 0.31,
            "left": 0.25,
            "right": 0.25,
        },
        # 车体净空阈值（m）；过小会导致制动距离不够而撞障
        "safe_distance": 0.35,
        "side_safe_distance": 0.25,
        "critical_distance": 0.18,
        "emergency_stop_distance": 0.12,
        "self_hit_margin": 0.03,
        "sector_half_width": 30,
        "side_sector_half_width": 55,
        "bypass_speed": 0.1,
        "creep_ratio": 0.0,
        "max_sidestep": 0.30,
        "pass_distance": 0.35,
        "retreat_speed_ratio": 0.5,
        "approach_slow_distance": 0.50,
        "approach_slow_ratio": 1.0,
        "max_linear_speed": 0.30,
        "max_angular_speed": 0.4,
        "control_rate": 20,
        "max_bypass_time": 20.0,
        "move_timeout": 90.0,
        "zone_margin": 0.15,
        "default_zone_size": [0.8, 0.8],
    }


def _parse_zone_box(name, node, margin, default_size):
    if not isinstance(node, dict):
        return None
    if node.get("mode", "center_size") not in ("center_size", None, ""):
        return None
    center = node.get("center") or {}
    size = node.get("size") or default_size
    if len(size) < 2:
        size = default_size
    return {
        "name": name,
        "cx": float(center.get("x", 0.0)),
        "cy": float(center.get("y", 0.0)),
        "yaw": float(center.get("yaw", 0.0)),
        "half_x": float(size[0]) * 0.5 + margin,
        "half_y": float(size[1]) * 0.5 + margin,
    }


def build_zone_whitelist(config):
    """
    构建方案二白名单区域。
    中心一律用 zones/waypoints；zone_bounds 只提供 size。
    """
    avoid = dict(default_avoidance_cfg())
    avoid.update(config.get("obstacle_avoidance") or {})
    wl = config.get("zone_whitelist") or {}
    include = set(wl.get("include") or ["pickup", "loading", "rescue"])
    margin = float(avoid.get("zone_margin", 0.15))
    default_size = list(avoid.get("default_zone_size") or [0.8, 0.8])
    waypoints = config.get("zones") or config.get("waypoints") or {}
    bounds = config.get("zone_bounds") or {}
    boxes = []

    def size_from_bound(bound_node):
        if isinstance(bound_node, dict):
            size = bound_node.get("size") or default_size
            if len(size) >= 2:
                return [float(size[0]), float(size[1])]
        return list(default_size)

    def add_from_pose(name, pose, bound_node=None):
        if not pose:
            return
        # PyYAML 1.1 may parse bare key `y` as True; also accept center: {...}
        if not isinstance(pose, dict):
            return
        if pose.get("x") is None and pose.get("y") is None and True not in pose:
            center = pose.get("center")
            if isinstance(center, dict):
                pose = center
        px = pose.get("x", 0.0)
        py = pose.get("y", pose.get(True, 0.0))
        pyaw = pose.get("yaw", 0.0)
        size = size_from_bound(bound_node)
        boxes.append(
            {
                "name": name,
                "cx": float(px),
                "cy": float(py),
                "yaw": float(pyaw),
                "half_x": float(size[0]) * 0.5 + margin,
                "half_y": float(size[1]) * 0.5 + margin,
            }
        )

    if "pickup" in include:
        add_from_pose("pickup", waypoints.get("pickup"), bounds.get("pickup"))
    if "loading" in include:
        add_from_pose("loading", waypoints.get("loading"), bounds.get("loading"))
    if "parking" in include:
        add_from_pose("parking", waypoints.get("parking"), bounds.get("parking"))
    if "rescue" in include:
        rescue_bounds = bounds.get("rescue") or {}
        for zid, pose in (waypoints.get("rescue") or {}).items():
            node = rescue_bounds.get(zid) or rescue_bounds.get(str(zid))
            add_from_pose("rescue_%s" % zid, pose, node)

    return boxes, avoid


def point_in_oriented_box(px, py, box):
    dx = px - box["cx"]
    dy = py - box["cy"]
    yaw = box["yaw"]
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    lx = c * dx - s * dy
    ly = s * dx + c * dy
    return abs(lx) <= box["half_x"] and abs(ly) <= box["half_y"]


def laser_to_body(range_m, ang_rad, lidar_mount):
    if lidar_mount == "front":
        return range_m * math.cos(ang_rad), range_m * math.sin(ang_rad)
    return (
        range_m * math.cos(ang_rad + math.pi),
        range_m * math.sin(ang_rad + math.pi),
    )


def body_to_map(bx, by, pose):
    yaw = pose["yaw"]
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (
        pose["x"] + c * bx - s * by,
        pose["y"] + s * bx + c * by,
    )


def filter_scan_by_zone_whitelist(laser_data, pose, zone_boxes, lidar_mount="rear"):
    if laser_data is None or not zone_boxes or pose is None:
        return laser_data, 0

    filtered = copy.copy(laser_data)
    ranges = list(laser_data.ranges)
    angle = laser_data.angle_min
    incr = laser_data.angle_increment
    rmin = laser_data.range_min
    rmax = laser_data.range_max
    masked = 0

    for i, r in enumerate(ranges):
        ang = angle + i * incr
        if not math.isfinite(r) or r < rmin or r > rmax:
            continue
        bx, by = laser_to_body(r, ang, lidar_mount)
        mx, my = body_to_map(bx, by, pose)
        for box in zone_boxes:
            if point_in_oriented_box(mx, my, box):
                ranges[i] = float("inf")
                masked += 1
                break

    filtered.ranges = ranges
    return filtered, masked


def _import_ros():
    import rospy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry

    return rospy, Twist, LaserScan, Odometry


class LaserOpenLoopNav(object):
    """开环位移 + 方案二白名单过滤后的激光绕障（唯一底盘运动实现）。"""

    def __init__(
        self,
        speed=0.5,
        turn_speed=0.5,
        align_speed=0.35,
        avoidance_cfg=None,
        zone_boxes=None,
        log_prefix="[OpenLoop]",
        node_name="openloop_duo",
        wait_laser=True,
        on_laser=None,
        cmd_vel_topic="/cmd_vel",
        odom_topic="/odom",
    ):
        rospy, Twist, LaserScan, Odometry = _import_ros()
        self._rospy = rospy
        self._Twist = Twist
        self._log_prefix = log_prefix
        self._on_laser = on_laser
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True)
        self.cmd_vel_topic = str(cmd_vel_topic) if cmd_vel_topic else "/cmd_vel"
        self.odom_topic = str(odom_topic) if odom_topic else "/odom"
        self.velocity_publisher = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        self.laser_subscriber = rospy.Subscriber(
            "/scan", LaserScan, self.laser_callback, queue_size=1
        )
        self.odom_subscriber = rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_callback, queue_size=1
        )
        self._odom_yaw = None
        self.laser_data = None
        self.rate = rospy.Rate(100)
        self.speed = float(speed)
        self.turn_speed = float(turn_speed)
        self.align_speed = float(align_speed)
        self.avoidance_cfg = dict(default_avoidance_cfg())
        if avoidance_cfg:
            self.avoidance_cfg.update(avoidance_cfg)
        self.zone_boxes = list(zone_boxes or [])
        self._est_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self._last_turn_spun = 0.0
        self._last_turn_goal = 0.0
        if wait_laser:
            self._wait_laser()
        rospy.loginfo(
            "%s 白名单区域 %d 个, 避障=%s mount=%s cmd_vel=%s",
            self._log_prefix,
            len(self.zone_boxes),
            self.avoidance_cfg.get("enabled", True),
            self.avoidance_cfg.get("lidar_mount", "rear"),
            self.cmd_vel_topic,
        )
        for box in self.zone_boxes:
            rospy.loginfo(
                "%s  ignore-zone %s center=(%.2f,%.2f) half=(%.2f,%.2f)",
                self._log_prefix,
                box["name"],
                box["cx"],
                box["cy"],
                box["half_x"],
                box["half_y"],
            )

    def set_est_pose(self, pose):
        self._est_pose = {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "yaw": float(pose["yaw"]),
        }

    def get_est_pose(self):
        return dict(self._est_pose)

    def _wait_laser(self, timeout=10.0):
        rospy = self._rospy
        rospy.loginfo("%s 等待 /scan ...", self._log_prefix)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown() and self.laser_data is None:
            if rospy.Time.now() > deadline:
                rospy.logwarn("%s 等待 /scan 超时，继续启动", self._log_prefix)
                return
            self.rate.sleep()
        if self.laser_data is not None:
            rospy.loginfo("%s /scan OK", self._log_prefix)

    def laser_callback(self, laser_data):
        self.laser_data = laser_data
        if self._on_laser is not None:
            self._on_laser(laser_data)

    def odom_callback(self, odom_msg):
        try:
            from tf.transformations import euler_from_quaternion

            quat = odom_msg.pose.pose.orientation
            _, _, yaw = euler_from_quaternion(
                [quat.x, quat.y, quat.z, quat.w]
            )
            # 原始 /odom 航向；转向闭环里会自动判定符号，避免写死取反导致卡住
            self._odom_yaw = normalize_yaw(float(yaw))
        except Exception:
            self._odom_yaw = None

    def get_odom_yaw(self):
        return self._odom_yaw

    def get_distance(self, laser_data, angle):
        angle_in_rad = math.radians(angle)
        idx = int(
            (angle_in_rad - laser_data.angle_min) / laser_data.angle_increment
        )
        ranges = laser_data.ranges
        if idx < 0 or idx >= len(ranges):
            return float("inf")
        return ranges[idx]

    def stop(self):
        vel_msg = self._Twist()
        self.velocity_publisher.publish(vel_msg)

    def _filtered_scan(self):
        cfg = self.avoidance_cfg
        if not cfg.get("enabled", True) or not self.zone_boxes:
            return self.laser_data, 0
        return filter_scan_by_zone_whitelist(
            self.laser_data,
            self._est_pose,
            self.zone_boxes,
            lidar_mount=cfg.get("lidar_mount", "rear"),
        )

    def _integrate_pose(self, vx, vy, dt):
        yaw = self._est_pose["yaw"]
        c = math.cos(yaw)
        s = math.sin(yaw)
        self._est_pose["x"] += (c * vx - s * vy) * dt
        self._est_pose["y"] += (s * vx + c * vy) * dt

    def turn_ang(self, ang_speed, goal_rotation, guard=None, target_yaw=None):
        """转向：优先 /odom 闭环；odom 无反馈/卡住时回退开环定时。

        guard=None: 跟随 obstacle_avoidance.enabled
        guard=True: 侧向急停会暂停转向（绕障中用）
        guard=False: 强制纯开环/无激光急停（到位后 face_yaw 用）
        target_yaw: 闭环结束后写入估计航向的目标值（可选）
        """
        rospy = self._rospy
        self._last_turn_spun = 0.0
        self._last_turn_goal = float(goal_rotation)
        if abs(goal_rotation) < 1e-4:
            rospy.loginfo(
                "%s turn_ang 目标角度过小 (%.3f°)，跳过",
                self._log_prefix,
                math.degrees(goal_rotation),
            )
            return
        cfg = self.avoidance_cfg
        if guard is None:
            use_guard = bool(cfg.get("enabled", True))
        else:
            use_guard = bool(guard)
        dt = 1.0 / float(cfg.get("control_rate", 20)) if use_guard else 0.01
        deadline = rospy.Time.now().to_sec() + float(cfg.get("move_timeout", 90.0))

        remaining_goal = goal_rotation
        if self._odom_yaw is not None:
            ok, spun = self._turn_ang_closed_loop(
                ang_speed, goal_rotation, use_guard, dt, deadline
            )
            self._last_turn_spun = float(spun)
            if ok:
                if target_yaw is not None:
                    self._est_pose["yaw"] = normalize_yaw(target_yaw)
                else:
                    self._est_pose["yaw"] = normalize_yaw(
                        self._est_pose["yaw"] + spun
                    )
                rospy.loginfo(
                    "%s turn_ang closed-loop done: spun=%.1f° remain=%.1f° "
                    "final_odom=%.1f°",
                    self._log_prefix,
                    math.degrees(spun),
                    math.degrees(normalize_yaw(goal_rotation - spun)),
                    math.degrees(self._odom_yaw or 0.0),
                )
                return
            # 仅当剩余极小（~2°）才跳过开环；否则按真实剩余补转
            remaining_goal = normalize_yaw(goal_rotation - spun)
            if abs(remaining_goal) < 0.035:
                # 按实际转角更新，避免“假到位”污染后续航向
                self._est_pose["yaw"] = normalize_yaw(self._est_pose["yaw"] + spun)
                rospy.loginfo(
                    "%s turn_ang closed-loop near-done (remain=%.1f°)，跳过开环补转",
                    self._log_prefix,
                    math.degrees(remaining_goal),
                )
                return
            rospy.logwarn(
                "%s 闭环反馈异常，回退开环补转 %.1f°",
                self._log_prefix,
                math.degrees(remaining_goal),
            )

        closed_spun = float(self._last_turn_spun)
        self._turn_ang_open_loop(
            ang_speed, remaining_goal, use_guard, dt, deadline, target_yaw
        )
        # 开环补转后累计总转角，供 face_yaw 精对齐使用
        self._last_turn_spun = closed_spun + float(self._last_turn_spun)

    def _turn_ang_closed_loop(self, ang_speed, goal_rotation, use_guard, dt, deadline):
        """相对闭环转向。返回 (success, spun_signed)。

        - 锁定初始转向方向，避免 ±180° 最短路径来回拧
        - 自动判定 odom 符号（rmep_bringup 对 yaw 取反时也能用）
        - 到位容差约 2°；末段保持较高转速，卡住时先满速再顶一次
        """
        rospy = self._rospy
        direction = 1 if goal_rotation > 0 else -1
        target_abs = abs(goal_rotation)
        tolerance = 0.035  # ~2°，保证 90°/180° 到位精度
        start_yaw = self._odom_yaw
        last_yaw = start_yaw
        progress_raw = 0.0  # 按 /odom 原始增量累计
        odom_sign = None  # +1 / -1，首次明显运动后锁定
        stall_timeout = 1.50
        min_turn_speed = max(0.35, abs(ang_speed) * 0.70)
        force_full_speed = False
        stall_boost_used = False
        last_progress_abs = 0.0
        last_progress_time = rospy.Time.now().to_sec()
        rospy.loginfo(
            "%s turn_ang closed-loop: goal=%.1f° start_odom=%.1f° dir=%s guard=%s",
            self._log_prefix,
            math.degrees(goal_rotation),
            math.degrees(start_yaw or 0.0),
            "CCW(+)" if direction > 0 else "CW(-)",
            use_guard,
        )
        twist_ang = self._Twist()

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            if now > deadline:
                rospy.logwarn(
                    "%s 闭环转向超时，已转 %.1f°/%.1f°",
                    self._log_prefix,
                    math.degrees(progress_raw * (odom_sign or 1) * direction),
                    math.degrees(goal_rotation),
                )
                self.stop()
                sign = odom_sign or 1
                progressed_abs = progress_raw * sign * direction
                if target_abs - progressed_abs <= tolerance:
                    return True, progress_raw * sign
                return False, progress_raw * sign

            odom_yaw = self._odom_yaw
            if odom_yaw is None:
                self.stop()
                return False, progress_raw * (odom_sign or 1)

            delta = normalize_yaw(odom_yaw - last_yaw)
            progress_raw += delta
            last_yaw = odom_yaw

            # 首次看到明显运动时，锁定 odom 符号：使 progress 与 direction 同向
            if odom_sign is None and abs(progress_raw) > 0.08:
                odom_sign = 1 if (progress_raw * direction) > 0 else -1
                rospy.loginfo(
                    "%s closed-loop odom_sign=%+d (raw_progress=%.1f°)",
                    self._log_prefix,
                    odom_sign,
                    math.degrees(progress_raw),
                )

            sign = odom_sign if odom_sign is not None else 1
            progressed_abs = progress_raw * sign * direction
            remaining = target_abs - progressed_abs
            if remaining <= tolerance:
                self.stop()
                return True, progress_raw * sign

            # odom 长时间几乎不动：先满速再顶一次；仍不动再回退开环
            if abs(progressed_abs - last_progress_abs) > 0.03:
                last_progress_abs = progressed_abs
                last_progress_time = now
            elif now - last_progress_time > stall_timeout:
                if remaining <= tolerance:
                    self.stop()
                    return True, progress_raw * sign
                if not stall_boost_used and remaining <= 0.50:
                    stall_boost_used = True
                    force_full_speed = True
                    last_progress_time = now
                    rospy.logwarn(
                        "%s 闭环末段无进展 (remain=%.1f°)，满速再顶一次",
                        self._log_prefix,
                        math.degrees(remaining),
                    )
                else:
                    rospy.logwarn(
                        "%s 闭环 odom 无进展 (%.2fs)，回退开环",
                        self._log_prefix,
                        stall_timeout,
                    )
                    self.stop()
                    return False, progress_raw * sign

            speed = abs(ang_speed)
            if force_full_speed:
                speed = abs(ang_speed)
            elif remaining < 0.40 and odom_sign is not None:
                # 末段略减速，但最低转速不能太低（底盘对 <0.3 常无明显响应）
                speed = max(min_turn_speed, abs(ang_speed) * (remaining / 0.40))
            signed_w = -speed if direction > 0 else speed

            if use_guard:
                scan, _ = self._filtered_scan()
                if scan is None and self.avoidance_cfg.get("fail_closed", True):
                    self.stop()
                    rospy.logwarn_throttle(
                        1.0, "%s 转向无激光，fail-closed 停车", self._log_prefix
                    )
                    rospy.sleep(dt)
                    continue
                if is_emergency(scan, self.avoidance_cfg, 0.0, 0.0, signed_w):
                    self.stop()
                    rospy.logwarn_throttle(
                        1.0, "%s 转向遇障，等待", self._log_prefix
                    )
                    rospy.sleep(dt)
                    continue

            twist_ang.angular.z = signed_w
            self.velocity_publisher.publish(twist_ang)
            rospy.sleep(dt)

        self.stop()
        return False, progress_raw * (odom_sign or 1)

    def _turn_ang_open_loop(
        self, ang_speed, goal_rotation, use_guard, dt, deadline, target_yaw=None
    ):
        """开环定时转向。对超调用 duration_scale 略缩短时间。"""
        rospy = self._rospy
        if abs(goal_rotation) < 1e-4:
            return
        # 实机开环常会转过一点：默认略缩短；补转场景可用参数覆盖
        duration_scale = float(self.avoidance_cfg.get("turn_duration_scale", 0.98))
        duration_scale = max(0.5, min(1.0, duration_scale))
        signed_w = -ang_speed if goal_rotation > 0 else ang_speed
        target_rotation_time = (
            math.fabs(goal_rotation / max(abs(ang_speed), 1e-6)) * duration_scale
        )
        twist_ang = self._Twist()
        twist_ang.angular.z = signed_w
        progressed = 0.0
        rospy.loginfo(
            "%s turn_ang open-loop: goal=%.1f° speed=%.2f rad/s dt=%.3f "
            "duration=%.2fs scale=%.2f guard=%s",
            self._log_prefix,
            math.degrees(goal_rotation),
            signed_w,
            dt,
            target_rotation_time,
            duration_scale,
            use_guard,
        )

        while progressed < target_rotation_time and not rospy.is_shutdown():
            if rospy.Time.now().to_sec() > deadline:
                rospy.logwarn(
                    "%s 转向超时 (%.1f°/%.1f°)",
                    self._log_prefix,
                    math.degrees(progressed * abs(ang_speed) / max(duration_scale, 1e-6)),
                    math.degrees(abs(goal_rotation)),
                )
                break
            if use_guard:
                scan, _ = self._filtered_scan()
                if scan is None and self.avoidance_cfg.get("fail_closed", True):
                    self.stop()
                    rospy.logwarn_throttle(
                        1.0, "%s 转向无激光，fail-closed 停车", self._log_prefix
                    )
                    rospy.sleep(dt)
                    continue
                if is_emergency(scan, self.avoidance_cfg, 0.0, 0.0, signed_w):
                    self.stop()
                    rospy.logwarn_throttle(1.0, "%s 转向遇障，等待", self._log_prefix)
                    rospy.sleep(dt)
                    continue
            self.velocity_publisher.publish(twist_ang)
            rospy.sleep(dt)
            progressed += dt

        spun = math.copysign(
            progressed * abs(ang_speed) / max(duration_scale, 1e-6), goal_rotation
        )
        if abs(spun) > abs(goal_rotation):
            spun = goal_rotation
        self._last_turn_spun = float(spun)
        if target_yaw is not None:
            self._est_pose["yaw"] = normalize_yaw(target_yaw)
        else:
            self._est_pose["yaw"] = normalize_yaw(self._est_pose["yaw"] + spun)
        self.stop()
        rospy.loginfo(
            "%s turn_ang open-loop done: progressed=%.2fs/%.2fs spun=%.1f°",
            self._log_prefix,
            progressed,
            target_rotation_time,
            math.degrees(spun),
        )

    def go_linear_x(self, linear_speed, goal_distance):
        if self.avoidance_cfg.get("enabled", True):
            self._move_with_avoidance(linear_speed, 0.0, abs(goal_distance), axis="x")
            return
        self._go_linear_timed("x", linear_speed, goal_distance)

    def go_linear_y(self, linear_speed, goal_distance):
        if self.avoidance_cfg.get("enabled", True):
            self._move_with_avoidance(0.0, linear_speed, abs(goal_distance), axis="y")
            return
        self._go_linear_timed("y", linear_speed, goal_distance)

    def _go_linear_timed(self, axis, linear_speed, goal_distance):
        rospy = self._rospy
        if abs(goal_distance) < 1e-4 or abs(linear_speed) < 1e-6:
            return
        target_linear_time = math.fabs(goal_distance / linear_speed)
        twist = self._Twist()
        if axis == "x":
            twist.linear.x = linear_speed
        else:
            twist.linear.y = linear_speed
        go_start_time = rospy.Time.now()
        while (rospy.Time.now() - go_start_time).to_sec() < target_linear_time:
            if rospy.is_shutdown():
                break
            self.velocity_publisher.publish(twist)
            self.rate.sleep()
        elapsed = min(
            (rospy.Time.now() - go_start_time).to_sec(), target_linear_time
        )
        if axis == "x":
            self._integrate_pose(linear_speed, 0.0, elapsed)
        else:
            self._integrate_pose(0.0, linear_speed, elapsed)
        self.stop()

    def _clamp_speed(self, vx, vy):
        max_v = float(self.avoidance_cfg.get("max_linear_speed", 0.20))
        return (
            max(-max_v, min(max_v, vx)),
            max(-max_v, min(max_v, vy)),
        )

    def _publish_motion(self, vx, vy, dt):
        vx, vy = self._clamp_speed(vx, vy)
        twist = self._Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        self.velocity_publisher.publish(twist)
        self._integrate_pose(vx, vy, dt)
        return vx, vy

    def _obstacle_side_clear(self, scan, cfg, bypass_side):
        """绕障后，绕障侧对侧是否已离开障碍（可回归车道）。"""
        if not bypass_side or scan is None:
            return True
        side_safe = float(cfg.get("side_safe_distance", 0.25))
        clearances = get_clearances(scan, cfg)
        if bypass_side == "left":
            return clearances["right"] >= side_safe
        if bypass_side == "right":
            return clearances["left"] >= side_safe
        return True

    def _approach_speed(self, speed, scan, vx, vy, cfg):
        """接近障碍时提前减速，增大制动余量。"""
        if scan is None:
            return speed * float(cfg.get("approach_slow_ratio", 1.0))
        clearances = get_clearances(scan, cfg)
        slow_at = float(cfg.get("approach_slow_distance", 0.50))
        ratio = float(cfg.get("approach_slow_ratio", 1.0))
        if ratio >= 0.999:
            return speed
        if vx > 0.01:
            d = clearances["front"]
        elif vx < -0.01:
            d = clearances["back"]
        elif vy > 0.01:
            d = clearances["left"]
        elif vy < -0.01:
            d = clearances["right"]
        else:
            return speed
        if d >= slow_at:
            return speed
        if d <= float(cfg.get("safe_distance", 0.35)):
            return speed * ratio
        # 线性插值：safe→slow_at 之间从 ratio*speed 过渡到 speed
        t = (d - float(cfg.get("safe_distance", 0.35))) / max(
            1e-3, slow_at - float(cfg.get("safe_distance", 0.35))
        )
        return speed * (ratio + (1.0 - ratio) * max(0.0, min(1.0, t)))

    def _move_with_avoidance(self, vx, vy, goal_distance, axis):
        """
        沿轴运动 goal_distance；遇障时结构化侧移绕行再回车道。

        阶段：normal → step_out → pass → step_back
        两侧都堵时 retreat 后再试。主轴只累计净前进距离，终点不变。
        """
        rospy = self._rospy
        cfg = self.avoidance_cfg
        main_v = vx if axis == "x" else vy
        speed = abs(main_v)
        if speed < 1e-6:
            return

        dt = 1.0 / float(cfg.get("control_rate", 20))
        progress = 0.0
        deadline = rospy.Time.now().to_sec() + float(cfg.get("move_timeout", 90.0))
        max_bypass = float(cfg.get("max_bypass_time", 20.0))
        max_sidestep = float(cfg.get("max_sidestep", cfg.get("sidestep_distance", 0.30)))
        pass_need = float(cfg.get("pass_distance", 0.35))
        bypass_speed = float(cfg.get("bypass_speed", 0.08))
        creep_ratio = float(cfg.get("creep_ratio", 0.0))
        retreat_ratio = float(cfg.get("retreat_speed_ratio", 0.5))

        phase = "normal"
        lateral_offset = 0.0
        pass_progress = 0.0
        bypass_vy = 0.0
        bypass_side = ""
        phase_start = rospy.Time.now().to_sec()

        while progress < goal_distance and not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            if now > deadline:
                rospy.logwarn(
                    "%s 避障移动超时 (%.2fm/%.2fm) phase=%s",
                    self._log_prefix,
                    progress,
                    goal_distance,
                    phase,
                )
                break

            scan, masked = self._filtered_scan()
            if scan is None and cfg.get("fail_closed", True):
                self.stop()
                rospy.logwarn_throttle(1.0, "%s 无激光，fail-closed 停车", self._log_prefix)
                rospy.sleep(dt)
                continue

            # 紧急：停车并沿主轴反向拉开
            if is_emergency(scan, cfg, vx, vy, 0.0):
                self.stop()
                rospy.logwarn_throttle(1.0, "%s 紧急距离，反向拉开", self._log_prefix)
                retreat = speed * retreat_ratio
                if axis == "x":
                    rvx = -retreat if main_v > 0 else retreat
                    self._publish_motion(rvx, 0.0, dt)
                else:
                    rvy = -retreat if main_v > 0 else retreat
                    self._publish_motion(0.0, rvy, dt)
                rospy.sleep(dt)
                if phase not in ("step_out", "pass", "step_back"):
                    phase = "normal"
                continue

            cmd_vx = 0.0
            cmd_vy = 0.0

            if phase == "normal":
                if is_path_clear(scan, vx, vy, cfg):
                    run_speed = self._approach_speed(speed, scan, vx, vy, cfg)
                    if axis == "x":
                        cmd_vx = run_speed if main_v > 0 else -run_speed
                    else:
                        cmd_vy = run_speed if main_v > 0 else -run_speed
                    self._publish_motion(cmd_vx, cmd_vy, dt)
                    progress += run_speed * dt
                else:
                    vy_pick, side = pick_bypass_direction(scan, cfg)
                    if side == "backward":
                        phase = "retreat"
                        phase_start = now
                        rospy.logwarn("%s 两侧受阻，先后退", self._log_prefix)
                    else:
                        phase = "step_out"
                        phase_start = now
                        bypass_vy = vy_pick
                        bypass_side = side
                        pass_progress = 0.0
                        lateral_offset = 0.0
                        rospy.loginfo(
                            "%s 开始侧移绕障 side=%s masked=%d",
                            self._log_prefix,
                            bypass_side,
                            masked,
                        )
                    rospy.sleep(dt)
                    continue

            elif phase == "step_out":
                if now - phase_start > max_bypass:
                    rospy.logwarn("%s 侧移超时，改后退重试", self._log_prefix)
                    phase = "retreat"
                    phase_start = now
                    rospy.sleep(dt)
                    continue

                # 默认纯侧移；可选微小主轴 creep（默认 0，避免顶障）
                creep = speed * creep_ratio
                if axis == "x":
                    cmd_vx = (creep if main_v > 0 else -creep) if creep > 1e-6 else 0.0
                    cmd_vy = bypass_vy
                else:
                    cmd_vy = (creep if main_v > 0 else -creep) if creep > 1e-6 else 0.0
                    cmd_vx = bypass_vy

                # 侧移方向自身也要检查紧急
                if is_emergency(scan, cfg, cmd_vx, cmd_vy, 0.0):
                    self.stop()
                    rospy.logwarn_throttle(1.0, "%s 侧移方向危急，改后退", self._log_prefix)
                    phase = "retreat"
                    phase_start = now
                    rospy.sleep(dt)
                    continue

                self._publish_motion(cmd_vx, cmd_vy, dt)
                lateral_offset += abs(bypass_speed) * dt

                # 侧移后主轴已通，或达到最大侧移，进入通过段
                if is_path_clear(scan, vx, vy, cfg) or lateral_offset >= max_sidestep:
                    phase = "pass"
                    pass_progress = 0.0
                    phase_start = now
                    rospy.loginfo(
                        "%s 侧移 %.2fm，进入直行通过",
                        self._log_prefix,
                        lateral_offset,
                    )

            elif phase == "pass":
                if now - phase_start > max_bypass:
                    rospy.logwarn("%s 通过段超时，继续侧移", self._log_prefix)
                    phase = "step_out"
                    phase_start = now
                    rospy.sleep(dt)
                    continue

                if not is_path_clear(scan, vx, vy, cfg):
                    # 通过中又被挡：再侧移一点
                    phase = "step_out"
                    phase_start = now
                    rospy.loginfo("%s 通过中再次遇障，继续侧移", self._log_prefix)
                    rospy.sleep(dt)
                    continue

                run_speed = self._approach_speed(speed, scan, vx, vy, cfg)
                if axis == "x":
                    cmd_vx = run_speed if main_v > 0 else -run_speed
                else:
                    cmd_vy = run_speed if main_v > 0 else -run_speed
                self._publish_motion(cmd_vx, cmd_vy, dt)
                progress += run_speed * dt
                pass_progress += run_speed * dt

                if pass_progress >= pass_need:
                    if self._obstacle_side_clear(scan, cfg, bypass_side):
                        phase = "step_back"
                        phase_start = now
                        rospy.loginfo(
                            "%s 通过 %.2fm 后侧方畅通，回归车道",
                            self._log_prefix,
                            pass_progress,
                        )
                    else:
                        rospy.loginfo_throttle(
                            1.0,
                            "%s 侧方仍有障，继续直行 (%.2fm)",
                            self._log_prefix,
                            pass_progress,
                        )

            elif phase == "step_back":
                if axis == "x":
                    cmd_vy = -bypass_vy
                    cmd_vx = 0.0
                else:
                    cmd_vx = -bypass_vy
                    cmd_vy = 0.0

                if is_emergency(scan, cfg, cmd_vx, cmd_vy, 0.0):
                    # 回车道受阻：放弃回移，以当前横向偏移继续
                    phase = "normal"
                    lateral_offset = 0.0
                    rospy.logwarn("%s 回车道受阻，保持横向偏移继续", self._log_prefix)
                    rospy.sleep(dt)
                    continue

                self._publish_motion(cmd_vx, cmd_vy, dt)
                lateral_offset = max(0.0, lateral_offset - abs(bypass_speed) * dt)
                if lateral_offset <= 0.02:
                    phase = "normal"
                    lateral_offset = 0.0
                    rospy.loginfo("%s 已回原车道，继续直行", self._log_prefix)

            elif phase == "retreat":
                if now - phase_start > max_bypass:
                    rospy.logwarn("%s 后退超时，恢复前进尝试", self._log_prefix)
                    phase = "normal"
                    phase_start = now
                    rospy.sleep(dt)
                    continue

                retreat = speed * retreat_ratio
                if axis == "x":
                    cmd_vx = -retreat if main_v > 0 else retreat
                else:
                    cmd_vy = -retreat if main_v > 0 else retreat
                self._publish_motion(cmd_vx, cmd_vy, dt)

                vy_pick, side = pick_bypass_direction(scan, cfg)
                if side != "backward" and is_path_clear(scan, vx, vy, cfg):
                    phase = "normal"
                    rospy.loginfo("%s 后退后路径改善，重新前进", self._log_prefix)
                elif side != "backward":
                    phase = "step_out"
                    bypass_vy = vy_pick
                    bypass_side = side
                    pass_progress = 0.0
                    lateral_offset = 0.0
                    phase_start = now
                    rospy.loginfo(
                        "%s 后退后可侧移 side=%s",
                        self._log_prefix,
                        bypass_side,
                    )

            rospy.sleep(dt)

        self.stop()

    def move_to_initial_position(self, goal_left, goal_back):
        rospy = self._rospy
        retries_y = 0
        retries_x = 0
        while not rospy.is_shutdown():
            if self.laser_data:
                laser_data = self.laser_data
                left_distance = self.get_distance(laser_data, -90)
                back_distance = self.get_distance(laser_data, 0)
                if left_distance == float("inf") or back_distance == float("inf"):
                    continue
                if abs(left_distance - goal_left) > 0.02 and retries_y < 3:
                    error_y = abs(left_distance - goal_left)
                    speed = -0.1 * (left_distance - goal_left) / abs(
                        left_distance - goal_left
                    )
                    self.go_linear_y(speed, error_y)
                    retries_y += 1
                    rospy.sleep(1)
                    continue
                if abs(goal_back - back_distance) > 0.02 and retries_x < 3:
                    error_x = abs(goal_back - back_distance)
                    speed = 0.1 * (goal_back - back_distance) / abs(
                        goal_back - back_distance
                    )
                    self.go_linear_x(speed, error_x)
                    retries_x += 1
                    rospy.sleep(1)
                    continue
                break
            self.rate.sleep()
        if self.laser_data:
            print(
                self.get_distance(self.laser_data, -90),
                self.get_distance(self.laser_data, 0),
            )

    def move_adjust_goal_position_lb(self, goal_left, goal_back):
        rospy = self._rospy
        retries_y = 0
        retries_x = 0
        backoff_step = 0.05
        while not rospy.is_shutdown():
            if self.laser_data:
                laser_data = self.laser_data
                left_distance = self.get_distance(laser_data, -90)
                back_distance = self.get_distance(laser_data, 0)
                if left_distance == float("inf") or back_distance == float("inf"):
                    continue
                if left_distance < 0.8:
                    rospy.logwarn("Obstacle detected (<0.8 m). Backing off 5 cm...")
                    self.go_linear_x(-0.1, backoff_step)
                    rospy.sleep(0.5)
                    laser_data = self.laser_data
                    left_distance = self.get_distance(laser_data, -90)
                    if left_distance < 0.8:
                        rospy.logerr("Still too close after backoff. Aborting.")
                        self.go_linear_x(0.1, backoff_step)
                        rospy.sleep(0.5)
                        return
                    continue
                if abs(left_distance - goal_left) > 0.02 and retries_y < 3:
                    error_y = abs(left_distance - goal_left)
                    speed = -0.1 * (left_distance - goal_left) / abs(
                        left_distance - goal_left
                    )
                    self.go_linear_y(speed, error_y)
                    retries_y += 1
                    rospy.sleep(1)
                    continue
                if abs(goal_back - back_distance) > 0.02 and retries_x < 3:
                    error_x = abs(goal_back - back_distance)
                    speed = 0.1 * (goal_back - back_distance) / abs(
                        goal_back - back_distance
                    )
                    self.go_linear_x(speed, error_x)
                    retries_x += 1
                    rospy.sleep(1)
                    continue
                break
            self.rate.sleep()

    def move_adjust_goal_position_rf(self, goal_right, goal_front):
        rospy = self._rospy
        retries_y = 0
        retries_x = 0
        backoff_step = 0.05
        while not rospy.is_shutdown():
            if self.laser_data:
                laser_data = self.laser_data
                right_distance = self.get_distance(laser_data, 90)
                front_distance = self.get_distance(laser_data, 180)
                if right_distance == float("inf") or front_distance == float("inf"):
                    continue
                if right_distance < 0.8:
                    rospy.logwarn("Obstacle detected (<0.8 m). Backing off 5 cm...")
                    self.go_linear_x(-0.1, backoff_step)
                    rospy.sleep(0.5)
                    laser_data = self.laser_data
                    right_distance = self.get_distance(laser_data, 90)
                    if right_distance < 0.8:
                        rospy.logerr("Still too close after backoff. Aborting.")
                        self.go_linear_x(0.1, backoff_step)
                        rospy.sleep(0.5)
                        return
                    continue
                if abs(right_distance - goal_right) > 0.02 and retries_y < 3:
                    error_y = abs(right_distance - goal_right)
                    speed = 0.1 * (right_distance - goal_right) / abs(
                        right_distance - goal_right
                    )
                    self.go_linear_y(speed, error_y)
                    retries_y += 1
                    rospy.sleep(1)
                    continue
                if abs(goal_front - front_distance) > 0.02 and retries_x < 3:
                    error_x = abs(goal_front - front_distance)
                    speed = -0.1 * (goal_front - front_distance) / abs(
                        goal_front - front_distance
                    )
                    self.go_linear_x(speed, error_x)
                    retries_x += 1
                    rospy.sleep(1)
                    continue
                break
            self.rate.sleep()

    def face_yaw(self, current_yaw, target_yaw):
        """转到目标航向。若 /odom 可用则闭环，否则开环；到位对齐关闭侧向急停。

        主转完成后若 odom 显示仍有 2°~25° 剩余，再做一次短闭环精对齐，
        以保证航点 90°/180° 到位；过小残差直接忽略，避免二次轻拧。
        """
        tol = 0.035  # ~2°
        dyaw = normalize_yaw(target_yaw - current_yaw)
        if abs(dyaw) < tol:
            self._est_pose["yaw"] = normalize_yaw(target_yaw)
            return target_yaw
        feedback = "closed-loop" if self._odom_yaw is not None else "open-loop"
        self._rospy.loginfo(
            "%s face_yaw %.1f° -> %.1f° (Δ=%.1f°) turn_speed=%.2f [%s]",
            self._log_prefix,
            math.degrees(current_yaw),
            math.degrees(target_yaw),
            math.degrees(dyaw),
            self.turn_speed,
            feedback,
        )
        self.turn_ang(self.turn_speed, dyaw, guard=False, target_yaw=None)

        # 按实际转角更新估计，再决定是否精对齐
        spun = float(getattr(self, "_last_turn_spun", dyaw))
        self._est_pose["yaw"] = normalize_yaw(current_yaw + spun)
        self._rospy.sleep(0.25)

        remain = normalize_yaw(target_yaw - self._est_pose["yaw"])
        if self._odom_yaw is not None and 0.035 < abs(remain) < 0.45:
            self._rospy.loginfo(
                "%s face_yaw 精对齐补转 %.1f°",
                self._log_prefix,
                math.degrees(remain),
            )
            before = self._est_pose["yaw"]
            self.turn_ang(self.turn_speed, remain, guard=False, target_yaw=None)
            spun2 = float(getattr(self, "_last_turn_spun", remain))
            self._est_pose["yaw"] = normalize_yaw(before + spun2)
            self._rospy.sleep(0.15)
            remain = normalize_yaw(target_yaw - self._est_pose["yaw"])

        # 残差已很小：锁定到任务目标航向（0 / ±90 / 180）
        if abs(remain) <= 0.05:
            self._est_pose["yaw"] = normalize_yaw(target_yaw)
        self._rospy.loginfo(
            "%s face_yaw done: est=%.1f° target=%.1f° err=%.1f°",
            self._log_prefix,
            math.degrees(self._est_pose["yaw"]),
            math.degrees(target_yaw),
            math.degrees(normalize_yaw(target_yaw - self._est_pose["yaw"])),
        )
        return self._est_pose["yaw"]

    def move_body_delta(self, body_x, body_y):
        speed = self.speed
        if abs(body_y) > 0.02:
            self.go_linear_y(speed if body_y > 0 else -speed, abs(body_y))
            self._rospy.sleep(0.4)
        if abs(body_x) > 0.02:
            self.go_linear_x(speed if body_x > 0 else -speed, abs(body_x))
            self._rospy.sleep(0.4)

    def go_pose_open_loop(self, from_pose, to_pose):
        """开环走到 to_pose；避障开启时对白名单外障碍绕行，并用速度积分更新估计位姿。"""
        self.set_est_pose(from_pose)
        dx = to_pose["x"] - from_pose["x"]
        dy = to_pose["y"] - from_pose["y"]
        yaw = from_pose["yaw"]
        body_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        body_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        mode = "开环+避障" if self.avoidance_cfg.get("enabled", True) else "纯开环"
        self._rospy.loginfo(
            "%s %s map Δ(%.2f, %.2f) -> body (x=%.2f, y=%.2f) "
            "yaw %.1f° -> %.1f°",
            self._log_prefix,
            mode,
            dx,
            dy,
            body_x,
            body_y,
            math.degrees(yaw),
            math.degrees(float(to_pose.get("yaw", yaw))),
        )
        self.move_body_delta(body_x, body_y)
        # 绕障可能已改 yaw，用当前估计做到位航向对齐
        self.face_yaw(self._est_pose["yaw"], to_pose["yaw"])
        return dict(to_pose)
