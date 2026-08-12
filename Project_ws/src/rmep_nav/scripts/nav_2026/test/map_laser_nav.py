#!/usr/bin/python3
# coding=UTF-8
"""
基于雷达地图的闭环运动与避障（测试用）。

定位：AMCL (/amcl_pose) 或 TF map→base_link
静态障：/map OccupancyGrid 射线净空
动态障：/scan + laser_avoidance 扇区净空
融合：各方向取 min(激光, 地图)
运动：闭环追航点 + 侧移绕障 + face_yaw 到位对齐

复用模块：
  - laser_avoidance（急停/畅通/绕障方向）
  - openloop_duo（normalize_yaw / face_yaw / turn_ang）
  - map_occupancy.OccupancyMap（本目录）
"""

from __future__ import print_function

import math
import sys
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
_NAV_DIR = _TEST_DIR.parent
if str(_NAV_DIR) not in sys.path:
    sys.path.insert(0, str(_NAV_DIR))
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from map_occupancy import OccupancyMap  # noqa: E402
from openloop_duo import LaserOpenLoopNav, normalize_yaw  # noqa: E402
from laser_avoidance import (  # noqa: E402
    get_clearances,
    is_emergency,
    is_path_clear,
    pick_bypass_direction,
)


def _fuse_clearances(laser_c, map_c):
    out = {}
    for key in ("front", "back", "left", "right"):
        lv = float(laser_c.get(key, float("inf")))
        mv = float(map_c.get(key, float("inf")))
        out[key] = min(lv, mv)
    return out


def _yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class MapLaserNav(object):
    """AMCL 定位 + 地图/激光融合避障的闭环导航器。"""

    def __init__(
        self,
        speed=0.25,
        turn_speed=0.45,
        avoidance_cfg=None,
        map_cfg=None,
        log_prefix="[MapNav]",
        node_name="map_laser_nav",
        use_tf_fallback=True,
    ):
        import rospy
        import tf
        from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
        from nav_msgs.msg import OccupancyGrid
        from sensor_msgs.msg import LaserScan

        self._rospy = rospy
        self._Twist = Twist
        self._log_prefix = log_prefix
        self.speed = float(speed)
        self.turn_speed = float(turn_speed)
        self.use_tf_fallback = bool(use_tf_fallback)

        self.avoidance_cfg = dict(avoidance_cfg or {})
        self.avoidance_cfg.setdefault("enabled", True)
        self.avoidance_cfg.setdefault("fail_closed", True)
        self.avoidance_cfg.setdefault("lidar_mount", "rear")
        self.avoidance_cfg.setdefault("control_rate", 20)
        self.avoidance_cfg.setdefault("safe_distance", 0.20)
        self.avoidance_cfg.setdefault("side_safe_distance", 0.15)
        self.avoidance_cfg.setdefault("critical_distance", 0.10)
        self.avoidance_cfg.setdefault("emergency_stop_distance", 0.05)
        self.avoidance_cfg.setdefault("max_linear_speed", 0.30)
        self.avoidance_cfg.setdefault("bypass_speed", 0.10)
        self.avoidance_cfg.setdefault("move_timeout", 90.0)

        map_cfg = dict(map_cfg or {})
        self.map_max_range = float(map_cfg.get("max_range", 3.0))
        self.map_inflate = float(map_cfg.get("inflate_m", 0.08))
        # 0.06 过严：AMCL 噪声 + 全速冲点易过冲却永不触发到位
        self.pos_tol = float(map_cfg.get("pos_tol", 0.12))
        self.yaw_tol = float(map_cfg.get("yaw_tol", 0.08))
        self.approach_dist = float(map_cfg.get("approach_dist", 0.45))
        self.commit_dist = float(map_cfg.get("commit_dist", 0.25))
        self.overshoot_accept = float(
            map_cfg.get("overshoot_accept", max(self.pos_tol * 2.5, 0.25))
        )
        self.require_map = bool(map_cfg.get("require_map", True))
        self.require_amcl = bool(map_cfg.get("require_amcl", True))
        self.max_amcl_age = float(map_cfg.get("max_amcl_age", 1.0))

        self.occ = OccupancyMap(
            occupied_thresh=int(map_cfg.get("occupied_thresh", 50)),
            unknown_as_occupied=bool(map_cfg.get("unknown_as_occupied", False)),
        )

        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True)

        self.velocity_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.laser_data = None
        self._amcl_pose = None
        self._amcl_stamp = None
        self._tf = tf.TransformListener()

        # 复用开环模块的转向（face_yaw / turn_ang，到位对齐 guard=False）
        # 共用本对象的 /cmd_vel 与 /scan，避免双发布者互抢。
        # 必须在 /scan 订阅之前初始化，否则回调会访问未创建的 _turn。
        self._turn = LaserOpenLoopNav(
            speed=self.speed,
            turn_speed=self.turn_speed,
            avoidance_cfg=self.avoidance_cfg,
            zone_boxes=[],
            log_prefix=log_prefix,
            node_name=node_name + "_turn",
            wait_laser=False,
        )
        self._turn.velocity_publisher = self.velocity_publisher
        try:
            self._turn.laser_subscriber.unregister()
        except Exception:
            pass
        self._turn.laser_data = None

        # 注册 /scan /map /amcl_pose 订阅者（必须在 _turn 初始化之后）
        rospy.Subscriber("/scan", LaserScan, self._on_scan, queue_size=1)
        rospy.Subscriber("/map", OccupancyGrid, self._on_map, queue_size=1)
        rospy.Subscriber(
            "/amcl_pose",
            PoseWithCovarianceStamped,
            self._on_amcl,
            queue_size=1,
        )

        self._wait_ready()

    def _on_scan(self, msg):
        self.laser_data = msg
        self._turn.laser_data = msg

    def _on_map(self, msg):
        self.occ.update(msg)

    def _on_amcl(self, msg):
        p = msg.pose.pose.position
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        self._amcl_pose = {"x": float(p.x), "y": float(p.y), "yaw": float(yaw)}
        self._amcl_stamp = self._rospy.Time.now()

    def _wait_ready(self, timeout=30.0):
        rospy = self._rospy
        rospy.loginfo("%s 等待 /scan /map /amcl_pose ...", self._log_prefix)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            scan_ok = self.laser_data is not None
            map_ok = self.occ.ready or not self.require_map
            pose_ok = self.get_pose() is not None or not self.require_amcl
            if scan_ok and map_ok and pose_ok:
                rospy.loginfo(
                    "%s ready map=%dx%d amcl=%s",
                    self._log_prefix,
                    self.occ.width,
                    self.occ.height,
                    self._amcl_pose is not None,
                )
                return
            rospy.sleep(0.1)
        rospy.logwarn(
            "%s 就绪超时 scan=%s map=%s amcl=%s（可继续，失败时 fail-closed）",
            self._log_prefix,
            self.laser_data is not None,
            self.occ.ready,
            self._amcl_pose is not None,
        )

    def get_pose(self):
        """优先 AMCL；过期则尝试 TF map→base_link。"""
        rospy = self._rospy
        if self._amcl_pose is not None and self._amcl_stamp is not None:
            age = (rospy.Time.now() - self._amcl_stamp).to_sec()
            if age <= self.max_amcl_age:
                return dict(self._amcl_pose)
        if not self.use_tf_fallback:
            return dict(self._amcl_pose) if self._amcl_pose else None
        try:
            self._tf.waitForTransform(
                "map", "base_link", rospy.Time(0), rospy.Duration(0.2)
            )
            (trans, rot) = self._tf.lookupTransform(
                "map", "base_link", rospy.Time(0)
            )
            import tf

            yaw = tf.transformations.euler_from_quaternion(rot)[2]
            return {"x": float(trans[0]), "y": float(trans[1]), "yaw": float(yaw)}
        except Exception:
            return dict(self._amcl_pose) if self._amcl_pose else None

    def stop(self):
        self.velocity_publisher.publish(self._Twist())

    def fused_clearances(self, pose=None):
        pose = pose or self.get_pose()
        cfg = self.avoidance_cfg
        laser_c = get_clearances(self.laser_data, cfg)
        if pose is None or not self.occ.ready:
            return laser_c, laser_c, dict(laser_c)
        map_c = self.occ.sector_clearances(
            pose["x"],
            pose["y"],
            pose["yaw"],
            max_range=self.map_max_range,
            inflate_m=self.map_inflate,
        )
        return laser_c, map_c, _fuse_clearances(laser_c, map_c)

    def _publish_body(self, vx, vy, dt=None):
        max_v = float(self.avoidance_cfg.get("max_linear_speed", 0.30))
        vx = max(-max_v, min(max_v, float(vx)))
        vy = max(-max_v, min(max_v, float(vy)))
        twist = self._Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        self.velocity_publisher.publish(twist)

    def _path_clear_fused(self, vx, vy, fused, cfg):
        """用融合净空判断主运动方向是否畅通。"""
        safe = float(cfg.get("safe_distance", 0.20))
        side_safe = float(cfg.get("side_safe_distance", 0.15))
        if vx > 0.01:
            return fused["front"] > safe
        if vx < -0.01:
            return fused["back"] > safe
        if vy > 0.01:
            return fused["left"] > side_safe
        if vy < -0.01:
            return fused["right"] > side_safe
        return True

    def _obstacle_cost(self, dist, min_dist, inflation_dist):
        """TEB-style obstacle cost: zero beyond inflation_dist, infinite below min_dist."""
        if dist < min_dist:
            return float("inf")
        if inflation_dist <= min_dist or dist >= inflation_dist:
            return 0.0
        return (inflation_dist - dist) / (inflation_dist - min_dist)

    def _teb_candidates(self, cmd_vx, cmd_vy, max_v):
        """Generate a set of candidate (vx, vy) velocities in body frame."""
        candidates = [(0.0, 0.0)]
        # preferred velocity
        candidates.append((cmd_vx, cmd_vy))
        # 8 directions at multiple speeds (TEB-like homotopy sampling)
        angles = [
            0.0,
            math.pi * 0.25,
            math.pi * 0.5,
            math.pi * 0.75,
            math.pi,
            -math.pi * 0.75,
            -math.pi * 0.5,
            -math.pi * 0.25,
        ]
        speeds = [max_v * 0.3, max_v * 0.6, max_v]
        for speed in speeds:
            for angle in angles:
                candidates.append((speed * math.cos(angle), speed * math.sin(angle)))
        # de-duplicate with small tolerance
        seen = set()
        out = []
        for vx, vy in candidates:
            key = (round(vx, 4), round(vy, 4))
            if key not in seen:
                seen.add(key)
                out.append((vx, vy))
        return out

    def _teb_select_velocity(self, pose, target, cmd_vx, cmd_vy, fused, cfg):
        """TEB-like velocity selection: minimize obstacle + goal + velocity cost."""
        max_v = float(cfg.get("max_linear_speed", 0.30))
        min_dist = float(cfg.get("min_obstacle_dist", 0.10))
        inflation_dist = float(cfg.get("inflation_dist", 0.40))
        weight_obstacle = float(cfg.get("weight_obstacle", 50.0))
        weight_goal = float(cfg.get("weight_goal", 1.0))
        weight_velocity = float(cfg.get("weight_velocity", 1.0))
        weight_path = float(cfg.get("weight_path", 5.0))

        dx = target["x"] - pose["x"]
        dy = target["y"] - pose["y"]
        target_yaw = math.atan2(dy, dx)
        best_cost = float("inf")
        best_vx, best_vy = 0.0, 0.0
        # fallback: even if no candidate is strictly feasible, pick the least bad one
        fallback_cost = float("inf")
        fallback_vx, fallback_vy = 0.0, 0.0

        for vx, vy in self._teb_candidates(cmd_vx, cmd_vy, max_v):
            speed = math.hypot(vx, vy)
            if speed > max_v + 1e-6:
                continue

            # 1. obstacle cost: pick the sector this velocity points to
            v_yaw = math.atan2(vy, vx)
            if abs(v_yaw) < math.pi * 0.25:
                sector = "front"
            elif abs(v_yaw) > math.pi * 0.75:
                sector = "back"
            elif v_yaw > 0:
                sector = "left"
            else:
                sector = "right"
            dist = fused.get(sector, float("inf"))
            obs_cost = self._obstacle_cost(dist, min_dist, inflation_dist)
            feasible = obs_cost != float("inf")

            # 2. goal direction cost
            yaw = pose["yaw"]
            v_yaw_world = normalize_yaw(yaw + v_yaw) if speed > 1e-3 else yaw
            goal_diff = abs(normalize_yaw(v_yaw_world - target_yaw))
            goal_cost = goal_diff / math.pi

            # 3. velocity preference cost (stay close to desired velocity)
            vel_diff = math.hypot(vx - cmd_vx, vy - cmd_vy)
            velocity_cost = vel_diff / max(max_v, 1e-3)

            # 4. path cost: check if the candidate direction crosses occupied map cells
            path_cost = 0.0
            if self.occ.ready and speed > 1e-3:
                look = min(math.hypot(dx, dy), 0.5)
                steps = max(1, int(look / max(self.occ.resolution, 0.02)))
                step_x = (vx / speed) * (look / steps)
                step_y = (vy / speed) * (look / steps)
                blocked = False
                for i in range(1, steps + 1):
                    px = pose["x"] + step_x * i
                    py = pose["y"] + step_y * i
                    if self.occ.is_occupied_world(px, py, inflate_m=self.map_inflate):
                        blocked = True
                        break
                if blocked:
                    path_cost = 1.0

            total = (
                weight_obstacle * obs_cost
                + weight_goal * goal_cost
                + weight_velocity * velocity_cost
                + weight_path * path_cost
            )
            if feasible and total < best_cost:
                best_cost = total
                best_vx, best_vy = vx, vy
            if total < fallback_cost:
                fallback_cost = total
                fallback_vx, fallback_vy = vx, vy

        if best_cost == float("inf"):
            # No strictly feasible candidate: use the least-bad fallback, but cap its speed
            # to a very low creep so the robot can try to wiggle out instead of stopping.
            self._rospy.logwarn_throttle(
                1.0,
                "%s TEB no feasible candidate; fused F/B/L/R=%.2f/%.2f/%.2f/%.2f "
                "fallback=(%.2f,%.2f)",
                self._log_prefix,
                fused.get("front", -1),
                fused.get("back", -1),
                fused.get("left", -1),
                fused.get("right", -1),
                fallback_vx,
                fallback_vy,
            )
            # Scale fallback speed to bypass_speed or lower
            bypass_speed = float(cfg.get("bypass_speed", 0.10))
            scale = bypass_speed / max(math.hypot(fallback_vx, fallback_vy), 1e-3)
            if scale < 1.0:
                fallback_vx *= scale
                fallback_vy *= scale
            return fallback_vx, fallback_vy

        return best_vx, best_vy

    def face_yaw(self, target_yaw):
        pose = self.get_pose()
        if pose is None:
            self._rospy.logwarn("%s face_yaw 无定位，跳过", self._log_prefix)
            return False
        self._turn.set_est_pose(pose)
        self._turn.face_yaw(pose["yaw"], float(target_yaw))
        return True

    def go_pose(self, target, label="goal"):
        """
        闭环走到地图系目标 {x,y,yaw}：
        每周期读 AMCL → 算 body 速度 → 地图+激光融合避障 → 到位 face_yaw。

        近点策略（避免冲过航点却不停）：
          - approach_dist 内按距离线性减速
          - commit_dist 内直冲目标，关闭 TEB 绕障/地图线段阻挡（仍保留急停）
          - 一旦进入 overshoot_accept 又离开 pos_tol（过冲），停车并拉回
        """
        rospy = self._rospy
        cfg = self.avoidance_cfg
        dt = 1.0 / float(cfg.get("control_rate", 20))
        deadline = rospy.Time.now().to_sec() + float(cfg.get("move_timeout", 90.0))
        target = {
            "x": float(target["x"]),
            "y": float(target["y"]),
            "yaw": float(target.get("yaw", 0.0)),
        }

        rospy.loginfo(
            "%s -> %s (%.3f, %.3f, yaw=%.3f) map+laser closed-loop "
            "pos_tol=%.3f approach=%.2f commit=%.2f",
            self._log_prefix,
            label,
            target["x"],
            target["y"],
            target["yaw"],
            self.pos_tol,
            self.approach_dist,
            self.commit_dist,
        )

        phase = "normal"
        stuck_start = None
        recovery_duration = 0.0
        min_dist = float("inf")
        ever_near = False

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            if now > deadline:
                rospy.logwarn(
                    "%s %s 超时，停车（min|Δ|=%.3f）",
                    self._log_prefix,
                    label,
                    min_dist if min_dist < float("inf") else -1.0,
                )
                self.stop()
                return False

            pose = self.get_pose()
            if pose is None:
                self.stop()
                if cfg.get("fail_closed", True):
                    rospy.logwarn_throttle(
                        1.0, "%s 无 AMCL/TF，fail-closed 停车", self._log_prefix
                    )
                rospy.sleep(dt)
                continue

            if self.laser_data is None and cfg.get("fail_closed", True):
                self.stop()
                rospy.logwarn_throttle(1.0, "%s 无激光，停车", self._log_prefix)
                rospy.sleep(dt)
                continue

            dx = target["x"] - pose["x"]
            dy = target["y"] - pose["y"]
            dist = math.hypot(dx, dy)
            prev_min = min_dist
            if dist < min_dist:
                min_dist = dist
            if dist <= self.overshoot_accept:
                ever_near = True

            if dist <= self.pos_tol:
                rospy.loginfo(
                    "%s %s 位置到达 |Δ|=%.3f，开始 face_yaw",
                    self._log_prefix,
                    label,
                    dist,
                )
                break

            # 已进入近区后又远离最近点：说明冲过/擦过航点
            leaving = ever_near and dist > prev_min + 0.06
            accept_after_pass = max(self.pos_tol * 1.8, 0.18)
            if leaving and min_dist <= accept_after_pass:
                rospy.logwarn(
                    "%s %s 过冲接受 |Δ|=%.3f min=%.3f（曾靠近航点），停车对齐",
                    self._log_prefix,
                    label,
                    dist,
                    min_dist,
                )
                break
            if leaving and min_dist <= self.overshoot_accept:
                rospy.logwarn(
                    "%s %s 过冲回收 |Δ|=%.3f min=%.3f，停车拉回",
                    self._log_prefix,
                    label,
                    dist,
                    min_dist,
                )
                self.stop()
                rospy.sleep(0.15)
                ever_near = False
                min_dist = dist
                phase = "normal"
                stuck_start = None
                recovery_duration = 0.0

            rospy.loginfo_throttle(
                1.0,
                "%s %s 追点 |Δ|=%.3f pose=(%.3f,%.3f) tgt=(%.3f,%.3f) phase=%s",
                self._log_prefix,
                label,
                dist,
                pose["x"],
                pose["y"],
                target["x"],
                target["y"],
                phase,
            )

            yaw = pose["yaw"]
            body_x = math.cos(yaw) * dx + math.sin(yaw) * dy
            body_y = -math.sin(yaw) * dx + math.cos(yaw) * dy

            # 近点线性减速，最低保底速度保证底盘有响应
            max_v = float(cfg.get("max_linear_speed", 0.30))
            creep = max(0.06, float(cfg.get("bypass_speed", 0.10)) * 0.6)
            if dist < self.approach_dist:
                speed_cap = creep + (self.speed - creep) * (
                    dist / max(self.approach_dist, 1e-3)
                )
                speed_cap = min(speed_cap, self.speed, max_v)
            else:
                speed_cap = min(self.speed, max_v)
            if dist < self.commit_dist:
                speed_cap = min(speed_cap, max(creep, self.speed * 0.35))

            scale = speed_cap / max(dist, 1e-3)
            cmd_vx = max(-speed_cap, min(speed_cap, body_x * scale))
            cmd_vy = max(-speed_cap, min(speed_cap, body_y * scale))

            laser_c, map_c, fused = self.fused_clearances(pose)
            # 紧急：用激光为主（地图滞后），方向随当前指令
            if is_emergency(self.laser_data, cfg, cmd_vx, cmd_vy, 0.0):
                self.stop()
                rospy.logwarn_throttle(
                    1.0,
                    "%s 紧急停车 L(f=%.2f) M(f=%.2f)",
                    self._log_prefix,
                    laser_c["front"],
                    map_c.get("front", -1),
                )
                # 近点不后退（避免把已到达附近的车推离航点）
                if dist > self.commit_dist:
                    retreat = self.speed * 0.4
                    if abs(cmd_vx) >= abs(cmd_vy):
                        self._publish_body(
                            -math.copysign(retreat, cmd_vx or 1.0), 0.0
                        )
                    else:
                        self._publish_body(
                            0.0, -math.copysign(retreat, cmd_vy or 1.0)
                        )
                rospy.sleep(dt)
                phase = "normal"
                stuck_start = None
                continue

            committing = dist <= self.commit_dist

            # 目标直线是否穿过地图占用（近点直冲时忽略，避免区界膨胀把车推飞）
            map_line_blocked = False
            if self.occ.ready and not committing:
                look = min(dist, 0.40)
                map_line_blocked = self.occ.segment_blocked(
                    pose["x"],
                    pose["y"],
                    pose["x"] + math.cos(math.atan2(dy, dx)) * look,
                    pose["y"] + math.sin(math.atan2(dy, dx)) * look,
                    inflate_m=self.map_inflate,
                )

            if committing:
                # 末段直冲：只做激光前方畅通检查，不做 TEB 绕障
                clear = is_path_clear(self.laser_data, cmd_vx, cmd_vy, cfg)
                if clear:
                    self._publish_body(cmd_vx, cmd_vy)
                else:
                    # 被挡则蠕动侧向分量清零，沿主轴试探；仍挡则停车等
                    primary_vx = cmd_vx if abs(cmd_vx) >= abs(cmd_vy) else 0.0
                    primary_vy = cmd_vy if abs(cmd_vy) > abs(cmd_vx) else 0.0
                    if is_path_clear(self.laser_data, primary_vx, primary_vy, cfg):
                        self._publish_body(primary_vx, primary_vy)
                    else:
                        self.stop()
                        rospy.logwarn_throttle(
                            1.0,
                            "%s %s 近点直冲被挡 |Δ|=%.3f，等待净空",
                            self._log_prefix,
                            label,
                            dist,
                        )
                stuck_start = None
                rospy.sleep(dt)
                continue

            clear = self._path_clear_fused(cmd_vx, cmd_vy, fused, cfg)
            laser_clear = is_path_clear(self.laser_data, cmd_vx, cmd_vy, cfg)
            clear = clear and laser_clear and not map_line_blocked

            if phase == "normal":
                if clear:
                    self._publish_body(cmd_vx, cmd_vy)
                    stuck_start = None
                else:
                    # TEB-like continuous bypass: sample candidate velocities and
                    # pick the one with lowest obstacle + goal + velocity cost.
                    teb_vx, teb_vy = self._teb_select_velocity(
                        pose, target, cmd_vx, cmd_vy, fused, cfg
                    )
                    if math.hypot(teb_vx, teb_vy) < 1e-3:
                        # No feasible direction -> recovery mode.
                        phase = "recovery"
                        stuck_start = rospy.Time.now().to_sec()
                        rospy.logwarn_throttle(
                            1.0, "%s 无可行速度，进入后退恢复", self._log_prefix
                        )
                        continue
                    self._publish_body(teb_vx, teb_vy)
                    if stuck_start is None:
                        stuck_start = rospy.Time.now().to_sec()
                    elif (
                        rospy.Time.now().to_sec() - stuck_start
                        > float(cfg.get("stuck_timeout", 5.0))
                    ):
                        phase = "recovery"
                        rospy.logwarn(
                            "%s 长时间未摆脱障碍，进入后退恢复", self._log_prefix
                        )
            elif phase == "recovery":
                # Back away slowly until we find a feasible direction again.
                # 已靠近过目标时不要长距离后退，否则会冲离取货区
                if ever_near or dist < self.approach_dist:
                    phase = "normal"
                    recovery_duration = 0.0
                    stuck_start = None
                    rospy.loginfo(
                        "%s 近目标跳过后退恢复，改直冲 |Δ|=%.3f",
                        self._log_prefix,
                        dist,
                    )
                else:
                    retreat = -self.speed * 0.4
                    self._publish_body(retreat, 0.0)
                    recovery_duration += dt
                    if recovery_duration >= float(cfg.get("recovery_duration", 1.0)):
                        phase = "normal"
                        recovery_duration = 0.0
                        stuck_start = None
                        rospy.loginfo("%s 后退恢复结束，继续追点", self._log_prefix)

            rospy.sleep(dt)

        self.stop()
        rospy.sleep(0.3)
        ok = self.face_yaw(target["yaw"])
        self.stop()
        pose = self.get_pose()
        if pose:
            rospy.loginfo(
                "%s %s 完成 pose=(%.3f, %.3f, %.3f) face=%s",
                self._log_prefix,
                label,
                pose["x"],
                pose["y"],
                pose["yaw"],
                ok,
            )
        return True
