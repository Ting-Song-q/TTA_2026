#!/usr/bin/python3
# coding=UTF-8
"""nav_open_2026：转速（cmd_vel）× 时间 的纯开环控制版。

平移速度、旋转角速度、各航段运动时间均在同目录 nav_open_2026.yaml 中填写：
  - timed.speed / timed.turn_speed
  - segment_motion.<段名>.duration_x / duration_y / duration_turn

依赖（与本脚本同目录 main/）：
  - laser_avoidance.py
  - rescue_protocol.py
  - nav_open_2026.yaml
  - grasp/   （mobile_grasp 取货）
  - arm/     （place_apriltag 放置、reconnect_bus 重连）
  - output/  （放置用相机/手眼标定）

取货区：到达后调用 grasp/mobile_grasp.py
  （臂观察位 → 前进对齐 → 横移视觉对位 → 再前进 → 腕部相机抓取 → 后退复位）
  抓取结束后再调用 arm/reconnect_bus.py 握手舵机（带重试）。
  参数见 yaml 的 mobile_grasp / arm_reconnect；可用 --skip-grasp 仅联调导航。

装货区：到达后调用 arm/place_apriltag.py
  （放置观察位持块 → AprilTag 粗接近 → 视觉伺服 → 下降开爪 → 回 initial）
  未识别到 AprilTag 时小车略微前移后重试（yaml: no_tag_forward_m / no_tag_retries）。
  参数见 yaml 的 place_apriltag；可用 --skip-place 仅联调导航。

可选 obstacle_sidestep：出发前读雷达，前/后净空 < 阈值则：
  - 前方有障：取货「左→走→右」；装货「左→走→再左」
  - 仅后方有障：取货正常走；装货段先左移 0.4m → 正常行驶 → 再左移 0.4m

装货区 → 救援点：出发前读横移方向净空；有障则：
  后退 0.6m → 横移 → 前进 0.6m → 再继续 X/转向

用法：
  python3 nav_open_2026.py --skip-drone --rescue 3
  python3 nav_open_2026.py --config nav_open_2026.yaml --skip-drone --rescue 3
  python3 nav_open_2026.py --skip-drone --skip-grasp --skip-place --rescue 3
"""

from __future__ import print_function

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import rospy
import yaml
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

# main/：配置与依赖模块均在同目录；上一级 nav_2026/ 仅作兼容回退
_HERE = Path(__file__).resolve().parent
_NAV_2026 = _HERE.parent
_DEFAULT_CONFIG = _HERE / "nav_open_2026.yaml"
for _p in (str(_HERE), str(_NAV_2026)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from laser_avoidance import get_clearances  # noqa: E402
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


def resolve_tta_root(configured=None):
    """定位含 grasp/ 与 arm/ 的包根目录（默认即本脚本所在 main/）。"""
    markers = (
        Path("grasp") / "mobile_grasp.py",
        Path("arm") / "place_apriltag.py",
    )

    def _ok(root):
        return any((root / m).is_file() for m in markers)

    if configured:
        root = Path(str(configured)).expanduser().resolve()
        if not _ok(root):
            raise FileNotFoundError(
                "tta_root 无效，未找到 grasp/mobile_grasp.py 或 arm/place_apriltag.py: %s"
                % root
            )
        return root
    for parent in [_HERE] + list(_HERE.parents):
        for candidate in (parent, parent / "tta"):
            if _ok(candidate):
                return candidate.resolve()
    raise FileNotFoundError(
        "无法自动定位含 grasp/ 与 arm/ 的目录，请在 yaml 中设置 "
        "mobile_grasp.tta_root 或 place_apriltag.tta_root（通常为本 main/）"
    )


def resolve_under_root(root, value, default_rel):
    """相对路径优先相对 tta_root；已是绝对路径则直接用。"""
    text = str(value if value not in (None, "") else default_rel)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _candidate_conda_env_pythons(env_name):
    """常见 conda/mamba 环境下的 python 路径。"""
    home = Path.home()
    roots = []
    for key in ("CONDA_ROOT", "CONDA_PREFIX", "MAMBA_ROOT_PREFIX"):
        val = os.environ.get(key)
        if val:
            roots.append(Path(val))
    # CONDA_PREFIX 可能已是 env 目录
    for root in list(roots):
        if root.name == env_name and (root / "bin" / "python").is_file():
            yield root / "bin" / "python"
        parent = root.parent
        if parent.name == "envs":
            roots.append(parent.parent)
    bases = [
        home / "miniconda3",
        home / "anaconda3",
        home / "mambaforge",
        home / "miniforge3",
        Path("/opt/conda"),
        Path("/home/tta/miniconda3"),
        Path("/home/tta/anaconda3"),
        Path("/home/tta/mambaforge"),
        Path("/home/tta/miniforge3"),
    ]
    for base in bases + roots:
        for rel in (
            Path("envs") / env_name / "bin" / "python",
            Path(env_name) / "bin" / "python",
        ):
            cand = base / rel
            if cand.is_file():
                yield cand.resolve()


def resolve_grasp_python_cmd(configured=None, env_name="lerobot"):
    """解析取货子进程解释器。

    支持：
      - 绝对路径: /home/tta/miniconda3/envs/lerobot/bin/python
      - conda:lerobot  → conda run -n lerobot ...
      - 空 / python3 / auto → 自动找 conda 环境 lerobot 的 python
    返回 argv 前缀列表（通常长度为 1）。
    """
    text = str(configured or "").strip()
    if text.startswith("conda:"):
        name = text.split(":", 1)[1].strip() or env_name
        conda = shutil.which("conda")
        if not conda:
            raise FileNotFoundError(
                "yaml 写了 conda:%s 但找不到 conda 命令" % name
            )
        return [
            conda,
            "run",
            "-n",
            name,
            "--no-capture-output",
            "python",
        ]

    if text and text not in ("python3", "python", "auto", "lerobot"):
        path = Path(text).expanduser()
        if path.is_file():
            return [str(path.resolve())]
        which = shutil.which(text)
        if which:
            return [which]
        raise FileNotFoundError("mobile_grasp.python 无效: %s" % text)

    # 优先当前已激活的同名环境
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and Path(conda_prefix).name == env_name:
        py = Path(conda_prefix) / "bin" / "python"
        if py.is_file():
            return [str(py.resolve())]

    for cand in _candidate_conda_env_pythons(env_name):
        return [str(cand)]

    conda = shutil.which("conda")
    if conda:
        return [
            conda,
            "run",
            "-n",
            env_name,
            "--no-capture-output",
            "python",
        ]

    # 最后回退：当前进程（系统 python，通常没有 lerobot）
    fallback = sys.executable or shutil.which("python3") or "python3"
    return [str(fallback)]

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

    def run_segment(
        self,
        body_x_sign,
        body_y_sign,
        yaw_delta_sign,
        seg,
        do_y=True,
        do_x=True,
        do_turn=True,
    ):
        speed = float(seg.get("speed", self.speed))
        turn_speed = float(seg.get("turn_speed", self.turn_speed))
        dx_t = float(seg.get("duration_x", 0.0))
        dy_t = float(seg.get("duration_y", 0.0))
        dt_t = float(seg.get("duration_turn", 0.0))

        rospy.loginfo(
            "%s segment speed=%.2f turn=%.2f "
            "dur_x=%.2fs dur_y=%.2fs dur_turn=%.2fs signs=(x=%+.0f,y=%+.0f,yaw=%+.0f) "
            "axes=(y=%s,x=%s,turn=%s)",
            self.log_prefix,
            speed,
            turn_speed,
            dx_t,
            dy_t,
            dt_t,
            body_x_sign,
            body_y_sign,
            yaw_delta_sign,
            do_y,
            do_x,
            do_turn,
        )

        if do_y and dy_t > 1e-4 and abs(body_y_sign) > 1e-9:
            self.go_linear_y_timed(
                speed if body_y_sign > 0 else -speed, dy_t
            )
            if self.segment_pause_sec > 0:
                rospy.sleep(self.segment_pause_sec)

        if do_x and dx_t > 1e-4 and abs(body_x_sign) > 1e-9:
            self.go_linear_x_timed(
                speed if body_x_sign > 0 else -speed, dx_t
            )
            if self.segment_pause_sec > 0:
                rospy.sleep(self.segment_pause_sec)

        if do_turn and dt_t > 1e-4 and abs(yaw_delta_sign) > 1e-9:
            self.turn_ang_timed(
                turn_speed, dt_t, direction=1 if yaw_delta_sign > 0 else -1
            )
            if self.segment_pause_sec > 0:
                rospy.sleep(self.segment_pause_sec)

    def go_linear_x_distance(self, distance, speed=None):
        """车体前后平移固定距离：distance>0 前进，<0 后退。"""
        dist = float(distance)
        if abs(dist) < 1e-4:
            return
        spd = abs(float(speed if speed is not None else self.speed))
        if spd < 1e-6:
            return
        duration = abs(dist) / spd
        rospy.loginfo(
            "%s linear_x_distance: dist=%+.3fm speed=%.3f duration=%.3fs",
            self.log_prefix,
            dist,
            spd if dist > 0 else -spd,
            duration,
        )
        self.go_linear_x_timed(spd if dist > 0 else -spd, duration)
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

    def go_lateral_y(self, distance, speed=None):
        """车体侧移：distance>0 表示任务语义「向左」，<0 表示「向右」。

        实车 cmd_vel.linear.y 与常见 +Y=左 相反，此处对输出取反，
        保证上层「左移用正距离、右移用负距离」与实车一致。
        """
        dist = float(distance)
        if abs(dist) < 1e-4:
            return
        spd = abs(float(speed if speed is not None else self.speed))
        if spd < 1e-6:
            return
        duration = abs(dist) / spd
        # 语义左(+dist) → 发 -vy；语义右(-dist) → 发 +vy
        signed_speed = -spd if dist > 0 else spd
        rospy.loginfo(
            "%s lateral_y: intent=%s dist=%+.3fm cmd_vy=%.3f duration=%.3fs",
            self.log_prefix,
            "left" if dist > 0 else "right",
            dist,
            signed_speed,
            duration,
        )
        self.go_linear_y_timed(signed_speed, duration)
        if self.segment_pause_sec > 0:
            rospy.sleep(self.segment_pause_sec)


class NavTimed(object):
    """救援任务：运动由 YAML 中的 speed + duration 驱动。"""

    def __init__(
        self,
        config_path=None,
        skip_drone=False,
        skip_grasp=False,
        skip_place=False,
        rescue_zone=2,
        cmd_vel_topic=None,
        autostart=True,
    ):
        rospy.init_node("nav_open_2026", anonymous=True)
        self.skip_drone = bool(skip_drone)
        self.skip_grasp = bool(skip_grasp)
        self.skip_place = bool(skip_place)
        self.rescue_zone = int(rescue_zone)
        self.config_path = Path(config_path or _DEFAULT_CONFIG)
        self.config = load_timed_config(self.config_path)

        timed = dict(self.config.get("timed") or {})
        self.mission_cfg = dict(self.config.get("mission") or {})
        self.segment_motion = dict(self.config.get("segment_motion") or {})
        self.mobile_grasp_cfg = dict(self.config.get("mobile_grasp") or {})
        self.place_apriltag_cfg = dict(self.config.get("place_apriltag") or {})
        self.arrive_pause_sec = float(timed.get("arrive_pause_sec", 0.50))
        self.obstacle_cfg = dict(self.config.get("obstacle_sidestep") or {})
        # none | front | back（出发前一次判定）
        self._obstacle_mode = "none"
        self._scan = None
        self._scan_stamp = None

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

        scan_topic = str(self.obstacle_cfg.get("scan_topic", "/scan"))
        self._scan_sub = rospy.Subscriber(
            scan_topic, LaserScan, self._on_scan, queue_size=1
        )

        rospy.loginfo(
            "[Mission] nav_open_2026 | config=%s | skip_drone=%s "
            "skip_grasp=%s skip_place=%s | rescue=%s | speed=%.2f turn=%.2f | scan=%s",
            self.config_path,
            self.skip_drone,
            self.skip_grasp,
            self.skip_place,
            self.rescue_zone,
            speed,
            turn_speed,
            scan_topic,
        )
        if autostart:
            ok = self.run_mission()
            rospy.loginfo("[Mission] 结束 ok=%s", ok)

    def _on_scan(self, msg):
        self._scan = msg
        self._scan_stamp = rospy.Time.now().to_sec()

    def stop(self):
        self.nav.stop()

    def _wait_scan(self, timeout):
        timeout = max(0.0, float(timeout))
        t0 = rospy.Time.now().to_sec()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self._scan is not None:
                return self._scan
            if rospy.Time.now().to_sec() - t0 >= timeout:
                return None
            rate.sleep()
        return None

    def _decide_obstacle_mode(self):
        """出发前读雷达：区分前方/后方 0.6m 内障碍。

        返回:
          - "front": 前方有障（若前后都有，优先前方）→ 取货/装货用原左右侧移
          - "back":  仅后方有障 → 取货正常；装货左移→行驶→再左移
          - "none":  无障或未启用/无激光
        """
        cfg = self.obstacle_cfg
        if not cfg.get("enable", True):
            rospy.loginfo("[Mission] obstacle_sidestep 已关闭")
            return "none"

        check_dist = float(cfg.get("check_distance", 0.6))
        wait_timeout = float(cfg.get("scan_wait_timeout", 3.0))

        scan = self._wait_scan(wait_timeout)
        if scan is None:
            rospy.logwarn(
                "[Mission] 等待激光超时 %.1fs，视为无障碍",
                wait_timeout,
            )
            return "none"

        clearances = get_clearances(scan, self._laser_cfg())
        front = float(clearances["front"])
        back = float(clearances["back"])
        front_hit = front < check_dist
        back_hit = back < check_dist

        if front_hit:
            mode = "front"
            action = "前方障碍→取货/装货左右侧移绕障"
        elif back_hit:
            mode = "back"
            action = "后方障碍→取货直行；装货左移→行驶→再左移"
        else:
            mode = "none"
            action = "通路畅通"

        rospy.loginfo(
            "[Mission] 雷达判障 front=%.3fm back=%.3fm thresh=%.3fm → %s (%s)",
            front,
            back,
            check_dist,
            mode,
            action,
        )
        return mode

    def _sidestep_speed(self):
        cfg = self.obstacle_cfg
        if "lateral_speed" in cfg and cfg.get("lateral_speed") is not None:
            return abs(float(cfg["lateral_speed"]))
        return abs(float(self.nav.speed))

    def _sidestep_distance(self):
        """前方障碍用的侧移距离（左出 / 右回）。"""
        return abs(float(self.obstacle_cfg.get("lateral_distance", 0.5)))

    def _rear_loading_lateral_distance(self):
        """后方障碍时装货段侧移距离（默认 0.4m）。"""
        return abs(
            float(self.obstacle_cfg.get("rear_loading_lateral_distance", 0.4))
        )

    def _laser_cfg(self):
        cfg = self.obstacle_cfg
        return {
            "lidar_mount": cfg.get("lidar_mount", "rear"),
            "lidar_to_body": dict(cfg.get("lidar_to_body") or {}),
            "sector_half_width": cfg.get("sector_half_width", 25),
            "side_sector_half_width": cfg.get("side_sector_half_width", 55),
            "self_hit_margin": cfg.get("self_hit_margin", 0.03),
        }

    def _check_rescue_lateral_obstacle(self, body_y_sign):
        """装货→救援：判断即将横移的一侧是否有障碍。

        body_y_sign>0 → 发 +vy，按 laser_avoidance 约定查左侧；
        body_y_sign<0 → 发 -vy，查右侧。
        """
        cfg = self.obstacle_cfg
        if not cfg.get("enable", True):
            return False
        if not cfg.get("rescue_lateral_check_enable", True):
            rospy.loginfo("[Mission] rescue 横移判障已关闭")
            return False
        if abs(float(body_y_sign)) < 1e-9:
            return False

        wait_timeout = float(cfg.get("scan_wait_timeout", 3.0))
        thresh = float(
            cfg.get(
                "rescue_lateral_check_distance",
                cfg.get("check_distance", 0.6),
            )
        )
        scan = self._wait_scan(wait_timeout)
        if scan is None:
            rospy.logwarn(
                "[Mission] 救援横移判障：等待激光超时 %.1fs，视为无障碍",
                wait_timeout,
            )
            return False

        clearances = get_clearances(scan, self._laser_cfg())
        # 与 run_segment / compute_bypass_twist 一致：+vy→left，-vy→right
        side = "left" if body_y_sign > 0 else "right"
        clearance = float(clearances[side])
        hit = clearance < thresh
        rospy.loginfo(
            "[Mission] 救援横移判障 side=%s clear=%.3fm thresh=%.3fm → %s",
            side,
            clearance,
            thresh,
            "有障碍(后退再横移)" if hit else "畅通",
        )
        return hit

    def _rescue_lateral_backoff_m(self):
        return abs(
            float(self.obstacle_cfg.get("rescue_lateral_backoff_m", 0.6))
        )

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

        # 方向符号在侧移前按原航点差计算，避免侧移改变 signs 导致段时长语义漂移
        from_pose = self.nav.get_est_pose()
        sx, sy, st, body_x, body_y, dyaw = self._signs_from_poses(from_pose, pose)

        mode = self._obstacle_mode
        # front：取货/装货 左→走→右；back：仅装货 左→走→左；pickup 在 back 时直行
        front_sidestep = mode == "front" and zone_name in ("pickup", "loading")
        rear_loading = mode == "back" and zone_name == "loading"
        dy_t = float(seg.get("duration_y", 0.0))
        rescue_lateral_obs = (
            zone_name == "rescue"
            and dy_t > 1e-4
            and abs(sy) > 1e-9
            and self._check_rescue_lateral_obstacle(sy)
        )

        rospy.loginfo(
            "[Mission] goto %s via %s (%.3f, %.3f, yaw=%.3f) "
            "body≈(%.3f, %.3f) dyaw=%.1f° obstacle_mode=%s "
            "front_sidestep=%s rear_loading=%s rescue_lateral_obs=%s",
            label,
            key,
            pose["x"],
            pose["y"],
            pose["yaw"],
            body_x,
            body_y,
            math.degrees(dyaw),
            mode,
            front_sidestep,
            rear_loading,
            rescue_lateral_obs,
        )

        spd = self._sidestep_speed()
        if front_sidestep:
            dist = self._sidestep_distance()
            rospy.loginfo(
                "[Mission] %s 前方绕障：先向左平移 %.2fm", label, dist
            )
            self.nav.go_lateral_y(+dist, speed=spd)
        elif rear_loading:
            dist = self._rear_loading_lateral_distance()
            rospy.loginfo(
                "[Mission] %s 后方绕障：先向左平移 %.2fm", label, dist
            )
            self.nav.go_lateral_y(+dist, speed=spd)

        if rescue_lateral_obs:
            back_m = self._rescue_lateral_backoff_m()
            seg_speed = abs(float(seg.get("speed", self.nav.speed)))
            rospy.loginfo(
                "[Mission] %s 横移方向有障：先后退 %.2fm → 横移 → 前进 %.2fm → 再继续",
                label,
                back_m,
                back_m,
            )
            self.nav.go_linear_x_distance(-back_m, speed=seg_speed)
            self.nav.run_segment(
                sx, sy, st, seg, do_y=True, do_x=False, do_turn=False
            )
            self.nav.go_linear_x_distance(+back_m, speed=seg_speed)
            self.nav.run_segment(
                sx, sy, st, seg, do_y=False, do_x=True, do_turn=True
            )
        else:
            self.nav.run_segment(sx, sy, st, seg)

        if front_sidestep:
            dist = self._sidestep_distance()
            if zone_name == "loading":
                # 装货段含约 180° 转向：车体左右相对场地对调，到位后再左移才是回走廊
                rospy.loginfo(
                    "[Mission] %s 到位后向左平移 %.2fm（180°后仍用左移回正）",
                    label,
                    dist,
                )
                self.nav.go_lateral_y(+dist, speed=spd)
            else:
                # 取货：未掉头，右移回正
                rospy.loginfo(
                    "[Mission] %s 到位后向右平移 %.2fm", label, dist
                )
                self.nav.go_lateral_y(-dist, speed=spd)
        elif rear_loading:
            dist = self._rear_loading_lateral_distance()
            rospy.loginfo(
                "[Mission] %s 到位后再向左平移 %.2fm", label, dist
            )
            self.nav.go_lateral_y(+dist, speed=spd)

        self.nav.set_est_pose(pose)
        self._current_zone = zone_name
        self._current_zone_id = zone_id
        self.stop()
        if self.arrive_pause_sec > 0:
            rospy.sleep(self.arrive_pause_sec)
        return True

    def _run_external_script(
        self, label, cmd, cwd, timeout_sec, success_log=None, return_code=False
    ):
        """停车后启动外部 Python 脚本，结束后再停车。

        return_code=False（默认）：返回是否成功（exit==0）。
        return_code=True：返回进程 exit code；超时/无法启动返回 None。
        """
        self.stop()
        rospy.sleep(0.2)
        rospy.loginfo(
            "[Mission] 启动%s: cwd=%s cmd=%s timeout=%.0fs",
            label,
            cwd,
            " ".join(cmd),
            timeout_sec,
        )
        completed = None
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(cwd),
                timeout=timeout_sec if timeout_sec > 0 else None,
            )
        except subprocess.TimeoutExpired:
            rospy.logerr(
                "[Mission] %s 超时 (%.0fs)，已停止底盘", label, timeout_sec
            )
            self.stop()
            return None if return_code else False
        except OSError as exc:
            rospy.logerr("[Mission] 无法启动 %s: %s", label, exc)
            return None if return_code else False
        finally:
            self.stop()

        code = None if completed is None else int(completed.returncode)
        if code == 0:
            if success_log:
                rospy.loginfo("[Mission] %s", success_log)
        else:
            rospy.logerr("[Mission] %s 失败 exit_code=%s", label, code)

        if return_code:
            return code
        return code == 0

    def run_mobile_grasp(self):
        """取货区：调用 grasp/mobile_grasp.py（底盘对齐 + SO-101 视觉抓取）。"""
        cfg = self.mobile_grasp_cfg
        enabled = bool(cfg.get("enable", True)) and not self.skip_grasp
        if not enabled:
            rospy.logwarn("[Mission] mobile_grasp 已跳过（enable=false 或 --skip-grasp）")
            return True

        try:
            tta_root = resolve_tta_root(cfg.get("tta_root"))
        except FileNotFoundError as exc:
            rospy.logerr("[Mission] %s", exc)
            return False

        script = resolve_under_root(
            tta_root, cfg.get("script"), "grasp/mobile_grasp.py"
        )
        grasp_config = resolve_under_root(
            tta_root, cfg.get("config"), "grasp/configs/mobile_grasp.yaml"
        )
        if not script.is_file():
            rospy.logerr("[Mission] 取货脚本不存在: %s", script)
            return False
        if not grasp_config.is_file():
            rospy.logerr("[Mission] 取货配置不存在: %s", grasp_config)
            return False

        try:
            python_cmd = resolve_grasp_python_cmd(cfg.get("python"))
        except FileNotFoundError as exc:
            rospy.logerr("[Mission] %s", exc)
            return False

        cmd = python_cmd + [
            "-u",
            str(script),
            "--config",
            str(grasp_config),
            "--yes",
        ]
        return self._run_external_script(
            "mobile_grasp",
            cmd,
            tta_root,
            float(cfg.get("timeout_sec", 300.0)),
            success_log="mobile_grasp 成功",
        )

    def run_arm_reconnect(self, after="grasp"):
        """抓取结束后重新连接机械臂舵机（带重试，不断力矩）。"""
        cfg = dict(self.config.get("arm_reconnect") or {})
        if not bool(cfg.get("enable", True)):
            rospy.logwarn("[Mission] arm_reconnect 已跳过（enable=false）")
            return True

        tta_root_cfg = cfg.get("tta_root") or self.mobile_grasp_cfg.get(
            "tta_root"
        )
        try:
            tta_root = resolve_tta_root(tta_root_cfg)
        except FileNotFoundError as exc:
            rospy.logerr("[Mission] %s", exc)
            return False

        arm_root = tta_root / "arm"
        script = resolve_under_root(
            tta_root, cfg.get("script"), "arm/reconnect_bus.py"
        )
        og_config = resolve_under_root(
            tta_root,
            cfg.get("og_config") or cfg.get("config"),
            "arm/reset_and_grasp.yaml",
        )
        if not script.is_file():
            rospy.logerr("[Mission] 重连脚本不存在: %s", script)
            return False
        if not og_config.is_file():
            rospy.logerr("[Mission] 重连配置不存在: %s", og_config)
            return False

        python_cfg = cfg.get("python")
        if python_cfg in (None, ""):
            python_cfg = self.mobile_grasp_cfg.get("python", "auto")
        try:
            python_cmd = resolve_grasp_python_cmd(python_cfg)
        except FileNotFoundError as exc:
            rospy.logerr("[Mission] %s", exc)
            return False

        max_attempts = int(cfg.get("connect_max_attempts", 4))
        retry_s = float(cfg.get("connect_retry_s", 1.0))
        hold_s = float(cfg.get("hold_s", 0.3))
        cmd = python_cmd + [
            "-u",
            str(script),
            "--yes",
            "--config",
            str(og_config),
            "--max-attempts",
            str(max_attempts),
            "--retry-s",
            str(retry_s),
            "--hold-s",
            str(hold_s),
        ]
        if bool(cfg.get("allow_missing_gripper", True)):
            cmd.append("--allow-missing-gripper")
        else:
            cmd.append("--require-gripper")
        cwd = arm_root if arm_root.is_dir() else tta_root
        return self._run_external_script(
            "arm_reconnect(%s)" % after,
            cmd,
            cwd,
            float(cfg.get("timeout_sec", 60.0)),
            success_log="舵机重连完成，准备前往装货区",
        )

    def run_place_apriltag(self):
        """装货区：调用 arm/place_apriltag.py（AprilTag 视觉放置）。"""
        cfg = self.place_apriltag_cfg
        enabled = bool(cfg.get("enable", True)) and not self.skip_place
        if not enabled:
            rospy.logwarn(
                "[Mission] place_apriltag 已跳过（enable=false 或 --skip-place）"
            )
            return True

        tta_root_cfg = cfg.get("tta_root") or self.mobile_grasp_cfg.get(
            "tta_root"
        )
        try:
            tta_root = resolve_tta_root(tta_root_cfg)
        except FileNotFoundError as exc:
            rospy.logerr("[Mission] %s", exc)
            return False

        arm_root = tta_root / "arm"
        script = resolve_under_root(
            tta_root, cfg.get("script"), "arm/place_apriltag.py"
        )
        og_config = resolve_under_root(
            tta_root,
            cfg.get("og_config"),
            "arm/reset_and_grasp.yaml",
        )
        vision_config = resolve_under_root(
            tta_root,
            cfg.get("vision_config") or cfg.get("config"),
            "arm/place_apriltag.yaml",
        )
        if not script.is_file():
            rospy.logerr("[Mission] 放置脚本不存在: %s", script)
            return False
        if not og_config.is_file():
            rospy.logerr("[Mission] 放置 og 配置不存在: %s", og_config)
            return False
        if not vision_config.is_file():
            rospy.logerr("[Mission] 放置 vision 配置不存在: %s", vision_config)
            return False

        python_cfg = cfg.get("python")
        if python_cfg in (None, ""):
            python_cfg = self.mobile_grasp_cfg.get("python", "auto")
        try:
            python_cmd = resolve_grasp_python_cmd(python_cfg)
        except FileNotFoundError as exc:
            rospy.logerr("[Mission] %s", exc)
            return False

        cmd = python_cmd + [
            "-u",
            str(script),
            "--yes",
            "--og-config",
            str(og_config),
            "--vision-config",
            str(vision_config),
        ]
        # 车载无显示器时默认关预览，避免 OpenCV 窗口卡住
        if bool(cfg.get("no_live_preview", True)):
            cmd.append("--no-live-preview")
        extra = cfg.get("extra_args") or []
        if isinstance(extra, (list, tuple)):
            cmd.extend(str(x) for x in extra)

        cwd = arm_root if arm_root.is_dir() else tta_root
        timeout_sec = float(cfg.get("timeout_sec", 300.0))
        # 未识别到 AprilTag（place 脚本 exit=2）时：小车略微前移再重试
        no_tag_retries = max(0, int(cfg.get("no_tag_retries", 2)))
        no_tag_forward_m = float(cfg.get("no_tag_forward_m", 0.05))
        no_tag_speed = cfg.get("no_tag_forward_speed")
        if no_tag_speed in (None, ""):
            no_tag_speed = self.nav.speed
        else:
            no_tag_speed = float(no_tag_speed)

        attempts = no_tag_retries + 1
        for attempt in range(1, attempts + 1):
            cmd_attempt = list(cmd)
            # 非最后一次：保持持块 exit=2，便于前移重试
            # 最后一次：未识别则开爪+回初始位 exit=0，导航继续正常运动
            if attempt < attempts:
                cmd_attempt.append("--keep-hold-on-no-tag")
            code = self._run_external_script(
                "place_apriltag",
                cmd_attempt,
                cwd,
                timeout_sec,
                success_log="place_apriltag 成功，装货完成",
                return_code=True,
            )
            if code == 0:
                if attempt > 1:
                    rospy.loginfo(
                        "[Mission] 放置结束（含未识别 Tag 时开爪复位），继续任务"
                    )
                return True
            # place_apriltag.py：--keep-hold-on-no-tag 时未检测到 Tag → exit 2
            if code == 2 and attempt < attempts and no_tag_forward_m > 1e-4:
                rospy.logwarn(
                    "[Mission] 未检测到 AprilTag (attempt %d/%d)，"
                    "小车前移 %.3fm 后重试放置",
                    attempt,
                    attempts,
                    no_tag_forward_m,
                )
                self.nav.go_linear_x_distance(
                    no_tag_forward_m, speed=no_tag_speed
                )
                self.stop()
                rospy.sleep(0.3)
                continue
            if code == 2:
                rospy.logerr(
                    "[Mission] 多次前移后仍未检测到 AprilTag（共 %d 次）",
                    attempts,
                )
            return False
        return False

    def run_mission(self):
        timeouts = self.config.get("timeouts", {})
        mission = self.mission_cfg

        self._obstacle_mode = self._decide_obstacle_mode()
        if self._obstacle_mode == "front":
            rospy.logwarn(
                "[Mission] 前方障碍：取货「左→走→右」；装货「左→走→左」"
            )
        elif self._obstacle_mode == "back":
            rospy.logwarn(
                "[Mission] 后方障碍：取货直行；装货区「左移→正常→再左移」"
            )

        if not self.goto_zone("pickup"):
            return self._abort("无法到达取货区")
        pickup_hold = float(mission.get("pickup_hold_sec", 1.0))
        if pickup_hold > 0:
            rospy.loginfo("[Mission] 取货区等待 %.1fs", pickup_hold)
            rospy.sleep(pickup_hold)

        if not self.run_mobile_grasp():
            abort_on_fail = bool(
                self.mobile_grasp_cfg.get("abort_on_fail", True)
            )
            if abort_on_fail:
                return self._abort("取货失败")
            rospy.logwarn("[Mission] 取货失败但 abort_on_fail=false，继续任务")
        elif not self.skip_grasp:
            # 抓取子进程会断串口；重连失败默认不中止（夹爪掉线时可降级 1–5）
            if not self.run_arm_reconnect(after="grasp"):
                abort_reconnect = bool(
                    (self.config.get("arm_reconnect") or {}).get(
                        "abort_on_fail", False
                    )
                )
                if abort_reconnect:
                    return self._abort("抓取后舵机重连失败")
                rospy.logwarn(
                    "[Mission] 抓取后舵机重连失败，仍继续前往装货区"
                )

        if not self.goto_zone("loading"):
            return self._abort("无法到达装货区")

        loading_settle = float(mission.get("loading_settle_sec", 0.0))
        if loading_settle > 0:
            rospy.loginfo("[Mission] 装货区稳定等待 %.1fs", loading_settle)
            rospy.sleep(loading_settle)

        if not self.run_place_apriltag():
            abort_on_fail = bool(
                self.place_apriltag_cfg.get("abort_on_fail", True)
            )
            if abort_on_fail:
                return self._abort("装货放置失败")
            rospy.logwarn(
                "[Mission] 放置失败但 abort_on_fail=false，继续任务"
            )

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

        if mission.get("pre_rescue_backoff_enable", False):
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
            else:
                rospy.logwarn(
                    "[Mission] pre_rescue_backoff_enable 但缺少 segment_motion.pre_rescue_backoff"
                )
        else:
            rospy.loginfo("[Mission] 跳过前往救援区前的后退")

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
        description="nav_open_2026：转速×时间纯开环救援导航"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="参数配置 YAML（默认同目录 nav_open_2026.yaml）",
    )
    parser.add_argument("--no-autostart", action="store_true")
    parser.add_argument("--skip-drone", action="store_true")
    parser.add_argument(
        "--skip-grasp",
        action="store_true",
        help="跳过取货区 mobile_grasp（仅联调导航）",
    )
    parser.add_argument(
        "--skip-place",
        action="store_true",
        help="跳过装货区 place_apriltag（仅联调导航）",
    )
    parser.add_argument(
        "--rescue", type=int, default=2, choices=(1, 2, 3, 4)
    )
    parser.add_argument("--cmd-vel-topic", type=str, default=None)
    args = parser.parse_args()
    NavTimed(
        config_path=args.config,
        skip_drone=args.skip_drone,
        skip_grasp=args.skip_grasp,
        skip_place=args.skip_place,
        rescue_zone=args.rescue,
        cmd_vel_topic=args.cmd_vel_topic,
        autostart=not args.no_autostart,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
