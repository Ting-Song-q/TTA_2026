#!/usr/bin/python3
# coding=UTF-8
"""把机械臂相机视野方向映射为 base_link 的前、后、左、右风险。"""

import math

_SECTOR_CENTERS = {
    "front": 0.0,
    "left": 90.0,
    "back": 180.0,
    "right": -90.0,
}


def normalize_angle_deg(angle):
    """把角度归一化到 [-180, 180)。"""
    angle = float(angle)
    if not math.isfinite(angle):
        raise ValueError("相机偏航角必须是有限数值")
    return (angle + 180.0) % 360.0 - 180.0


def angular_distance_deg(first, second):
    """返回两个角度之间的最小绝对夹角。"""
    return abs(normalize_angle_deg(float(first) - float(second)))


def body_sector_for_yaw(yaw_deg):
    """返回与给定 base_link 偏航角最接近的车体方向。"""
    return min(
        _SECTOR_CENTERS,
        key=lambda name: angular_distance_deg(yaw_deg, _SECTOR_CENTERS[name]),
    )


def map_camera_risks_to_body(
    camera_yaw_deg, front_risk, left_risk, right_risk, horizontal_fov_deg
):
    """将相机画面三分区风险转换为车体四方向的稀疏风险。"""
    mapped = {"front": -1.0, "back": -1.0, "left": -1.0, "right": -1.0}
    # 左右图像区域中心约位于水平视场中心两侧四分之一视场处。
    offset = max(1.0, float(horizontal_fov_deg) * 0.25)
    observations = (
        (camera_yaw_deg, front_risk),
        (camera_yaw_deg + offset, left_risk),
        (camera_yaw_deg - offset, right_risk),
    )
    for yaw, risk in observations:
        sector = body_sector_for_yaw(yaw)
        mapped[sector] = max(mapped[sector], max(0.0, min(1.0, float(risk))))
    return mapped


class CameraScanPlanner:
    """按目标角度循环扫描，并在机械臂到位后保持指定观察时间。"""

    def __init__(self, angles=None, tolerance_deg=5.0, dwell_time=0.35):
        self.angles = [float(value) for value in (angles or [0, 90, 180, -90])]
        self.tolerance_deg = float(tolerance_deg)
        self.dwell_time = float(dwell_time)
        self.index = 0
        self.reached_at = None

    @property
    def target(self):
        """返回当前机械臂相机目标偏航角。"""
        return self.angles[self.index]

    def update(self, current_yaw_deg, now):
        """根据反馈角度推进扫描目标，返回当前应发布的目标角。"""
        if angular_distance_deg(current_yaw_deg, self.target) > self.tolerance_deg:
            self.reached_at = None
            return self.target
        if self.reached_at is None:
            self.reached_at = float(now)
            return self.target
        if float(now) - self.reached_at >= self.dwell_time:
            self.index = (self.index + 1) % len(self.angles)
            self.reached_at = None
        return self.target

    def observation_ready(self, current_yaw_deg, now, settle_time=0.0):
        """机械臂到达目标并稳定指定时间后，才允许当前图像进入方向缓存。"""
        if angular_distance_deg(current_yaw_deg, self.target) > self.tolerance_deg:
            return False
        if self.reached_at is None:
            return False
        settle_time = max(0.0, float(settle_time))
        return float(now) - self.reached_at >= settle_time
