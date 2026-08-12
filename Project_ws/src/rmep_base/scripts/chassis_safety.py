#!/usr/bin/python3
# coding=UTF-8
"""底盘速度指令看门狗：使用单调时钟判断末次非零指令是否超时。"""

import time
import math


def limit_chassis_command(x, y, angular_z, max_linear, max_angular):
    """按平移合速度和角速度上限裁剪指令，拒绝 NaN 和无穷值。"""
    values = tuple(float(value) for value in (x, y, angular_z))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("底盘速度必须是有限数值")

    x, y, angular_z = values
    max_linear = max(0.0, float(max_linear))
    max_angular = max(0.0, float(max_angular))
    linear_speed = math.hypot(x, y)
    if linear_speed > max_linear and linear_speed > 0.0:
        scale = max_linear / linear_speed
        x *= scale
        y *= scale
    angular_z = max(-max_angular, min(max_angular, angular_z))
    return x, y, angular_z


class ChassisCommandWatchdog:
    """记录速度指令时间，并确保每次超时只请求一次停车。"""

    def __init__(self, timeout=0.30, clock=None):
        self.timeout = max(0.05, float(timeout))
        self.clock = clock or time.monotonic
        self.last_command_at = None
        self.stopped = True

    def note_motion_command(self, received_at=None):
        """记录一条已经成功下发的非零运动指令。"""
        self.last_command_at = (
            self.clock() if received_at is None else float(received_at)
        )
        self.stopped = False

    def mark_stopped(self):
        """记录底盘已经收到零轮速，避免定时器重复发送停车。"""
        self.last_command_at = None
        self.stopped = True

    def should_stop(self, now=None):
        """非零指令超过允许时间后返回 True。"""
        if self.stopped or self.last_command_at is None:
            return False
        now = self.clock() if now is None else float(now)
        return now - self.last_command_at > self.timeout
