#!/usr/bin/python3
# coding=UTF-8
"""静态占用栅格（/map）查询：世界坐标 ↔ 栅格、射线畅通、扇区净空。"""

from __future__ import print_function

import math


class OccupancyMap(object):
    """缓存 OccupancyGrid，提供地图系避障查询。"""

    def __init__(self, occupied_thresh=50, unknown_as_occupied=False):
        self.occupied_thresh = int(occupied_thresh)
        self.unknown_as_occupied = bool(unknown_as_occupied)
        self.info = None
        self.data = None
        self.width = 0
        self.height = 0
        self.resolution = 0.05
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_yaw = 0.0

    @property
    def ready(self):
        return self.data is not None and self.width > 0 and self.height > 0

    def update(self, msg):
        """接收 nav_msgs/OccupancyGrid。"""
        info = msg.info
        self.info = info
        self.width = int(info.width)
        self.height = int(info.height)
        self.resolution = float(info.resolution)
        self.origin_x = float(info.origin.position.x)
        self.origin_y = float(info.origin.position.y)
        q = info.origin.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.origin_yaw = math.atan2(siny, cosy)
        self.data = list(msg.data)

    def world_to_index(self, x, y):
        """地图系 (x,y) → (col, row)；越界返回 None。"""
        if not self.ready:
            return None
        dx = x - self.origin_x
        dy = y - self.origin_y
        c = math.cos(-self.origin_yaw)
        s = math.sin(-self.origin_yaw)
        lx = c * dx - s * dy
        ly = s * dx + c * dy
        col = int(math.floor(lx / self.resolution))
        row = int(math.floor(ly / self.resolution))
        if col < 0 or row < 0 or col >= self.width or row >= self.height:
            return None
        return col, row

    def cell_value(self, col, row):
        if not self.ready:
            return -1
        if col < 0 or row < 0 or col >= self.width or row >= self.height:
            return -1
        return int(self.data[row * self.width + col])

    def is_occupied_world(self, x, y, inflate_m=0.0):
        """点是否占用；inflate_m>0 时在周围做圆形膨胀检查。"""
        if not self.ready:
            return self.unknown_as_occupied
        if inflate_m <= 1e-6:
            idx = self.world_to_index(x, y)
            if idx is None:
                return self.unknown_as_occupied
            return self._occupied_value(self.cell_value(idx[0], idx[1]))

        step = max(self.resolution * 0.5, 0.02)
        r = float(inflate_m)
        n = max(1, int(math.ceil(r / step)))
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                if (i * step) ** 2 + (j * step) ** 2 > r * r + 1e-9:
                    continue
                if self.is_occupied_world(x + i * step, y + j * step, inflate_m=0.0):
                    return True
        return False

    def _occupied_value(self, value):
        if value < 0:
            return self.unknown_as_occupied
        return value >= self.occupied_thresh

    def ray_clearance(self, x, y, yaw, max_range=3.0, inflate_m=0.05):
        """从 (x,y) 沿 yaw 射线，返回碰到占用前的净空 (m)。"""
        if not self.ready:
            return 0.0 if self.unknown_as_occupied else float(max_range)
        step = max(self.resolution * 0.5, 0.02)
        max_range = float(max_range)
        dist = 0.0
        c = math.cos(yaw)
        s = math.sin(yaw)
        while dist <= max_range:
            px = x + c * dist
            py = y + s * dist
            if self.is_occupied_world(px, py, inflate_m=inflate_m):
                return max(0.0, dist - step)
            dist += step
        return max_range

    def sector_clearances(self, x, y, yaw, max_range=3.0, inflate_m=0.05):
        """车体前/后/左/右四个地图净空（相对当前航向）。"""
        return {
            "front": self.ray_clearance(x, y, yaw, max_range, inflate_m),
            "back": self.ray_clearance(x, y, yaw + math.pi, max_range, inflate_m),
            "left": self.ray_clearance(
                x, y, yaw + math.pi * 0.5, max_range, inflate_m
            ),
            "right": self.ray_clearance(
                x, y, yaw - math.pi * 0.5, max_range, inflate_m
            ),
        }

    def segment_blocked(self, x0, y0, x1, y1, inflate_m=0.08, step=None):
        """两点间线段是否穿过占用栅格。"""
        if not self.ready:
            return self.unknown_as_occupied
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return self.is_occupied_world(x0, y0, inflate_m=inflate_m)
        step = step if step is not None else max(self.resolution * 0.5, 0.02)
        n = max(1, int(math.ceil(length / step)))
        for i in range(n + 1):
            t = float(i) / n
            if self.is_occupied_world(
                x0 + dx * t, y0 + dy * t, inflate_m=inflate_m
            ):
                return True
        return False
