#!/usr/bin/python3
# coding=UTF-8
"""OccupancyMap 离线自检（不需要 ROS）。"""

from __future__ import print_function

import math
import sys
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from map_occupancy import OccupancyMap  # noqa: E402


class _Info(object):
    def __init__(self):
        self.width = 20
        self.height = 20
        self.resolution = 0.05
        self.origin = type(
            "O",
            (),
            {
                "position": type("P", (), {"x": 0.0, "y": 0.0})(),
                "orientation": type(
                    "Q", (), {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                )(),
            },
        )()


class _Msg(object):
    def __init__(self, data):
        self.info = _Info()
        self.data = data


def main():
    w, h = 20, 20
    data = [0] * (w * h)
    # 竖墙 x≈0.5m → col=10
    for row in range(h):
        data[row * w + 10] = 100

    occ = OccupancyMap(occupied_thresh=50)
    occ.update(_Msg(data))
    assert occ.ready
    assert occ.is_occupied_world(0.52, 0.25)
    assert not occ.is_occupied_world(0.10, 0.25)

    # 从原点向 +x 射线，应在墙前停下
    clr = occ.ray_clearance(0.05, 0.25, 0.0, max_range=2.0, inflate_m=0.0)
    assert 0.30 < clr < 0.55, "unexpected clearance %.3f" % clr

    assert occ.segment_blocked(0.05, 0.25, 0.80, 0.25)
    assert not occ.segment_blocked(0.05, 0.25, 0.30, 0.25)

    sectors = occ.sector_clearances(0.05, 0.25, 0.0, max_range=2.0)
    assert sectors["front"] < sectors["left"]
    print("map_occupancy self-check OK  front=%.3f left=%.3f" % (
        sectors["front"],
        sectors["left"],
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
