#!/usr/bin/python3
# coding=UTF-8
"""底盘速度看门狗离线测试，不连接 ROS 和真实机器人。"""

import sys
import unittest
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from chassis_safety import ChassisCommandWatchdog, limit_chassis_command


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class ChassisCommandWatchdogTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.watchdog = ChassisCommandWatchdog(0.30, clock=self.clock)

    def test_startup_state_does_not_repeat_stop(self):
        """启动状态已经视为停车，定时器不应持续重复发送零轮速。"""
        self.clock.now = 1.0
        self.assertFalse(self.watchdog.should_stop())

    def test_motion_command_times_out(self):
        """非零运动指令超过 0.30 秒后必须请求停车。"""
        self.watchdog.note_motion_command()
        self.clock.now = 0.30
        self.assertFalse(self.watchdog.should_stop())
        self.clock.now = 0.31
        self.assertTrue(self.watchdog.should_stop())

    def test_mark_stopped_prevents_repeated_timeout(self):
        """已经发送停车后，同一次超时不能反复触发。"""
        self.watchdog.note_motion_command()
        self.clock.now = 0.31
        self.assertTrue(self.watchdog.should_stop())
        self.watchdog.mark_stopped()
        self.assertFalse(self.watchdog.should_stop())

    def test_new_command_rearms_watchdog(self):
        """停车后的新运动指令必须重新启动超时计时。"""
        self.watchdog.mark_stopped()
        self.clock.now = 1.0
        self.watchdog.note_motion_command()
        self.clock.now = 1.29
        self.assertFalse(self.watchdog.should_stop())
        self.clock.now = 1.31
        self.assertTrue(self.watchdog.should_stop())


class ChassisCommandLimitTest(unittest.TestCase):
    def test_linear_vector_is_limited_without_changing_direction(self):
        """斜向速度按合速度限幅，不能分别限幅后超过总上限。"""
        x, y, angular = limit_chassis_command(0.3, 0.4, 0.0, 0.2, 0.5)
        self.assertAlmostEqual((x * x + y * y) ** 0.5, 0.2)
        self.assertAlmostEqual(x / y, 0.3 / 0.4)
        self.assertEqual(angular, 0.0)

    def test_angular_speed_is_limited(self):
        """过大的角速度必须限制在底盘最终安全上限内。"""
        _, _, angular = limit_chassis_command(0.0, 0.0, 1.2, 0.2, 0.5)
        self.assertEqual(angular, 0.5)

    def test_non_finite_command_is_rejected(self):
        """NaN 速度不得进入轮速换算。"""
        with self.assertRaises(ValueError):
            limit_chassis_command(float("nan"), 0.0, 0.0, 0.2, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
