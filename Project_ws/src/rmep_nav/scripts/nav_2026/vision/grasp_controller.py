#!/usr/bin/python3
# coding=UTF-8

import time
from typing import Callable, Optional

import rospy
from geometry_msgs.msg import Twist

from vision.pickup_detector import PickupTarget

try:
    from rmep_base.srv import RobotArm, RobotGrip
except ImportError:
    RobotArm = None
    RobotGrip = None


class GraspController:
    """底盘微调 + 机械臂/夹爪抓取。"""

    def __init__(self, config=None, cmd_vel_pub=None):
        grasp_cfg = (config or {}).get("vision", {}).get("grasp", {})
        align_cfg = grasp_cfg.get("align", {})

        self.enabled = grasp_cfg.get("enabled", True)
        self.require_hardware = grasp_cfg.get("require_hardware", False)
        self.cell_positions = grasp_cfg.get("arm", {}).get("cell_positions", {})
        self.default_arm = grasp_cfg.get("arm", {}).get("default", [0.10, 0.10])
        self.lift_pos = grasp_cfg.get("arm", {}).get("lift_pos", [0.05, 0.15])
        self.open_time = grasp_cfg.get("gripper", {}).get("open_time", 0.5)
        self.close_time = grasp_cfg.get("gripper", {}).get("close_time", 0.8)

        self.align_enabled = align_cfg.get("enabled", True)
        self.pixel_deadband = align_cfg.get("pixel_deadband", 35)
        self.linear_speed = align_cfg.get("linear_speed", 0.05)
        self.angular_speed = align_cfg.get("angular_speed", 0.25)
        self.max_align_iters = align_cfg.get("max_iters", 8)
        self.image_center_offset = align_cfg.get("center_offset", [0, 0])

        self._cmd_vel_pub = cmd_vel_pub
        self._arm = None
        self._gripper = None
        self._init_services()

    def _init_services(self):
        if RobotArm is None or RobotGrip is None:
            rospy.logwarn("rmep_base services unavailable, arm/grasp disabled")
            return
        try:
            rospy.wait_for_service("ep_arm", timeout=2.0)
            rospy.wait_for_service("ep_gripper", timeout=2.0)
            self._arm = rospy.ServiceProxy("ep_arm", RobotArm)
            self._gripper = rospy.ServiceProxy("ep_gripper", RobotGrip)
            rospy.loginfo("grasp controller connected to ep_arm / ep_gripper")
        except rospy.ROSException:
            rospy.logwarn("ep_arm / ep_gripper not ready")

    @property
    def hardware_ready(self):
        return self._arm is not None and self._gripper is not None

    def align_to_target(
        self,
        target: PickupTarget,
        frame_shape,
        get_frame: Callable,
        detect_fn: Callable,
        twist_guard_fn: Optional[Callable] = None,
    ) -> Optional[PickupTarget]:
        if not self.align_enabled or self._cmd_vel_pub is None:
            return target

        h, w = frame_shape[:2]
        cx = w / 2.0 + self.image_center_offset[0]
        cy = h / 2.0 + self.image_center_offset[1]
        current = target

        for _ in range(self.max_align_iters):
            err_u = current.pixel_u - cx
            err_v = current.pixel_v - cy
            if abs(err_u) < self.pixel_deadband and abs(err_v) < self.pixel_deadband:
                return current

            twist = Twist()
            twist.linear.x = max(-self.linear_speed, min(self.linear_speed, -err_v * 0.0008))
            twist.linear.y = max(-self.linear_speed, min(self.linear_speed, -err_u * 0.0008))
            twist.angular.z = max(
                -self.angular_speed, min(self.angular_speed, -err_u * 0.002)
            )
            if twist_guard_fn is not None:
                twist = twist_guard_fn(twist)

            duration = 0.35
            start = rospy.Time.now()
            while (rospy.Time.now() - start).to_sec() < duration:
                cmd = twist
                if twist_guard_fn is not None:
                    cmd = twist_guard_fn(twist)
                self._cmd_vel_pub.publish(cmd)
                rospy.sleep(0.05)
            self._stop()

            frame = get_frame()
            if frame is None:
                break
            refreshed = detect_fn(frame)
            if refreshed is not None:
                current = refreshed
        return current

    def _stop(self):
        if self._cmd_vel_pub is not None:
            self._cmd_vel_pub.publish(Twist())

    def _arm_pos_for_cell(self, cell_index: int):
        key = str(cell_index)
        if key in self.cell_positions:
            pos = self.cell_positions[key]
            return float(pos[0]), float(pos[1])
        if cell_index in self.cell_positions:
            pos = self.cell_positions[cell_index]
            return float(pos[0]), float(pos[1])
        return float(self.default_arm[0]), float(self.default_arm[1])

    def _gripper_open(self):
        if self._gripper is None:
            return False
        self._gripper(state=1, value=self.open_time)
        return True

    def _gripper_close(self):
        if self._gripper is None:
            return False
        self._gripper(state=0, value=self.close_time)
        return True

    def _arm_move(self, x, y):
        if self._arm is None:
            return False
        self._arm(x=x, y=y)
        return True

    def execute(self, target: PickupTarget) -> bool:
        if not self.enabled:
            return True
        if not self.hardware_ready:
            if self.require_hardware:
                rospy.logerr("grasp hardware required but services missing")
                return False
            rospy.logwarn("grasp hardware missing, detect-only mode")
            return True

        ax, ay = self._arm_pos_for_cell(target.cell_index)
        lx, ly = self.lift_pos

        rospy.loginfo(
            "grasp cell=%d arm=(%.3f, %.3f)", target.cell_index, ax, ay
        )
        self._gripper_open()
        rospy.sleep(0.2)
        if not self._arm_move(lx, ly):
            return False
        rospy.sleep(0.2)
        if not self._arm_move(ax, ay):
            return False
        rospy.sleep(0.3)
        if not self._gripper_close():
            return False
        rospy.sleep(0.2)
        if not self._arm_move(lx, ly):
            return False
        return True
