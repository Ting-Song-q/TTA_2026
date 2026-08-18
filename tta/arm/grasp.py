#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视觉定位前方红色方块并抓取（参考 grasp_test/grasp_vision/test1_grasp.py）。

流程:
  1. 机械臂到观察位，夹爪张开
  2. HSV+RGB 检测红色方块中心
  3. 视觉伺服：shoulder_pan 左右 / reach 前后 对准
  4. 高位接近 → 下降抓取 → 闭合 → 抬起（夹爪保持闭合）

与原版差异:
  - 相机: OpenCV V4L2（默认 /dev/video11），不再依赖 ROS /usb_cam
  - 机械臂: LeRobot FeetechMotorsBus（与 start.py 一致），关节空间位姿
  - 底盘补距: 默认关闭（可用 --chassis 占位扩展）

依赖:
  pip install opencv-python numpy feetech-servo-sdk
  # 以及已安装的 lerobot

用法:
  python3 grasp.py --port /dev/ttyACM0 --camera 11 --dry-run
  python3 grasp.py --port /dev/ttyACM0 --camera 11
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 复用同目录 start.py 的总线工具
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from start import (
        DEFAULT_BAUD,
        PROTOCOL_VERSION,
        auto_pick_port,
        connect_bus,
        move_to,
        read_positions,
    )
except ImportError as exc:
    raise SystemExit(f"无法导入 start.py 总线工具: {exc}") from exc


# ---------------------------------------------------------------------------
# 可调参数（现场主要改这里）
# ---------------------------------------------------------------------------
CFG = {
    "camera_index": 11,  # /dev/video11
    "camera_width": 640,
    "camera_height": 480,
    "settle_frames": 8,
    "frame_timeout": 3.0,
    "save_debug": True,
    "debug_dir": str(Path.home() / "Desktop" / "grasp_test" / "debug" / "tta_arm_grasp"),
    # 红色检测（沿用 test1_grasp）
    "roi": [0.08, 0.15, 0.92, 0.92],
    "hsv_lower1": [0, 75, 45],
    "hsv_upper1": [18, 255, 255],
    "hsv_lower2": [165, 75, 45],
    "hsv_upper2": [180, 255, 255],
    "min_r": 95,
    "min_r_minus_g": 28,
    "min_r_minus_b": 22,
    "max_g": 205,
    "min_area": 500,
    "min_rect_fill": 0.40,
    # 视觉伺服图像目标点（640x480）
    "x_center": 320.0,
    "y_center": 395.0,
    "x_pid_p": 0.35,  # 像素误差 → shoulder_pan ticks
    "y_pid_p": 0.25,  # 像素误差 → reach ticks（前后）
    "pan_init": 2048,
    "reach_init": 0,
    "pan_limits": [1600, 2500],
    "reach_limits": [-250, 250],
    "stable_err_u": 22.0,
    "stable_err_v": 28.0,
    "stable_frames": 3,
    "track_timeout": 14.0,
    # 关节位姿（原始 ticks，按你的臂现场标定后改）
    # 未给出的关节保持观察位数值
    "observe_pose": {
        "shoulder_pan": 2048,
        "shoulder_lift": 1800,
        "elbow_flex": 2400,
        "wrist_flex": 2048,
        "wrist_roll": 2048,
        "gripper": 1200,  # 张开（按你的夹爪方向可能相反，需微调）
    },
    "track_base_pose": {
        "shoulder_lift": 1850,
        "elbow_flex": 2350,
        "wrist_flex": 2048,
        "wrist_roll": 2048,
    },
    # reach>0 时 elbow 前伸、lift 略降（简易前后映射）
    "reach_elbow_scale": 0.8,
    "reach_lift_scale": -0.35,
    # 抓取序列
    "approach_pose": {
        "shoulder_lift": 1750,
        "elbow_flex": 2500,
        "wrist_flex": 2100,
    },
    "grasp_pose": {
        "shoulder_lift": 1650,
        "elbow_flex": 2600,
        "wrist_flex": 2150,
    },
    "lift_pose": {
        "shoulder_lift": 1900,
        "elbow_flex": 2300,
        "wrist_flex": 2048,
    },
    "open_gripper": 1200,
    "close_gripper": 2100,
    "move_settle": 1.0,
    "grasp_settle": 1.2,
}


@dataclass
class RedDetection:
    center_u: float
    center_v: float
    area: float
    bbox: Tuple[int, int, int, int]
    contour: np.ndarray
    mask: np.ndarray
    roi: Tuple[int, int, int, int]
    rect_fill: float


class SimplePID:
    def __init__(self, p: float):
        self.p = float(p)
        self.setpoint = 0.0
        self.output = 0.0

    def clear(self) -> None:
        self.output = 0.0

    def update(self, measurement: float) -> float:
        self.output = self.p * (self.setpoint - measurement)
        return self.output


class RedDetector:
    def __init__(self, cfg: dict):
        self.roi_norm = cfg["roi"]
        self.lower1 = np.array(cfg["hsv_lower1"], dtype=np.uint8)
        self.upper1 = np.array(cfg["hsv_upper1"], dtype=np.uint8)
        self.lower2 = np.array(cfg["hsv_lower2"], dtype=np.uint8)
        self.upper2 = np.array(cfg["hsv_upper2"], dtype=np.uint8)
        self.min_r = int(cfg["min_r"])
        self.min_r_minus_g = int(cfg["min_r_minus_g"])
        self.min_r_minus_b = int(cfg["min_r_minus_b"])
        self.max_g = int(cfg["max_g"])
        self.min_area = float(cfg["min_area"])
        self.min_rect_fill = float(cfg["min_rect_fill"])

    def _roi(self, frame):
        h, w = frame.shape[:2]
        x1n, y1n, x2n, y2n = self.roi_norm
        x1 = max(0, min(w - 1, int(w * x1n)))
        y1 = max(0, min(h - 1, int(h * y1n)))
        x2 = max(x1 + 1, min(w, int(w * x2n)))
        y2 = max(y1 + 1, min(h, int(h * y2n)))
        return x1, y1, x2, y2

    def detect(self, frame) -> Optional[RedDetection]:
        if frame is None:
            return None
        x1, y1, x2, y2 = self._roi(frame)
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower1, self.upper1) | cv2.inRange(
            hsv, self.lower2, self.upper2
        )

        b, g, r = cv2.split(crop)
        r = r.astype(np.int16)
        g = g.astype(np.int16)
        b = b.astype(np.int16)
        rgb_mask = (
            (r >= self.min_r)
            & ((r - g) >= self.min_r_minus_g)
            & ((r - b) >= self.min_r_minus_b)
            & (g <= self.max_g)
        ).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, rgb_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            fill = area / float(max(w * h, 1))
            if fill < self.min_rect_fill:
                continue
            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue
            cu = m["m10"] / m["m00"] + x1
            cv_ = m["m01"] / m["m00"] + y1
            score = area * fill
            if score > best_score:
                best_score = score
                best = RedDetection(
                    cu,
                    cv_,
                    area,
                    (x + x1, y + y1, x + x1 + w, y + y1 + h),
                    contour.copy(),
                    mask.copy(),
                    (x1, y1, x2, y2),
                    fill,
                )
        return best

    def draw(self, frame, det: Optional[RedDetection], desired=None):
        vis = frame.copy()
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = det.roi if det is not None else self._roi(frame)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if desired is not None:
            cv2.drawMarker(
                vis,
                (int(desired[0]), int(desired[1])),
                (255, 255, 0),
                cv2.MARKER_CROSS,
                18,
                2,
            )
        if det is None:
            cv2.putText(
                vis,
                "no red block",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )
            return vis
        bx1, by1, bx2, by2 = det.bbox
        contour = det.contour + np.array([[[x1, y1]]], dtype=np.int32)
        cv2.drawContours(vis, [contour], -1, (0, 255, 255), 2)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        cv2.circle(vis, (int(det.center_u), int(det.center_v)), 5, (255, 0, 0), -1)
        if desired is not None:
            cv2.putText(
                vis,
                "err_u={:+.0f} err_v={:+.0f}".format(
                    det.center_u - desired[0], det.center_v - desired[1]
                ),
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (50, 220, 50),
                2,
            )
        return vis


class Camera:
    """OpenCV V4L2 相机。"""

    def __init__(self, index: int, width: int, height: int, settle: int):
        self.index = index
        self.width = width
        self.height = height
        self.settle = settle
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # 丢弃启动时的旧帧
        for _ in range(max(1, self.settle)):
            self.cap.read()
        return True

    def get_frame(self, discard: int = 1) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            return None
        frame = None
        for _ in range(max(1, discard)):
            ok, frame = self.cap.read()
            if not ok:
                return None
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class ArmController:
    """LeRobot Feetech 关节控制（原始 ticks）。"""

    def __init__(self, bus):
        self.bus = bus
        self.bus.enable_torque()

    def write_pose(self, pose: Dict[str, float], settle: float = 1.0) -> None:
        goals = {k: int(v) for k, v in pose.items()}
        move_to(self.bus, goals, wait_s=settle)

    def set_gripper(self, value: int, settle: float = 0.5) -> None:
        move_to(self.bus, {"gripper": int(value)}, wait_s=settle)

    def read(self) -> Dict[str, float]:
        return read_positions(self.bus)


class FrontRedGrasp:
    """前方红色方块：视觉伺服 + 防碰抓取（关节空间版）。"""

    def __init__(self, cfg: dict, arm: Optional[ArmController], camera: Camera):
        self.cfg = cfg
        self.arm = arm
        self.camera = camera
        self.detector = RedDetector(cfg)
        self.x_pid = SimplePID(cfg["x_pid_p"])
        self.y_pid = SimplePID(cfg["y_pid_p"])
        self.debug_dir = Path(cfg["debug_dir"])
        self._dbg_i = 0
        self.pan = float(cfg["pan_init"])
        self.reach = float(cfg["reach_init"])

    def _log(self, msg: str, *args) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        text = msg % args if args else msg
        print(f"[grasp {stamp}] {text}")

    def _save_debug(self, frame, tag: str, det: Optional[RedDetection] = None) -> None:
        if not self.cfg["save_debug"] or frame is None:
            return
        desired = (self.cfg["x_center"], self.cfg["y_center"])
        vis = self.detector.draw(frame, det, desired)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        path = self.debug_dir / f"{self._dbg_i:03d}_{tag}.jpg"
        self._dbg_i += 1
        cv2.imwrite(str(path), vis)
        self._log("debug -> %s", path)

    def _track_pose(self) -> Dict[str, int]:
        """由 pan/reach 生成跟踪关节位姿。"""
        cfg = self.cfg
        base = dict(cfg["track_base_pose"])
        pose = {
            "shoulder_pan": int(self.pan),
            "shoulder_lift": int(
                base["shoulder_lift"] + self.reach * float(cfg["reach_lift_scale"])
            ),
            "elbow_flex": int(
                base["elbow_flex"] + self.reach * float(cfg["reach_elbow_scale"])
            ),
            "wrist_flex": int(base["wrist_flex"]),
            "wrist_roll": int(base["wrist_roll"]),
            "gripper": int(cfg["open_gripper"]),
        }
        return pose

    def go_observe(self) -> None:
        if self.arm is None:
            return
        self._log("observe pose, open gripper")
        pose = dict(self.cfg["observe_pose"])
        pose["gripper"] = int(self.cfg["open_gripper"])
        self.arm.write_pose(pose, settle=float(self.cfg["move_settle"]) + 0.5)
        self.pan = float(pose.get("shoulder_pan", self.cfg["pan_init"]))
        self.reach = float(self.cfg["reach_init"])
        self.x_pid.clear()
        self.y_pid.clear()

    def visual_servo_align(self) -> Optional[RedDetection]:
        cfg = self.cfg
        x_center = float(cfg["x_center"])
        y_center = float(cfg["y_center"])
        pan_low, pan_high = cfg["pan_limits"]
        reach_low, reach_high = cfg["reach_limits"]
        stable_need = int(cfg["stable_frames"])
        stable_count = 0
        deadline = time.time() + float(cfg["track_timeout"])
        last_det = None
        first = True

        if self.arm is not None:
            self._log("open gripper before tracking")
            self.arm.set_gripper(int(cfg["open_gripper"]), settle=0.35)

        while time.time() < deadline:
            frame = self.camera.get_frame(discard=2)
            if frame is None:
                self._log("no camera frame")
                time.sleep(0.05)
                continue
            det = self.detector.detect(frame)
            self._save_debug(frame, "track", det)
            if det is None:
                stable_count = 0
                time.sleep(0.05)
                continue

            err_u = det.center_u - x_center
            err_v = det.center_v - y_center
            last_det = det

            # 与原版一致：小误差时锁定当前点，降低抖动
            self.x_pid.setpoint = det.center_u if abs(err_u) < 10 else x_center
            self.x_pid.update(det.center_u)
            # err_v<0 目标偏上(更远) → reach 增大前伸
            self.y_pid.setpoint = det.center_v if abs(err_v) < 10 else y_center
            self.y_pid.update(det.center_v)

            self.pan = max(float(pan_low), min(float(pan_high), self.pan + self.x_pid.output))
            # y_pid.output = P*(setpoint - v)；目标偏上时 v 小 → output>0 → reach 增大
            self.reach = max(
                float(reach_low),
                min(float(reach_high), self.reach + self.y_pid.output),
            )

            if self.arm is not None:
                settle = 0.25 if first else 0.08
                first = False
                self.arm.write_pose(self._track_pose(), settle=settle)

            self._log(
                "servo err_u=%+.1f err_v=%+.1f pan=%d reach=%+.0f",
                err_u,
                err_v,
                int(self.pan),
                self.reach,
            )

            if abs(err_u) < float(cfg["stable_err_u"]) and abs(err_v) < float(
                cfg["stable_err_v"]
            ):
                stable_count += 1
                if stable_count >= stable_need:
                    self._log("aligned stable_count=%d", stable_count)
                    break
            else:
                stable_count = 0

        if last_det is None:
            return None
        if stable_count < stable_need:
            self._log(
                "align timeout, use last pose pan=%d reach=%+.0f",
                int(self.pan),
                self.reach,
            )
        return last_det

    def collision_safe_grasp(self) -> bool:
        """高位接近 → 下降 → 闭合 → 抬起。"""
        if self.arm is None:
            return False
        cfg = self.cfg
        pan = int(self.pan)

        self._log("grasp step1: ensure gripper open")
        self.arm.set_gripper(int(cfg["open_gripper"]), settle=0.4)

        def _merge(extra: dict, gripper: int) -> Dict[str, int]:
            pose = self._track_pose()
            pose.update({k: int(v) for k, v in extra.items()})
            pose["shoulder_pan"] = pan
            pose["gripper"] = int(gripper)
            return pose

        self._log("grasp step2: approach hover")
        self.arm.write_pose(
            _merge(cfg["approach_pose"], cfg["open_gripper"]),
            settle=float(cfg["grasp_settle"]),
        )

        self._log("grasp step3: descend")
        self.arm.write_pose(
            _merge(cfg["grasp_pose"], cfg["open_gripper"]),
            settle=float(cfg["grasp_settle"]),
        )

        self._log("grasp step4: close gripper")
        self.arm.set_gripper(int(cfg["close_gripper"]), settle=0.85)

        self._log("grasp step5: lift")
        self.arm.write_pose(
            _merge(cfg["lift_pose"], cfg["close_gripper"]),
            settle=float(cfg["grasp_settle"]),
        )
        self._log("grasp finished")
        return True

    def reset_holding(self) -> None:
        """复位到观察位，夹爪保持闭合。"""
        if self.arm is None:
            return
        pose = dict(self.cfg["observe_pose"])
        pose["gripper"] = int(self.cfg["close_gripper"])
        self._log("reset holding, gripper closed=%d", pose["gripper"])
        self.arm.write_pose(pose, settle=float(self.cfg["move_settle"]) + 0.5)
        self.arm.set_gripper(int(self.cfg["close_gripper"]), settle=0.3)

    def run(self, dry_run: bool = False) -> bool:
        self._log("=== front red block grasp start ===")
        if not dry_run:
            self.go_observe()

        self._log("wait camera...")
        frame = self.camera.get_frame(discard=self.cfg["settle_frames"])
        if frame is None:
            self._log("camera not ready (index=%s)", self.cfg["camera_index"])
            return False
        time.sleep(0.3)

        det = self.detector.detect(frame)
        self._save_debug(frame, "detect0", det)
        if det is None:
            for i in range(5):
                frame = self.camera.get_frame(discard=2)
                det = self.detector.detect(frame) if frame is not None else None
                self._save_debug(frame, f"detect{i + 1}", det)
                if det is not None:
                    break
                time.sleep(0.15)
        if det is None:
            self._log("前方未检测到红色方块")
            return False
        self._log(
            "found red u=%.0f v=%.0f area=%.0f", det.center_u, det.center_v, det.area
        )

        if dry_run:
            self._log("dry-run: skip arm motion / grasp")
            return True

        aligned = self.visual_servo_align()
        if aligned is None:
            self._log("visual align failed")
            return False

        frame = self.camera.get_frame(discard=2)
        self._save_debug(frame, "aligned", aligned)

        ok = self.collision_safe_grasp()
        after = self.camera.get_frame(discard=2)
        self._save_debug(
            after,
            "after_grasp",
            self.detector.detect(after) if after is not None else None,
        )
        if ok:
            self.reset_holding()
        return ok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="视觉定位并抓取前方红色方块（LeRobot 臂）")
    p.add_argument("--port", default=None, help="舵机串口，如 /dev/ttyACM0")
    p.add_argument("--camera", type=int, default=None, help="摄像头编号，默认 11")
    p.add_argument("--dry-run", action="store_true", help="只检测并保存调试图，不运动")
    p.add_argument(
        "--list-ports", action="store_true", help="列出串口后退出（调用 start.py 逻辑）"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = copy.deepcopy(CFG)
    if args.camera is not None:
        cfg["camera_index"] = int(args.camera)

    if args.list_ports:
        from start import cmd_list_ports

        return cmd_list_ports()

    camera = Camera(
        index=int(cfg["camera_index"]),
        width=int(cfg["camera_width"]),
        height=int(cfg["camera_height"]),
        settle=int(cfg["settle_frames"]),
    )
    if not camera.open():
        print(f"无法打开摄像头 /dev/video{cfg['camera_index']}")
        return 1

    bus = None
    arm = None
    try:
        if not args.dry_run:
            port = args.port or auto_pick_port()
            if not port:
                print("未找到串口，请指定 --port /dev/ttyACM0")
                return 1
            if port.startswith("/dev/cu.") and sys.platform.startswith("linux"):
                print(f"警告: {port} 是 macOS 设备名，请改用 /dev/ttyACM* 或 /dev/ttyUSB*")
                return 1
            print(f"连接机械臂 {port} @ baud={DEFAULT_BAUD} protocol={PROTOCOL_VERSION}")
            bus = connect_bus(port)
            arm = ArmController(bus)

        runner = FrontRedGrasp(cfg, arm=arm, camera=camera)
        ok = runner.run(dry_run=args.dry_run)
        return 0 if ok else 1
    finally:
        camera.close()
        if bus is not None:
            try:
                # 抓取成功后通常希望保持力矩夹住物体；失败时也可保持使能
                pass
            finally:
                try:
                    bus.disconnect()
                except Exception:
                    pass
                print("已断开机械臂连接。")


if __name__ == "__main__":
    raise SystemExit(main())
