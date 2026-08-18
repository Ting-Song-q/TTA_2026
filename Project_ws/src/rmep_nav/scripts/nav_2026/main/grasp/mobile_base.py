"""ROS open-loop mobile-base control."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path


class TimedMobileBase:
    def __init__(self, topic: str, rate_hz: int, ros_python: str, helper: Path) -> None:
        self.topic = topic
        self.rate_hz = int(rate_hz)
        self.ros_python = ros_python
        self.helper = helper
        if not helper.is_file():
            raise FileNotFoundError("ROS cmd_vel helper not found: %s" % helper)
        print(f"[base] system ROS helper ready: python={ros_python} topic={topic}", flush=True)

    def stop(self) -> None:
        self.drive(vx=0.0, vy=0.0, duration_s=0.0, label="stop")

    def drive(self, *, vx: float = 0.0, vy: float = 0.0, duration_s: float, label: str) -> None:
        duration_s = max(0.0, float(duration_s))
        print(f"[base] {label}: vx={vx:+.3f} vy={vy:+.3f} t={duration_s:.2f}s", flush=True)
        command = [
            self.ros_python, str(self.helper), "--topic", self.topic,
            "--vx", str(float(vx)), "--vy", str(float(vy)),
            "--duration", str(duration_s), "--rate", str(self.rate_hz), "--label", label,
        ]
        subprocess.run(command, check=True, timeout=max(10.0, duration_s + 5.0))
        print(f"[base] {label}: stopped", flush=True)

    def forward_m(self, distance_m: float, speed_mps: float) -> None:
        self.drive(vx=abs(speed_mps), duration_s=abs(distance_m) / abs(speed_mps), label="forward")

    def backward_m(self, distance_m: float, speed_mps: float) -> None:
        self.drive(vx=-abs(speed_mps), duration_s=abs(distance_m) / abs(speed_mps), label="backward")

    def strafe_m(self, distance_m: float, speed_mps: float) -> None:
        direction = "left" if distance_m >= 0 else "right"
        self.drive(vy=(1.0 if distance_m >= 0 else -1.0) * abs(speed_mps), duration_s=abs(distance_m) / abs(speed_mps), label=f"strafe_{direction}")

    def strafe_for_duration(self, direction: float, speed_mps: float, duration_s: float) -> None:
        """Strafe at a fixed speed for a fixed pulse duration."""
        sign = 1.0 if direction >= 0 else -1.0
        label = "strafe_left" if sign > 0 else "strafe_right"
        self.drive(vy=sign * abs(speed_mps), duration_s=duration_s, label=label)

    def pause(self, seconds: float) -> None:
        self.stop()
        time.sleep(max(0.0, float(seconds)))
