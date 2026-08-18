# -*- coding: utf-8 -*-
from pathlib import Path

src = Path(r"D:\code\tta\grasp_test\grasp_vision\test2.py").read_text(encoding="utf-8")
start = src.index("class RedFrameDetection")
end = src.index("class Camera")
chunk = src[start:end]
header = '''# -*- coding: utf-8 -*-
"""红色镂空放置框检测（自 ArmPi Pro grasp_vision/test2.py 移植，无 ROS 依赖）。"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


'''
out = Path(r"D:\code\tta\tta\arm\red_place_frame.py")
out.write_text(header + chunk, encoding="utf-8")
print("wrote", out, "chars", len(header + chunk))
