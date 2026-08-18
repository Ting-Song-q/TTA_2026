"""Compatibility helper required by arm/pixel_to_base.py.

The mobile prototype only uses the existing capture_frame helper, but
pixel_to_base imports open_capture at module load time.
"""

from __future__ import annotations

import cv2


def open_capture(index_or_path, width: int, height: int, fps: int):
    capture = cv2.VideoCapture(index_or_path)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    capture.set(cv2.CAP_PROP_FPS, int(fps))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open camera: {index_or_path}")
    return capture
