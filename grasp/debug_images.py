"""Deterministic, annotated camera images for one mobile-grasp run."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np


class DebugImages:
    def __init__(self, directory: Path, enabled: bool = True) -> None:
        self.directory = directory
        self.enabled = enabled

    def reset(self) -> None:
        if not self.enabled:
            return
        if self.directory.exists():
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        print(f"[debug] reset image directory: {self.directory}", flush=True)

    def save(self, tag: str, frame: np.ndarray, detector, detection, *,
             target_u: float | None = None, tolerance_px: float | None = None,
             status: str = "", candidates=None) -> Path | None:
        if not self.enabled:
            return None
        if detector is None:
            vis = frame.copy()
        else:
            vis = detector.draw(frame, detection, candidates)
        height, width = vis.shape[:2]
        if target_u is not None and tolerance_px is not None:
            left = max(0, int(round(target_u - tolerance_px)))
            right = min(width - 1, int(round(target_u + tolerance_px)))
            overlay = vis.copy()
            cv2.rectangle(overlay, (left, 0), (right, height - 1), (0, 180, 0), -1)
            cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)
            cv2.line(vis, (int(round(target_u)), 0), (int(round(target_u)), height - 1), (0, 255, 0), 2)
            cv2.putText(vis, f"target u={target_u:.1f} +/- {tolerance_px:.0f}px",
                        (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
        if detection is not None:
            cv2.drawMarker(vis, (round(detection.center_u), round(detection.center_v)),
                           (255, 0, 255), cv2.MARKER_CROSS, 26, 2)
        if status:
            cv2.putText(vis, status, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 3)
            cv2.putText(vis, status, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (30, 30, 30), 1)
        path = self.directory / f"{tag}.jpg"
        cv2.imwrite(str(path), vis)
        print(f"[debug] saved: {path}", flush=True)
        return path
