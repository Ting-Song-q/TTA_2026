#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 正运动学：关节 ticks → 基座系末端齐次变换 T_base_ee。

运动链取自 OpenRAL / SO-ARM100 ``so101_new_calib`` 的关节 origin（米、弧度）。
零位约定：各关节 mid_ticks（默认 2048）对应 URDF 零角；符号可用 yaml 翻转。

末端帧：``gripper_base``（腕滚后、夹爪开合之前），适合眼在手上的手眼标定。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

# (origin_xyz_m, origin_rpy_rad, axis_xyz)
# wrist_roll 在部分清单里缺 origin，按单位变换处理。
DEFAULT_JOINT_ORIGINS: Dict[str, Tuple[Sequence[float], Sequence[float], Sequence[float]]] = {
    "shoulder_pan": (
        (0.0207909, -0.0230745, 0.0948817),
        (-np.pi, 0.0, np.pi / 2),
        (0.0, 0.0, 1.0),
    ),
    "shoulder_lift": (
        (-0.0303992, -0.0182778, -0.0542),
        (np.pi / 2, -np.pi / 2, np.pi),
        (0.0, 0.0, 1.0),
    ),
    "elbow_flex": (
        (-0.11257, -0.028, 0.0),
        (0.0, 0.0, np.pi / 2),
        (0.0, 0.0, 1.0),
    ),
    "wrist_flex": (
        (-0.1349, 0.0052, 0.0),
        (0.0, 0.0, -np.pi / 2),
        (0.0, 0.0, 1.0),
    ),
    "wrist_roll": (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
}

TICKS_PER_REV = 4096.0


def rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF 约定：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def axis_angle_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=float)
    n = np.linalg.norm(ax)
    if n < 1e-12:
        return np.eye(3)
    ax = ax / n
    x, y, z = ax
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def make_transform(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def joint_local_transform(
    origin_xyz: Sequence[float],
    origin_rpy: Sequence[float],
    axis: Sequence[float],
    q: float,
) -> np.ndarray:
    T_origin = make_transform(rpy_matrix(origin_rpy), origin_xyz)
    T_joint = make_transform(axis_angle_matrix(axis, q), (0.0, 0.0, 0.0))
    return T_origin @ T_joint


def ticks_to_rad(
    ticks: float,
    *,
    mid_ticks: float = 2048.0,
    sign: float = 1.0,
    ticks_per_rev: float = TICKS_PER_REV,
) -> float:
    return float(sign) * (float(ticks) - float(mid_ticks)) * (2.0 * np.pi / float(ticks_per_rev))


def ticks_dict_to_rad(
    ticks: Mapping[str, float],
    *,
    mid_ticks: Optional[Mapping[str, float]] = None,
    signs: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    mid = mid_ticks or {}
    sgn = signs or {}
    out: Dict[str, float] = {}
    for name in JOINT_ORDER:
        if name not in ticks:
            raise KeyError(f"missing joint ticks: {name}")
        out[name] = ticks_to_rad(
            ticks[name],
            mid_ticks=float(mid.get(name, 2048.0)),
            sign=float(sgn.get(name, 1.0)),
        )
    return out


def matrix_to_rpy_xyz(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 外旋 xyz 欧拉角 (rad)，不依赖 scipy。

    与 ``rpy_matrix`` / SciPy ``Rotation.as_euler('xyz')`` 约定一致。
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    sy = -float(R[2, 0])
    sy = max(-1.0, min(1.0, sy))
    pitch = float(np.arcsin(sy))
    if abs(sy) < 0.999999:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        # 万向节锁：yaw 置 0，把剩余角并入 roll
        roll = float(np.arctan2(-R[0, 1], R[1, 1]))
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


class SO101FK:
    def __init__(
        self,
        *,
        mid_ticks: Optional[Mapping[str, float]] = None,
        signs: Optional[Mapping[str, float]] = None,
        joint_origins: Optional[Mapping[str, dict]] = None,
        ee_offset_xyz: Sequence[float] = (0.0, 0.0, 0.0),
        ee_offset_rpy: Sequence[float] = (0.0, 0.0, 0.0),
    ):
        self.mid_ticks = {n: float((mid_ticks or {}).get(n, 2048.0)) for n in JOINT_ORDER}
        self.signs = {n: float((signs or {}).get(n, 1.0)) for n in JOINT_ORDER}
        self.origins = dict(DEFAULT_JOINT_ORIGINS)
        if joint_origins:
            for name, spec in joint_origins.items():
                xyz = tuple(spec.get("xyz", self.origins[name][0]))
                rpy = tuple(spec.get("rpy", self.origins[name][1]))
                axis = tuple(spec.get("axis", self.origins[name][2]))
                self.origins[name] = (xyz, rpy, axis)
        self.T_ee_offset = make_transform(rpy_matrix(ee_offset_rpy), ee_offset_xyz)

    @classmethod
    def from_config(cls, cfg: Mapping) -> "SO101FK":
        fk_cfg = cfg.get("fk") if isinstance(cfg.get("fk"), dict) else cfg
        return cls(
            mid_ticks=fk_cfg.get("mid_ticks"),
            signs=fk_cfg.get("signs"),
            joint_origins=fk_cfg.get("joint_origins"),
            ee_offset_xyz=fk_cfg.get("ee_offset_xyz", (0.0, 0.0, 0.0)),
            ee_offset_rpy=fk_cfg.get("ee_offset_rpy", (0.0, 0.0, 0.0)),
        )

    def forward_rad(self, q: Mapping[str, float]) -> np.ndarray:
        T = np.eye(4, dtype=float)
        for name in JOINT_ORDER:
            xyz, rpy, axis = self.origins[name]
            T = T @ joint_local_transform(xyz, rpy, axis, float(q[name]))
        return T @ self.T_ee_offset

    def forward_ticks(self, ticks: Mapping[str, float]) -> np.ndarray:
        q = ticks_dict_to_rad(ticks, mid_ticks=self.mid_ticks, signs=self.signs)
        return self.forward_rad(q)

    def pose_xyz_rpy(self, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """返回位置 (m) 与外旋 xyz 欧拉角 (rad)。"""
        xyz = T[:3, 3].copy()
        rpy = matrix_to_rpy_xyz(T[:3, :3])
        return xyz, rpy


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti
