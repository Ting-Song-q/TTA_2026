#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 数值逆解：目标基座位置 → 关节 ticks（基于 so101_fk）。"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from so101_fk import JOINT_ORDER, SO101FK, TICKS_PER_REV, ticks_to_rad


def rad_to_ticks(
    q_rad: Mapping[str, float],
    *,
    mid_ticks: Optional[Mapping[str, float]] = None,
    signs: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    mid = mid_ticks or {}
    sgn = signs or {}
    out: Dict[str, float] = {}
    for name in JOINT_ORDER:
        m = float(mid.get(name, 2048.0))
        s = float(sgn.get(name, 1.0))
        # ticks = mid + sign^{-1} * q * ticks_per_rev / 2pi
        out[name] = m + (float(q_rad[name]) / s) * (TICKS_PER_REV / (2.0 * np.pi))
    return out


def clamp_ticks(
    ticks: Mapping[str, float],
    limits: Mapping[str, Sequence[float]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, val in ticks.items():
        lim = limits.get(name)
        if lim:
            out[name] = float(max(float(lim[0]), min(float(lim[1]), float(val))))
        else:
            out[name] = float(val)
    return out


class SO101IK:
    """阻尼最小二乘位置逆解（可选保持当前姿态）。"""

    def __init__(self, fk: Optional[SO101FK] = None):
        self.fk = fk or SO101FK()

    def _jacobian_pos(self, q_rad: Dict[str, float], eps: float = 1e-4) -> np.ndarray:
        T0 = self.fk.forward_rad(q_rad)
        p0 = T0[:3, 3]
        J = np.zeros((3, len(JOINT_ORDER)), dtype=float)
        for i, name in enumerate(JOINT_ORDER):
            qp = dict(q_rad)
            qp[name] = float(qp[name]) + eps
            pp = self.fk.forward_rad(qp)[:3, 3]
            J[:, i] = (pp - p0) / eps
        return J

    def solve_position(
        self,
        target_xyz_m: Sequence[float],
        q_ticks_seed: Mapping[str, float],
        *,
        joint_limits: Optional[Mapping[str, Sequence[float]]] = None,
        max_iters: int = 80,
        pos_tol_m: float = 0.008,
        damp: float = 1e-3,
        step_scale: float = 0.6,
        max_dq_rad: float = 0.12,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """返回 (ticks_arm, info)。ticks 仅含 5 个臂关节。"""
        target = np.asarray(target_xyz_m, dtype=float).reshape(3)
        q = {
            n: ticks_to_rad(
                float(q_ticks_seed[n]),
                mid_ticks=float(self.fk.mid_ticks[n]),
                sign=float(self.fk.signs[n]),
            )
            for n in JOINT_ORDER
        }
        limits = joint_limits or {}
        last_err = 1e9
        info: Dict[str, float] = {}

        for it in range(max_iters):
            T = self.fk.forward_rad(q)
            p = T[:3, 3]
            err = target - p
            err_n = float(np.linalg.norm(err))
            last_err = err_n
            if err_n < pos_tol_m:
                ticks = rad_to_ticks(q, mid_ticks=self.fk.mid_ticks, signs=self.fk.signs)
                ticks = clamp_ticks(ticks, limits)
                info = {"iters": float(it), "pos_err_m": err_n, "ok": 1.0}
                return ticks, info

            J = self._jacobian_pos(q)
            # dq = J^T (J J^T + λI)^{-1} err
            JJt = J @ J.T + damp * np.eye(3)
            try:
                dq = J.T @ np.linalg.solve(JJt, err)
            except np.linalg.LinAlgError:
                dq = J.T @ err
            n = float(np.linalg.norm(dq))
            if n > max_dq_rad:
                dq = dq * (max_dq_rad / n)
            dq *= step_scale
            for i, name in enumerate(JOINT_ORDER):
                q[name] = float(q[name] + dq[i])

            # 投影回限位（经 ticks）
            ticks_tmp = rad_to_ticks(q, mid_ticks=self.fk.mid_ticks, signs=self.fk.signs)
            ticks_tmp = clamp_ticks(ticks_tmp, limits)
            q = {
                n: ticks_to_rad(
                    ticks_tmp[n],
                    mid_ticks=float(self.fk.mid_ticks[n]),
                    sign=float(self.fk.signs[n]),
                )
                for n in JOINT_ORDER
            }

        ticks = rad_to_ticks(q, mid_ticks=self.fk.mid_ticks, signs=self.fk.signs)
        ticks = clamp_ticks(ticks, limits)
        info = {"iters": float(max_iters), "pos_err_m": last_err, "ok": 0.0}
        return ticks, info

    def solve_position_best(
        self,
        target_xyz_m: Sequence[float],
        seeds: Sequence[Mapping[str, float]],
        *,
        joint_limits: Optional[Mapping[str, Sequence[float]]] = None,
        max_iters: int = 100,
        pos_tol_m: float = 0.008,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """多种子取误差最小的解。"""
        best_ticks: Optional[Dict[str, float]] = None
        best_info: Dict[str, float] = {"pos_err_m": 1e9, "ok": 0.0, "iters": 0.0}
        for seed in seeds:
            ticks, info = self.solve_position(
                target_xyz_m,
                seed,
                joint_limits=joint_limits,
                max_iters=max_iters,
                pos_tol_m=pos_tol_m,
            )
            if info["pos_err_m"] < best_info["pos_err_m"]:
                best_ticks, best_info = ticks, info
            if info.get("ok"):
                break
        assert best_ticks is not None
        return best_ticks, best_info

    def find_reachable(
        self,
        start_xyz: Sequence[float],
        target_xyz: Sequence[float],
        q_ticks_seed: Mapping[str, float],
        *,
        joint_limits: Optional[Mapping[str, Sequence[float]]] = None,
        pos_tol_m: float = 0.012,
        max_err_m: float = 0.02,
        samples: int = 12,
    ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
        """在 start→target 线段上找最靠近 target 且 IK 误差 < max_err_m 的点。"""
        p0 = np.asarray(start_xyz, dtype=float).reshape(3)
        p1 = np.asarray(target_xyz, dtype=float).reshape(3)
        best_alpha = 0.0
        best_ticks = {n: float(q_ticks_seed[n]) for n in JOINT_ORDER}
        best_info: Dict[str, float] = {"pos_err_m": 1e9, "ok": 0.0, "iters": 0.0}
        seed = dict(q_ticks_seed)
        for i in range(1, samples + 1):
            alpha = i / float(samples)
            pt = p0 + alpha * (p1 - p0)
            ticks, info = self.solve_position(
                pt, seed, joint_limits=joint_limits, pos_tol_m=pos_tol_m, max_iters=80
            )
            if info["pos_err_m"] <= max_err_m:
                best_alpha = alpha
                best_ticks = ticks
                best_info = dict(info)
                best_info["alpha"] = alpha
                seed = ticks
        pt = p0 + best_alpha * (p1 - p0)
        best_info["alpha"] = best_alpha
        return pt, best_ticks, best_info

