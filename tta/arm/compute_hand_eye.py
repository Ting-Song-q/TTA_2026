#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 眼在手上：由采集样本求解 T_ee_cam。

输入: collect_hand_eye.py 输出的 samples.json
输出: handeye_ee_cam.yaml / .npz（相机 → 末端 gripper_base）

用法:
  python3 compute_hand_eye.py
  python3 compute_hand_eye.py --data-dir /home/tta/tta/output/hand_eye_20260813_155049
  python3 compute_hand_eye.py --method park
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_OUTPUT = _HERE.parent / "output"

METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def find_latest_hand_eye_dir(root: Path) -> Optional[Path]:
    if not root.is_dir():
        return None
    cands = sorted(
        [p for p in root.glob("hand_eye_*") if (p / "samples.json").is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def load_samples(data_dir: Path) -> List[dict]:
    path = data_dir / "samples.json"
    if not path.exists():
        # 兼容仅有 jsonl
        jsonl = data_dir / "samples.jsonl"
        if not jsonl.exists():
            raise FileNotFoundError(f"未找到 samples.json / samples.jsonl: {data_dir}")
        rows = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "samples" in raw:
        return list(raw["samples"])
    if isinstance(raw, list):
        return raw
    raise ValueError(f"无法解析 samples: {path}")


def boards_consistency(T_ee_cam: np.ndarray, samples: List[dict]) -> Dict[str, float]:
    """各样本推到基座的棋盘位姿应接近常数；返回平移标准差等。"""
    origins = []
    for s in samples:
        T_base_ee = np.asarray(s["T_base_ee"], dtype=float)
        T_cam_board = np.asarray(s["T_cam_board"], dtype=float)
        T_base_board = T_base_ee @ T_ee_cam @ T_cam_board
        origins.append(T_base_board[:3, 3])
    P = np.stack(origins, axis=0)
    mean = P.mean(axis=0)
    std = P.std(axis=0)
    rms = float(np.sqrt(np.mean(np.sum((P - mean) ** 2, axis=1))))
    return {
        "board_origin_mean_m": mean.tolist(),
        "board_origin_std_m": std.tolist(),
        "board_origin_rms_m": rms,
        "board_origin_rms_mm": rms * 1000.0,
    }


def solve_hand_eye(
    samples: List[dict], method_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R_g2b: List[np.ndarray] = []
    t_g2b: List[np.ndarray] = []
    R_t2c: List[np.ndarray] = []
    t_t2c: List[np.ndarray] = []

    for s in samples:
        T_base_ee = np.asarray(s["T_base_ee"], dtype=float)
        T_cam_board = np.asarray(s["T_cam_board"], dtype=float)
        R_g2b.append(T_base_ee[:3, :3].copy())
        t_g2b.append(T_base_ee[:3, 3].reshape(3, 1).copy())
        R_t2c.append(T_cam_board[:3, :3].copy())
        t_t2c.append(T_cam_board[:3, 3].reshape(3, 1).copy())

    method = METHODS[method_name]
    R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    T_ee_cam = make_T(R_c2g, t_c2g)
    return T_ee_cam, np.asarray(R_c2g, dtype=float), np.asarray(t_c2g, dtype=float).reshape(3)


def try_all_methods(samples: List[dict]) -> List[Tuple[str, np.ndarray, Dict[str, float]]]:
    results = []
    for name in METHODS:
        try:
            T, _, _ = solve_hand_eye(samples, name)
            stats = boards_consistency(T, samples)
            results.append((name, T, stats))
        except cv2.error as exc:
            print(f"[warn] method={name} 失败: {exc}")
    results.sort(key=lambda x: x[2]["board_origin_rms_m"])
    return results


def save_result(
    out_dir: Path,
    T_ee_cam: np.ndarray,
    method: str,
    stats: dict,
    n_samples: int,
    data_dir: Path,
) -> Tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    R = T_ee_cam[:3, :3]
    t = T_ee_cam[:3, 3]
    payload = {
        "type": "eye_in_hand",
        "T_name": "T_ee_cam",
        "description": "camera pose in end-effector (gripper_base) frame: P_ee = R @ P_cam + t",
        "ee_frame": "gripper_base (so101_fk)",
        "method": method,
        "n_samples": n_samples,
        "data_dir": str(data_dir),
        "created_at": stamp,
        "rotation_matrix": R.tolist(),
        "translation_m": t.tolist(),
        "T_ee_cam": T_ee_cam.tolist(),
        "quality": stats,
    }
    yaml_path = out_dir / "handeye_ee_cam.yaml"
    npz_path = out_dir / "handeye_ee_cam.npz"
    yaml_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    np.savez(npz_path, T_ee_cam=T_ee_cam, R=R, t=t)
    (out_dir / f"handeye_ee_cam_{stamp}.yaml").write_text(
        yaml_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    # 同步一份到全局 output，方便后续抓取脚本默认加载
    global_yaml = _OUTPUT / "handeye_ee_cam.yaml"
    global_npz = _OUTPUT / "handeye_ee_cam.npz"
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    global_yaml.write_text(yaml_path.read_text(encoding="utf-8"), encoding="utf-8")
    np.savez(global_npz, T_ee_cam=T_ee_cam, R=R, t=t)
    return yaml_path, global_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="眼在手上：求解 T_ee_cam")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="含 samples.json 的目录；默认取最新 hand_eye_*",
    )
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", *METHODS.keys()],
        help="calibrateHandEye 算法；auto 会试全部并选棋盘位姿最一致的",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = find_latest_hand_eye_dir(_OUTPUT)
        if data_dir is None:
            print(f"[error] 未找到 {_OUTPUT}/hand_eye_*/samples.json")
            return 2
    elif not data_dir.is_absolute():
        cand = (_OUTPUT / data_dir).resolve()
        data_dir = cand if cand.exists() else (Path.cwd() / data_dir).resolve()

    print(f"[data] {data_dir}")
    samples = load_samples(data_dir)
    if len(samples) < 5:
        print(f"[error] 样本过少: {len(samples)}")
        return 1
    print(f"样本数: {len(samples)}")

    if args.method == "auto":
        ranked = try_all_methods(samples)
        if not ranked:
            print("[error] 所有方法均失败（运动旋转是否太小？）")
            return 1
        print("\n各方法棋盘原点一致性（越小越好）:")
        for name, _, st in ranked:
            print(
                f"  {name:10s}  rms={st['board_origin_rms_mm']:.1f} mm  "
                f"std={np.array(st['board_origin_std_m'])*1000}"
            )
        method, T_ee_cam, stats = ranked[0]
        print(f"\n选用: {method}")
    else:
        method = args.method
        T_ee_cam, _, _ = solve_hand_eye(samples, method)
        stats = boards_consistency(T_ee_cam, samples)

    R = T_ee_cam[:3, :3]
    t = T_ee_cam[:3, 3]
    print("\n===== T_ee_cam（相机 → 末端）=====")
    print(f"R =\n{R}")
    print(f"t (m) = {t}")
    print(
        f"质量: 棋盘在基座下原点 RMS ≈ {stats['board_origin_rms_mm']:.1f} mm "
        f"(std={np.array(stats['board_origin_std_m'])*1000} mm)"
    )
    if stats["board_origin_rms_mm"] > 30:
        print("[warn] RMS>30mm：位姿旋转可能不够，或 FK/内参/棋盘尺寸有误，建议重采")
    elif stats["board_origin_rms_mm"] > 15:
        print("[warn] RMS 偏大，抓取精度可能一般；可再采更多、差异更大的位姿")
    else:
        print("[ok] 一致性尚可，可用于后续像素→基座坐标")

    yaml_path, global_yaml = save_result(
        data_dir, T_ee_cam, method, stats, len(samples), data_dir
    )
    print(f"saved: {yaml_path}")
    print(f"saved: {global_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
