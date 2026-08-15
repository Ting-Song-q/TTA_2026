#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 眼在手上：手眼标定数据采集。

每位姿保存：
  - 相机图 images/NNN.jpg
  - 关节 ticks
  - T_base_ee（so101_fk）
  - T_cam_board（内参 + solvePnP）

流程建议:
  1. 桌面固定棋盘（与内参标定时相同规格）
  2. SSH 无屏推荐:
       python3 collect_hand_eye.py --yes --teach --headless
     力矩关闭后手动摆臂，看到 FOUND 后按 Enter 存样；采够自动结束
  3. 下一脚本可用 samples.json 做 calibrateHandEye

依赖:
  - output/camera_calib/camera_intrinsics.yaml（或 --intrinsics）
  - og.yaml 的串口（可用 --port 覆盖）
"""

from __future__ import annotations

import argparse
import copy
import json
import select
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from calibrate_camera import (  # noqa: E402
    build_object_points,
    find_corners,
    open_capture,
)
from so101_fk import JOINT_ORDER, SO101FK  # noqa: E402
from start import (  # noqa: E402
    DEFAULT_BAUD,
    connect_bus,
    disable_torque_safe,
    enable_torque_safe,
    read_positions,
)

DEFAULT_INTRINSICS = _HERE.parent / "output" / "camera_calib" / "camera_intrinsics.yaml"
DEFAULT_OG_YAML = _HERE / "og.yaml"


def deep_update(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"yaml root must be mapping: {path}")
    return data


def load_intrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"未找到内参: {path}\n请先运行 calibrate_camera.py 或指定 --intrinsics"
        )
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        K = np.asarray(data["K"], dtype=float)
        dist = np.asarray(data["dist"], dtype=float).reshape(-1)
        meta = {"source": str(path)}
        return K, dist, meta
    raw = load_yaml(path)
    if "camera_matrix" in raw:
        K = np.asarray(raw["camera_matrix"], dtype=float)
        dist = np.asarray(raw.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=float).reshape(-1)
    elif path.with_suffix(".npz").exists():
        return load_intrinsics(path.with_suffix(".npz"))
    else:
        raise ValueError(f"内参文件缺少 camera_matrix: {path}")
    return K, dist, raw


def _stdin_line(timeout_s: float) -> Optional[str]:
    if not sys.stdin.isatty():
        time.sleep(timeout_s)
        return None
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    except (OSError, ValueError):
        time.sleep(timeout_s)
        return None
    if not ready:
        return None
    line = sys.stdin.readline()
    if not line:
        return None
    return line.strip().lower()


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def solve_board_pose(
    corners: np.ndarray,
    objp_m: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    ok, rvec, tvec = cv2.solvePnP(
        objp_m,
        corners,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return False, None, None, None
    T = rvec_tvec_to_T(rvec, tvec)
    return True, T, rvec.reshape(3), tvec.reshape(3)


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_meta(path: Path, meta: dict) -> None:
    path.write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")


def collect_loop(
    *,
    cap: cv2.VideoCapture,
    bus,
    fk: SO101FK,
    K: np.ndarray,
    dist: np.ndarray,
    pattern: Tuple[int, int],
    square_mm: float,
    out_dir: Path,
    min_samples: int,
    teach: bool,
) -> int:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    poses_path = out_dir / "poses_xyzrpy.txt"  # x,y,z,rx,ry,rz（米/弧度），兼容旧工具
    ticks_path = out_dir / "ticks.jsonl"

    objp_mm = build_object_points(pattern[0], pattern[1], square_mm)
    objp_m = objp_mm / 1000.0  # solvePnP 与 FK 统一用米

    samples: List[dict] = []
    last_status = ""
    idx = 0

    print(f"输出目录: {out_dir}")
    print(f"棋盘内角点={pattern[0]}x{pattern[1]} 方格={square_mm}mm")
    print(f"模式: {'示教(力矩OFF，手动摆臂)' if teach else '持力(力矩ON)'}")
    print("命令: Enter=保存当前样本  c=结束并汇总  q=放弃退出")
    print(f"目标样本数: {min_samples}（建议 ≥12，位姿要有明显旋转差异）")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_corners(gray, pattern)
            status = (
                f"FOUND  n={len(samples)}/{min_samples}"
                if found
                else f"no-board  n={len(samples)}/{min_samples}"
            )
            if status != last_status:
                print(f"[{time.strftime('%H:%M:%S')}] {status}")
                last_status = status

            cmd = _stdin_line(0.05)
            if cmd is None:
                continue

            if cmd in {"q", "quit", "exit"}:
                print("退出，未写汇总 json（jsonl 已增量保存）")
                return 1

            if cmd in {"c", "done", "finish"}:
                break

            # Enter 或 s / save
            if cmd not in {"", "s", "save"}:
                print(f"未知命令: {cmd!r}（Enter/s 保存，c 结束，q 退出）")
                continue

            # 按键瞬间检测可能闪烁：连采多帧直到检出棋盘
            frame_ok = None
            corners_ok = None
            for _ in range(15):
                ok_f, fr = cap.read()
                if not ok_f or fr is None:
                    time.sleep(0.03)
                    continue
                gray_f = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                found_f, corners_f = find_corners(gray_f, pattern)
                if found_f and corners_f is not None:
                    frame_ok, corners_ok = fr, corners_f
                    break
                time.sleep(0.03)
            if frame_ok is None or corners_ok is None:
                print("[skip] 未检测到完整棋盘（已重试），摆稳后再按 Enter")
                continue
            frame, corners = frame_ok, corners_ok

            ok_pnp, T_cam_board, rvec, tvec = solve_board_pose(corners, objp_m, K, dist)
            if not ok_pnp or T_cam_board is None:
                print("[skip] solvePnP 失败")
                continue

            ticks_raw = read_positions(bus)
            ticks = {k: float(ticks_raw[k]) for k in JOINT_ORDER}
            # 夹爪一并记录，FK 不使用
            if "gripper" in ticks_raw:
                ticks_all = {**ticks, "gripper": float(ticks_raw["gripper"])}
            else:
                ticks_all = dict(ticks)

            T_base_ee = fk.forward_ticks(ticks)
            xyz, rpy = fk.pose_xyz_rpy(T_base_ee)

            idx += 1
            img_name = f"{idx:03d}.jpg"
            img_path = images_dir / img_name
            # 标注角点便于事后检查
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, pattern, corners, True)
            cv2.imwrite(str(img_path), frame)
            cv2.imwrite(str(images_dir / f"{idx:03d}_corners.jpg"), vis)

            row = {
                "index": idx,
                "image": f"images/{img_name}",
                "ticks": ticks_all,
                "T_base_ee": T_base_ee.tolist(),
                "ee_xyz_m": xyz.tolist(),
                "ee_rpy_rad": rpy.tolist(),
                "T_cam_board": T_cam_board.tolist(),
                "rvec": rvec.tolist() if rvec is not None else None,
                "tvec": tvec.tolist() if tvec is not None else None,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            samples.append(row)
            append_jsonl(samples_path, row)
            append_jsonl(ticks_path, {"index": idx, "ticks": ticks_all})
            with poses_path.open("a", encoding="utf-8") as f:
                f.write(
                    ",".join(
                        f"{v:.8f}"
                        for v in [
                            float(xyz[0]),
                            float(xyz[1]),
                            float(xyz[2]),
                            float(rpy[0]),
                            float(rpy[1]),
                            float(rpy[2]),
                        ]
                    )
                    + "\n"
                )

            print(
                f"[saved] #{idx}  ee_xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) "
                f"board_z={float(T_cam_board[2, 3]):.3f}m  -> {img_name}"
            )

            if len(samples) >= min_samples:
                print(f"已达目标样本数 {min_samples}")
                break
    finally:
        pass

    if len(samples) < 5:
        print(f"[error] 有效样本过少: {len(samples)}（至少 5，建议 ≥12）")
        return 1

    summary = {
        "n_samples": len(samples),
        "pattern_cols": pattern[0],
        "pattern_rows": pattern[1],
        "square_mm": square_mm,
        "samples": samples,
    }
    (out_dir / "samples.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"汇总: {out_dir / 'samples.json'}")
    print("下一步: 用这些样本跑手眼求解（calibrateHandEye）得到 T_ee_cam")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SO-101 眼在手上：手眼数据采集")
    parser.add_argument("--config", type=Path, default=DEFAULT_OG_YAML, help="og.yaml（取串口等）")
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--camera", default="/dev/video11")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pattern", default="8x6", help="棋盘内角点 colsxrows")
    parser.add_argument("--square-mm", type=float, default=24.0)
    parser.add_argument("--min-samples", type=int, default=12)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="默认 output/hand_eye_<timestamp>",
    )
    parser.add_argument(
        "--teach",
        action="store_true",
        help="示教模式：力矩关闭，手动摆臂后 Enter 采集",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无 GUI（默认即终端交互；保留此开关便于脚本统一）",
    )
    parser.add_argument("--yes", action="store_true", help="确认连接电机/相机")
    parser.add_argument("--fk-config", type=Path, default=None, help="可选 FK yaml")
    args = parser.parse_args()

    if not args.yes:
        print("Refusing: pass --yes")
        return 2

    try:
        cols_s, rows_s = args.pattern.lower().split("x")
        pattern = (int(cols_s), int(rows_s))
    except ValueError:
        print("[error] --pattern 格式应为 9x6")
        return 2

    intrinsics_path = args.intrinsics
    if not intrinsics_path.is_absolute():
        cand = (_HERE.parent / intrinsics_path).resolve()
        intrinsics_path = cand if cand.exists() else (Path.cwd() / intrinsics_path).resolve()
    K, dist, K_meta = load_intrinsics(intrinsics_path)
    print(f"[intrinsics] {intrinsics_path}")
    print(f"  fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")

    cfg = {}
    if args.config.exists():
        cfg = load_yaml(args.config)
        print(f"[config] {args.config}")
    port = args.port or cfg.get("port", "/dev/ttyACM1")
    baud = int(args.baud or cfg.get("baud") or DEFAULT_BAUD)

    fk_cfg: dict = {}
    if args.fk_config and args.fk_config.exists():
        fk_cfg = load_yaml(args.fk_config)
    elif isinstance(cfg.get("fk"), dict):
        fk_cfg = cfg
    fk = SO101FK.from_config(fk_cfg) if fk_cfg else SO101FK()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = _HERE.parent / "output" / f"hand_eye_{stamp}"
    elif not out_dir.is_absolute():
        out_dir = (_HERE.parent / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    write_meta(
        out_dir / "meta.yaml",
        {
            "created_at": stamp,
            "camera": args.camera,
            "width": args.width,
            "height": args.height,
            "intrinsics": str(intrinsics_path),
            "pattern": f"{pattern[0]}x{pattern[1]}",
            "square_mm": float(args.square_mm),
            "port": port,
            "baud": baud,
            "teach": bool(args.teach),
            "ee_frame": "gripper_base (so101_fk)",
            "K": K.tolist(),
            "dist": dist.reshape(-1).tolist(),
            "intrinsics_meta_keys": list(K_meta.keys()) if isinstance(K_meta, dict) else [],
        },
    )

    # 复制内参到会话目录，便于归档
    try:
        (out_dir / "camera_intrinsics.yaml").write_text(
            yaml.safe_dump(
                {
                    "camera_matrix": K.tolist(),
                    "dist_coeffs": dist.reshape(-1).tolist(),
                    "source": str(intrinsics_path),
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass

    print(f"连接电机 {port} @ {baud} ...")
    bus = connect_bus(port, baud=baud, configure=False)
    try:
        if args.teach:
            disable_torque_safe(bus)
            print("[torque] OFF（示教：用手摆到新位姿后按 Enter）")
        else:
            enable_torque_safe(bus)
            print("[torque] ON（请用其它方式换位姿，或改用 --teach）")
        time.sleep(0.2)

        try:
            cap = open_capture(args.camera, args.width, args.height, args.fps)
        except RuntimeError as exc:
            print(f"[error] {exc}")
            return 2

        try:
            return collect_loop(
                cap=cap,
                bus=bus,
                fk=fk,
                K=K,
                dist=dist,
                pattern=pattern,
                square_mm=float(args.square_mm),
                out_dir=out_dir,
                min_samples=int(args.min_samples),
                teach=bool(args.teach),
            )
        finally:
            cap.release()
    finally:
        if args.teach:
            # 示教结束保持力矩关闭，避免突然上电
            try:
                disable_torque_safe(bus)
            except Exception:
                pass
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass
        print("已断开串口")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
