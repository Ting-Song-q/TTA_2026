#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 腕部相机内参一键标定（OpenCV 棋盘格）。

不依赖 ROS。默认棋盘：内角点 9×6，方格边长 25mm（请按实际修改）。

有显示器:
  python3 calibrate_camera.py --camera /dev/video11
  # 空格保存 / c 标定 / u 去畸变预览 / q 退出

无显示器（SSH）——推荐自动采图:
  python3 calibrate_camera.py --headless --camera /dev/video11
  # 检测到棋盘自动存图；凑够张数自动标定
  # 终端也可: Enter=尝试存一帧  c=立即标定  q=退出

仅从已存图片标定:
  python3 calibrate_camera.py --from-dir output/camera_calib/images

输出:
  output/camera_calib/camera_intrinsics.npz
  output/camera_calib/camera_intrinsics.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
DEFAULT_OUT = _HERE.parent / "output" / "camera_calib"


def camera_source(value: object):
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


def open_capture(source, width: int, height: int, fps: int) -> cv2.VideoCapture:
    src = camera_source(source)
    if isinstance(src, int) and os.name == "nt":
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    elif isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开相机: {src!r}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # 预热几帧
    for _ in range(8):
        cap.read()
        time.sleep(0.02)
    return cap


def build_object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_mm)
    return objp


def find_corners(
    gray: np.ndarray, pattern: Tuple[int, int]
) -> Tuple[bool, Optional[np.ndarray]]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def calibrate(
    obj_points: List[np.ndarray],
    img_points: List[np.ndarray],
    image_size: Tuple[int, int],
) -> Tuple[float, np.ndarray, np.ndarray]:
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    return float(rms), K, dist


def save_intrinsics(
    out_dir: Path,
    K: np.ndarray,
    dist: np.ndarray,
    rms: float,
    image_size: Tuple[int, int],
    pattern: Tuple[int, int],
    square_mm: float,
    camera: str,
    n_images: int,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_path = out_dir / "camera_intrinsics.npz"
    yaml_path = out_dir / "camera_intrinsics.yaml"
    stamped_npz = out_dir / f"camera_intrinsics_{stamp}.npz"

    payload = {
        "camera": camera,
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "pattern_cols": int(pattern[0]),
        "pattern_rows": int(pattern[1]),
        "square_mm": float(square_mm),
        "n_images": int(n_images),
        "rms_px": float(rms),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "created_at": stamp,
    }

    np.savez(
        npz_path,
        K=K,
        dist=dist,
        rms=np.array([rms]),
        image_size=np.array(image_size),
        pattern=np.array(pattern),
        square_mm=np.array([square_mm]),
    )
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    yaml_path.write_text(text, encoding="utf-8")
    stamped_npz.write_text(text, encoding="utf-8")
    (out_dir / f"camera_intrinsics_{stamp}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return npz_path, yaml_path


def resolve_path(path: Path, *, default_under_tta: bool = True) -> Path:
    """相对路径优先相对仓库 tta/，再试 cwd / arm/。"""
    if path.is_absolute():
        return path
    candidates = []
    if default_under_tta:
        candidates.append(_HERE.parent / path)  # ~/tta/<path>
        candidates.append(DEFAULT_OUT / "images")
    candidates.append(Path.cwd() / path)
    candidates.append(_HERE / path)
    for cand in candidates:
        if not cand.is_dir():
            continue
        if list_images(cand):
            return cand.resolve()
    # 都没有图时仍返回约定默认目录，便于报错提示
    if default_under_tta:
        return (_HERE.parent / path).resolve()
    return (Path.cwd() / path).resolve()


def list_images(images_dir: Path) -> List[Path]:
    return sorted(
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.JPG"))
        + list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpeg"))
        + list(images_dir.glob("*.JPEG"))
    )


def run_from_dir(
    images_dir: Path,
    pattern: Tuple[int, int],
    square_mm: float,
    out_dir: Path,
    camera_label: str,
) -> int:
    paths = list_images(images_dir)
    if not paths:
        default_images = DEFAULT_OUT / "images"
        print(f"[error] 目录无图片: {images_dir}")
        print(
            "说明: 交互采集默认把图存到:\n"
            f"  {default_images}\n"
            "请先带屏采集:\n"
            "  python3 calibrate_camera.py --camera /dev/video11\n"
            "  # 空格存图，至少约 12 张后再按 c；或:\n"
            f"  python3 calibrate_camera.py --from-dir {default_images}"
        )
        return 2

    objp = build_object_points(pattern[0], pattern[1], square_mm)
    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    image_size: Optional[Tuple[int, int]] = None

    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[skip] 读失败 {path.name}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        ok, corners = find_corners(gray, pattern)
        if not ok:
            print(f"[skip] 未检出棋盘 {path.name}")
            continue
        obj_points.append(objp.copy())
        img_points.append(corners)
        print(f"[ok] {path.name}")

    if len(obj_points) < 5:
        print(f"[error] 有效图片过少: {len(obj_points)}（建议 ≥10）")
        return 1

    assert image_size is not None
    rms, K, dist = calibrate(obj_points, img_points, image_size)
    npz_path, yaml_path = save_intrinsics(
        out_dir, K, dist, rms, image_size, pattern, square_mm, camera_label, len(obj_points)
    )
    print(f"\nRMS = {rms:.4f} px  (图像数={len(obj_points)})")
    print(f"K =\n{K}")
    print(f"dist = {dist.ravel()}")
    print(f"saved: {npz_path}")
    print(f"saved: {yaml_path}")
    if rms > 1.0:
        print("[warn] RMS>1px，建议重拍：棋盘更平、覆盖更多角度/距离")
    return 0


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


def _try_save_sample(
    frame: np.ndarray,
    corners: np.ndarray,
    objp: np.ndarray,
    images_dir: Path,
    obj_points: List[np.ndarray],
    img_points: List[np.ndarray],
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = images_dir / f"calib_{stamp}.jpg"
    cv2.imwrite(str(path), frame)
    obj_points.append(objp.copy())
    img_points.append(corners)
    print(f"[saved] {path.name}  total={len(obj_points)}")
    return path


def _finish_calibrate(
    obj_points: List[np.ndarray],
    img_points: List[np.ndarray],
    image_size: Tuple[int, int],
    out_dir: Path,
    pattern: Tuple[int, int],
    square_mm: float,
    camera: str,
) -> Tuple[float, np.ndarray, np.ndarray, Path, Path]:
    rms, K, dist = calibrate(obj_points, img_points, image_size)
    npz_path, yaml_path = save_intrinsics(
        out_dir, K, dist, rms, image_size, pattern, square_mm, camera, len(obj_points)
    )
    print("\n===== 标定完成 =====")
    print(f"RMS = {rms:.4f} px  n={len(obj_points)}")
    print(f"fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} cx={K[0, 2]:.2f} cy={K[1, 2]:.2f}")
    print(f"dist = {dist.ravel()}")
    print(f"saved: {npz_path}")
    print(f"saved: {yaml_path}")
    if rms > 1.0:
        print("[warn] RMS>1px，建议变换角度重拍或检查方格边长")
    return rms, K, dist, npz_path, yaml_path


def run_headless(
    camera: str,
    width: int,
    height: int,
    fps: int,
    pattern: Tuple[int, int],
    square_mm: float,
    out_dir: Path,
    min_samples: int,
    auto_interval_s: float,
) -> int:
    """无 GUI：检测到棋盘自动存图，凑够后自动标定；终端可手动干预。"""
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        cap = open_capture(camera, width, height, fps)
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 2

    objp = build_object_points(pattern[0], pattern[1], square_mm)
    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    image_size = (width, height)
    last_save_t = 0.0
    last_status = ""

    print(
        f"[headless] 相机={camera} {width}x{height} 棋盘={pattern[0]}x{pattern[1]} "
        f"方格={square_mm}mm"
    )
    print(f"[headless] 自动存图间隔 ≥{auto_interval_s:.1f}s；目标 {min_samples} 张")
    print("[headless] 请用手移动棋盘（远近/倾斜/左右），让相机看到完整棋盘")
    print("终端命令: Enter=强制尝试存图  c=立刻标定  q=退出")
    print(f"图片目录: {images_dir}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            image_size = (frame.shape[1], frame.shape[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_corners(gray, pattern)
            now = time.monotonic()

            status = (
                f"FOUND saved={len(obj_points)}/{min_samples}"
                if found
                else f"no-board saved={len(obj_points)}/{min_samples}"
            )
            if status != last_status:
                print(f"[{time.strftime('%H:%M:%S')}] {status}")
                last_status = status

            if found and corners is not None and (now - last_save_t) >= auto_interval_s:
                _try_save_sample(frame, corners, objp, images_dir, obj_points, img_points)
                last_save_t = now
                # 存图后稍等，方便换姿态
                time.sleep(0.15)

            if len(obj_points) >= min_samples:
                _finish_calibrate(
                    obj_points, img_points, image_size, out_dir, pattern, square_mm, camera
                )
                return 0

            cmd = _stdin_line(0.05)
            if cmd is None:
                continue
            if cmd in {"q", "quit", "exit"}:
                print("退出（未完成标定）")
                return 1
            if cmd in {"c", "calib", "calibrate"}:
                if len(obj_points) < 5:
                    print(f"[info] 有效图过少: {len(obj_points)}，至少 5 张")
                    continue
                _finish_calibrate(
                    obj_points, img_points, image_size, out_dir, pattern, square_mm, camera
                )
                return 0
            # Enter 或其它键：强制尝试存当前帧
            if found and corners is not None:
                _try_save_sample(frame, corners, objp, images_dir, obj_points, img_points)
                last_save_t = time.monotonic()
            else:
                print("[skip] 当前帧未检测到完整棋盘")
    finally:
        cap.release()

    return 1


def run_interactive(
    camera: str,
    width: int,
    height: int,
    fps: int,
    pattern: Tuple[int, int],
    square_mm: float,
    out_dir: Path,
    min_samples: int,
    *,
    force_headless: bool = False,
    auto_interval_s: float = 1.5,
) -> int:
    if force_headless or not os.environ.get("DISPLAY"):
        if force_headless:
            print("[info] --headless：使用无界面采集")
        else:
            print("[info] 未检测到 DISPLAY，自动切换无界面模式")
        return run_headless(
            camera, width, height, fps, pattern, square_mm, out_dir, min_samples, auto_interval_s
        )

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        cap = open_capture(camera, width, height, fps)
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 2

    objp = build_object_points(pattern[0], pattern[1], square_mm)
    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    last_K: Optional[np.ndarray] = None
    last_dist: Optional[np.ndarray] = None
    show_undistort = False
    image_size = (width, height)

    print(
        f"相机={camera} 分辨率={width}x{height} 棋盘内角点={pattern[0]}x{pattern[1]} "
        f"方格={square_mm}mm"
    )
    print("键: [空格]=保存  [c]=标定  [u]=去畸变预览  [q]=退出")
    print(f"图片保存目录: {images_dir}")

    win = "camera_calib"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except cv2.error:
        print("[warn] 无法创建窗口，切换无界面模式 …")
        cap.release()
        return run_headless(
            camera, width, height, fps, pattern, square_mm, out_dir, min_samples, auto_interval_s
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[warn] 读帧失败")
                time.sleep(0.05)
                continue

            image_size = (frame.shape[1], frame.shape[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_corners(gray, pattern)
            vis = frame.copy()
            if found:
                cv2.drawChessboardCorners(vis, pattern, corners, True)
                status = f"FOUND  saved={len(obj_points)}/{min_samples}"
                color = (40, 220, 40)
            else:
                status = f"no board  saved={len(obj_points)}/{min_samples}"
                color = (40, 40, 220)

            if show_undistort and last_K is not None and last_dist is not None:
                vis = cv2.undistort(frame, last_K, last_dist)
                status = "UNDISTORT preview | " + status
                color = (220, 180, 40)

            cv2.putText(vis, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(
                vis,
                "SPACE save | c calib | u undistort | q quit",
                (16, vis.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                1,
            )
            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("退出")
                break

            if key == ord("u"):
                if last_K is None:
                    print("[info] 尚无标定结果，先按 c")
                else:
                    show_undistort = not show_undistort
                continue

            if key == ord(" "):
                if not found or corners is None:
                    print("[skip] 当前帧未检测到完整棋盘")
                    continue
                _try_save_sample(frame, corners, objp, images_dir, obj_points, img_points)
                continue

            if key == ord("c"):
                if len(obj_points) < min_samples:
                    print(f"[info] 至少需要 {min_samples} 张，当前 {len(obj_points)}")
                    continue
                rms, K, dist, _, _ = _finish_calibrate(
                    obj_points, img_points, image_size, out_dir, pattern, square_mm, camera
                )
                last_K, last_dist = K, dist
                continue
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenCV 棋盘格相机内参标定")
    parser.add_argument("--camera", default="/dev/video11", help="相机索引或路径")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--pattern",
        default="8x6",
        help="棋盘内角点数 colsxrows，例如 9x6（不是格子数）",
    )
    parser.add_argument("--square-mm", type=float, default=24.0, help="单格边长 mm")
    parser.add_argument("--min-samples", type=int, default=12, help="最少采集张数")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="输出目录（含 images/ 与 intrinsics）",
    )
    parser.add_argument(
        "--from-dir",
        type=Path,
        default=None,
        help="从已有图片目录标定（跳过实时采集）",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无 GUI：检测到棋盘自动存图并标定（适合 SSH）",
    )
    parser.add_argument(
        "--auto-interval",
        type=float,
        default=1.5,
        help="无界面模式下两次自动存图的最小间隔秒数",
    )
    args = parser.parse_args()

    try:
        cols_s, rows_s = args.pattern.lower().split("x")
        pattern = (int(cols_s), int(rows_s))
    except ValueError:
        print("[error] --pattern 格式应为 9x6")
        return 2

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        # 与其它脚本一致：相对路径落在 tta/ 下，而不是 arm/
        out_dir = (_HERE.parent / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()

    if args.from_dir is not None:
        raw = args.from_dir
        if raw.is_absolute():
            images_dir = raw
        else:
            # 常见写法 output/camera_calib/images → ~/tta/output/...
            images_dir = resolve_path(raw)
        print(f"[from-dir] {images_dir}")
        return run_from_dir(
            images_dir, pattern, float(args.square_mm), out_dir, str(args.camera)
        )

    return run_interactive(
        camera=str(args.camera),
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        pattern=pattern,
        square_mm=float(args.square_mm),
        out_dir=out_dir,
        min_samples=int(args.min_samples),
        force_headless=bool(args.headless),
        auto_interval_s=float(args.auto_interval),
    )


if __name__ == "__main__":
    raise SystemExit(main())