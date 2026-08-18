#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 眼在手上：红块像素 → 基座系三维（桌面平面交点）。

依赖:
  - vision_3d.yaml（桌面 Z、方块尺寸）
  - camera_intrinsics.yaml
  - handeye_ee_cam.yaml（T_ee_cam）
  - so101_fk + 当前关节 ticks

用法:
  python3 pixel_to_base.py --yes
  # 循环：检测并打印；Enter 再测一帧，q 退出
  python3 pixel_to_base.py --yes --once
  python3 pixel_to_base.py --yes --u 320 --v 200   # 手动指定像素（不跑检测）
"""

from __future__ import annotations

import argparse
import select
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_TTA = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from calibrate_camera import open_capture  # noqa: E402
from so101_fk import JOINT_ORDER, SO101FK  # noqa: E402
from so101_red_block_camera_test import RedBlockDetector, capture_frame  # noqa: E402
from start import (  # noqa: E402
    DEFAULT_BAUD,
    connect_bus,
    disable_torque_safe,
    enable_torque_safe,
    read_positions,
)

DEFAULT_CONFIG = _HERE / "vision_3d.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"yaml root must be mapping: {path}")
    return data


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    for base in (_TTA, _HERE, Path.cwd()):
        cand = (base / path).resolve()
        if cand.exists():
            return cand
    return (_TTA / path).resolve()


def load_intrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"内参不存在: {path}")
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        return np.asarray(data["K"], dtype=float), np.asarray(data["dist"], dtype=float).reshape(-1)
    raw = load_yaml(path)
    K = np.asarray(raw["camera_matrix"], dtype=float)
    dist = np.asarray(raw.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=float).reshape(-1)
    return K, dist


def load_handeye(path: Path) -> np.ndarray:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"手眼结果不存在: {path}\n请先运行 compute_hand_eye.py"
        )
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        if "T_ee_cam" in data:
            return np.asarray(data["T_ee_cam"], dtype=float)
        return make_T(data["R"], data["t"])
    raw = load_yaml(path)
    if "T_ee_cam" in raw:
        return np.asarray(raw["T_ee_cam"], dtype=float)
    return make_T(raw["rotation_matrix"], raw["translation_m"])


def make_T(R, t) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def grasp_z(scene: dict) -> Tuple[float, float, str]:
    """返回 (交平面 Z, 抓取 Z, 说明)。"""
    table_z = float(scene["table_z_m"])
    size = scene.get("block_size_m", [0.03, 0.03, 0.03])
    h = float(size[2] if len(size) > 2 else size[0])
    mode = str(scene.get("grasp_height_mode", "center")).lower()
    if mode == "top":
        return table_z, table_z + h, "top surface"
    if mode == "table":
        return table_z, table_z, "table plane"
    # center
    return table_z, table_z + 0.5 * h, "block center"


def pixel_ray_cam(u: float, v: float, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """像素 → 相机系单位方向（已去畸变）。"""
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    x, y = float(und[0, 0, 0]), float(und[0, 0, 1])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    d = np.array([(x - cx) / fx, (y - cy) / fy, 1.0], dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-12:
        raise RuntimeError("ray direction degenerate")
    return d / n


def intersect_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_point_z: float,
    plane_normal: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """射线 o+t*d 与平面 n·(x-p)=0；水平桌面时 p=(0,0,z)。"""
    n = np.asarray(plane_normal, dtype=float).reshape(3)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        raise RuntimeError("invalid plane normal")
    n = n / nn
    p0 = np.array([0.0, 0.0, float(plane_point_z)], dtype=float)
    denom = float(np.dot(n, direction))
    if abs(denom) < 1e-9:
        raise RuntimeError("射线与桌面近乎平行，无法求交")
    t = float(np.dot(n, p0 - origin) / denom)
    if t <= 0:
        raise RuntimeError(f"交点在相机后方或无效 (t={t:.4f})，检查手眼/桌面高度/朝向")
    return origin + t * direction, t


def pixel_to_base(
    u: float,
    v: float,
    T_base_ee: np.ndarray,
    T_ee_cam: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    table_z: float,
    table_normal: np.ndarray,
    grasp_z_m: float,
) -> Dict[str, np.ndarray]:
    T_base_cam = T_base_ee @ T_ee_cam
    R = T_base_cam[:3, :3]
    o = T_base_cam[:3, 3]
    d_cam = pixel_ray_cam(u, v, K, dist)
    d_base = R @ d_cam
    d_base = d_base / max(np.linalg.norm(d_base), 1e-12)

    p_table, t_hit = intersect_plane(o, d_base, table_z, table_normal)
    # 同 (X,Y)，抬到抓取高度
    p_grasp = np.array([p_table[0], p_table[1], grasp_z_m], dtype=float)
    return {
        "T_base_cam": T_base_cam,
        "cam_origin_base": o,
        "ray_dir_base": d_base,
        "p_table": p_table,
        "p_grasp": p_grasp,
        "t_hit": np.array([t_hit]),
    }


def anchor_grasp_forward(
    ee_xyz: np.ndarray,
    p_vision: np.ndarray,
    forward_m: float,
    grasp_z_m: float,
    *,
    fallback_dir_xy: Sequence[float] = (0.0, -1.0),
) -> Dict[str, np.ndarray]:
    """用视觉定水平方位，用已知「夹爪前方距离」锚定抓取 XY。

    桌面平面交点在手眼/桌高不准时尺度会漂；观察位若已知木块在夹爪正前方
    数厘米，可用本函数把水平距离压到 forward_m，避免目标飞到工作空间外。
    """
    ee = np.asarray(ee_xyz, dtype=float).reshape(3)
    pv = np.asarray(p_vision, dtype=float).reshape(3)
    d_xy = pv[:2] - ee[:2]
    n = float(np.linalg.norm(d_xy))
    if n < 1e-4:
        d_xy = np.asarray(fallback_dir_xy, dtype=float).reshape(2)
        n = float(np.linalg.norm(d_xy))
        if n < 1e-12:
            d_xy = np.array([0.0, -1.0], dtype=float)
            n = 1.0
    d_xy = d_xy / n
    fwd = float(forward_m)
    p = np.array(
        [ee[0] + fwd * d_xy[0], ee[1] + fwd * d_xy[1], float(grasp_z_m)],
        dtype=float,
    )
    return {
        "p_grasp": p,
        "dir_xy": d_xy,
        "vision_horiz_m": np.array([n]),
        "forward_m": np.array([fwd]),
    }


def _stdin_cmd(timeout_s: float) -> Optional[str]:
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


def run_once(
    *,
    bus,
    fk: SO101FK,
    detector: Optional[RedBlockDetector],
    camera_cfg: dict,
    K: np.ndarray,
    dist: np.ndarray,
    T_ee_cam: np.ndarray,
    table_z: float,
    table_normal: np.ndarray,
    grasp_z_m: float,
    grasp_label: str,
    out_dir: Path,
    u_override: Optional[float],
    v_override: Optional[float],
    keep_cap_open: Optional[cv2.VideoCapture] = None,
) -> int:
    ticks_raw = read_positions(bus)
    ticks = {n: float(ticks_raw[n]) for n in JOINT_ORDER}
    T_base_ee = fk.forward_ticks(ticks)
    ee_xyz, ee_rpy = fk.pose_xyz_rpy(T_base_ee)

    if keep_cap_open is not None:
        ok, frame = keep_cap_open.read()
        if not ok or frame is None:
            print("[error] 读相机失败")
            return 2
        # 丢几帧刷新
        for _ in range(3):
            keep_cap_open.read()
        ok, frame = keep_cap_open.read()
        if not ok or frame is None:
            print("[error] 读相机失败")
            return 2
    else:
        frame = capture_frame(camera_cfg)

    if u_override is not None and v_override is not None:
        u, v = float(u_override), float(v_override)
        det = None
        print(f"[pixel] 手动 u={u:.1f} v={v:.1f}")
    else:
        assert detector is not None
        det = detector.detect(frame)
        if det is None:
            print("[vision] 未检测到红块")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = out_dir / f"miss_{stamp}.jpg"
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame)
            print(f"  已存原图: {path}")
            return 1
        u, v = float(det.center_u), float(det.center_v)
        print(f"[vision] center=({u:.1f},{v:.1f}) area={det.area:.0f}")

    try:
        res = pixel_to_base(
            u, v, T_base_ee, T_ee_cam, K, dist, table_z, table_normal, grasp_z_m
        )
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 2

    p_t = res["p_table"]
    p_g = res["p_grasp"]
    o = res["cam_origin_base"]

    print(f"[fk] ee_xyz=({ee_xyz[0]:.4f},{ee_xyz[1]:.4f},{ee_xyz[2]:.4f}) m")
    print(f"[cam] origin_base=({o[0]:.4f},{o[1]:.4f},{o[2]:.4f}) m")
    print(
        f"[3d] table hit  XYZ=({p_t[0]:.4f},{p_t[1]:.4f},{p_t[2]:.4f}) m  "
        f"(plane Z={table_z:.3f})"
    )
    print(
        f"[3d] grasp ({grasp_label}) XYZ=({p_g[0]:.4f},{p_g[1]:.4f},{p_g[2]:.4f}) m"
    )
    print(
        f"     → mm: ({p_g[0]*1000:.1f}, {p_g[1]*1000:.1f}, {p_g[2]*1000:.1f})"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis = frame.copy()
    if detector is not None:
        vis = detector.draw(vis, det)
    cv2.drawMarker(vis, (int(round(u)), int(round(v))), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    label = f"base mm ({p_g[0]*1000:.0f},{p_g[1]*1000:.0f},{p_g[2]*1000:.0f})"
    cv2.putText(vis, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 50), 2)
    img_path = out_dir / f"p2b_{stamp}.jpg"
    cv2.imwrite(str(img_path), vis)

    np.savez(
        out_dir / f"p2b_{stamp}.npz",
        u=u,
        v=v,
        p_table=p_t,
        p_grasp=p_g,
        T_base_ee=T_base_ee,
        T_base_cam=res["T_base_cam"],
        ticks=np.array([ticks[n] for n in JOINT_ORDER], dtype=float),
    )
    print(f"[saved] {img_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="红块像素 → 基座三维验证")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--once", action="store_true", help="只测一帧后退出")
    parser.add_argument("--teach", action="store_true", help="力矩关闭（可手动摆观察位）")
    parser.add_argument("--port", default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument("--handeye", type=Path, default=None)
    parser.add_argument("--u", type=float, default=None, help="手动像素 u（需同时给 --v）")
    parser.add_argument("--v", type=float, default=None, help="手动像素 v")
    parser.add_argument("--table-z", type=float, default=None, help="覆盖配置中的桌面 Z (m)")
    args = parser.parse_args()

    if not args.yes:
        print("Refusing: pass --yes")
        return 2
    if (args.u is None) ^ (args.v is None):
        print("[error] --u 与 --v 需同时提供")
        return 2

    cfg_path = args.config if args.config.is_absolute() else resolve_path(args.config)
    if not cfg_path.exists():
        # 允许直接 arm/vision_3d.yaml
        cfg_path = (_HERE / "vision_3d.yaml").resolve()
    cfg = load_yaml(cfg_path)
    print(f"[config] {cfg_path}")

    scene = cfg.get("scene") or {}
    table_z = float(args.table_z if args.table_z is not None else scene.get("table_z_m", 0.25))
    table_normal = np.asarray(scene.get("table_normal", [0, 0, 1]), dtype=float)
    _, grasp_z_m, grasp_label = grasp_z({**scene, "table_z_m": table_z})

    block = scene.get("block_size_m", [0.03, 0.03, 0.03])
    print(
        f"[scene] table_z={table_z:.3f} m  block={block} m  "
        f"grasp_z={grasp_z_m:.3f} m ({grasp_label})"
    )

    K, dist = load_intrinsics(Path(args.intrinsics or cfg.get("intrinsics")))
    T_ee_cam = load_handeye(Path(args.handeye or cfg.get("handeye")))
    print(f"[handeye] t_ee_cam (m) = {T_ee_cam[:3, 3]}")

    camera_cfg = dict(cfg.get("camera") or {})
    if args.camera is not None:
        camera_cfg["index_or_path"] = args.camera
    camera_cfg.setdefault("width", 640)
    camera_cfg.setdefault("height", 480)
    camera_cfg.setdefault("fps", 30)
    camera_cfg.setdefault("settle_frames", 8)
    camera_cfg.setdefault("frame_timeout_s", 5.0)

    detector = None
    if args.u is None:
        detector = RedBlockDetector(cfg)

    out_rel = cfg.get("output_directory", "output/pixel_to_base")
    out_dir = Path(out_rel)
    if not out_dir.is_absolute():
        out_dir = _TTA / out_dir

    port = args.port or cfg.get("port", "/dev/ttyACM1")
    baud = int(cfg.get("baud") or DEFAULT_BAUD)
    fk = SO101FK.from_config(cfg) if isinstance(cfg.get("fk"), dict) else SO101FK()

    print(f"连接电机 {port} @ {baud} ...")
    bus = connect_bus(port, baud=baud, configure=False)
    cap = None
    try:
        if args.teach:
            disable_torque_safe(bus)
            print("[torque] OFF")
        else:
            enable_torque_safe(bus)
            print("[torque] ON（保持当前位姿；可用 --teach 手动摆）")
        time.sleep(0.15)

        # 循环模式保持相机打开
        if not args.once:
            cap = open_capture(
                camera_cfg["index_or_path"],
                int(camera_cfg["width"]),
                int(camera_cfg["height"]),
                int(camera_cfg["fps"]),
            )

        print("命令: Enter=再测一帧  q=退出" if not args.once else "单次测量")
        while True:
            code = run_once(
                bus=bus,
                fk=fk,
                detector=detector,
                camera_cfg=camera_cfg,
                K=K,
                dist=dist,
                T_ee_cam=T_ee_cam,
                table_z=table_z,
                table_normal=table_normal,
                grasp_z_m=grasp_z_m,
                grasp_label=grasp_label,
                out_dir=out_dir,
                u_override=args.u,
                v_override=args.v,
                keep_cap_open=cap,
            )
            if args.once:
                return code
            cmd = _stdin_cmd(0.2)
            # 非阻塞等待命令；也允许空转直到 Enter
            while cmd is None:
                cmd = _stdin_cmd(0.3)
            if cmd in {"q", "quit", "exit"}:
                print("退出")
                return 0
            # Enter 或其他键继续
    finally:
        if cap is not None:
            cap.release()
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass
        print("已断开串口")


if __name__ == "__main__":
    raise SystemExit(main())
