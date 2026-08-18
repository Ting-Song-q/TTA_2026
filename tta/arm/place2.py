#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 红色镂空框放置（衔接 reset_and_grasp 抓取结束状态）。

检测逻辑移植自 grasp_test/grasp_vision/test2.py 的 RedPlacementFrameDetector；
运动/伺服栈复用 place_apriltag.py（持块、开环前伸、实时视觉伺服、下降开爪）。

流程：
  1. 放置观察位持块
  2. 侦察红框 → 沿相机光轴开环前伸
  3. 实时视觉伺服对准红框中心（腕部锁观察位朝前；必须连续到位；超时则自动开爪）
  4. （可选）最终前探
  5. 开爪前切到与地面平行位姿（poses.place_parallel）→ 开爪

用法:
  python3 reset_and_grasp.py --yes
  python3 place2.py --yes
  python3 place2.py --yes --no-live-preview
  python3 place2.py --yes --no-fine-align
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_TTA = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from grasp_3d import full_pose_from_ik, go_pose_strict, merge_configs  # noqa: E402
from og import JOINTS, TicksArm, clamp_pose, pose_i, print_pose  # noqa: E402
from pixel_to_base import load_handeye  # noqa: E402
from place_apriltag import (  # noqa: E402
    LiveCamera,
    _gui_available,
    _lock_place_joints,
    advance_along_forward_place,
    cam_forward_xy,
    descend_ee_hold_xy,
    ee_xyz_now,
    gripper_tip_uv,
    hold_forever_safe,
    pixel_error_to_approach_xy,
    resolve_hold_gripper,
    set_gripper,
    snapshot_pose,
)
from red_place_frame import (  # noqa: E402
    RedFrameDetection,
    RedPlacementFrameDetector,
    merge_red_frame_cfg,
)
from so101_fk import JOINT_ORDER, SO101FK  # noqa: E402
from so101_ik import SO101IK  # noqa: E402
from start import DEFAULT_BAUD, move_to  # noqa: E402

DEFAULT_RAG = _HERE / "reset_and_grasp.yaml"
DEFAULT_PLACE = _HERE / "place2.yaml"


def _sync_goal_from_actual(
    arm: TicksArm,
    g_hold: float,
    *,
    forward_arm: Optional[Dict[str, float]] = None,
    keep_forward_wrist: bool = False,
    keep_forward_elbow: bool = False,
) -> Dict[str, float]:
    """过载后用实际关节覆盖 goal；可选强制保持朝前腕部。"""
    actual = arm.read()
    synced = {j: float(actual.get(j, 2048)) for j in JOINTS}
    synced["gripper"] = float(g_hold)
    if keep_forward_wrist and forward_arm is not None:
        synced["wrist_flex"] = float(forward_arm["wrist_flex"])
        synced["wrist_roll"] = float(forward_arm["wrist_roll"])
        if keep_forward_elbow:
            synced["elbow_flex"] = float(forward_arm["elbow_flex"])
    arm.goal = dict(synced)
    return synced


def _hold_tick_safe(
    arm: TicksArm,
    g_hold: float,
    *,
    forward_arm: Optional[Dict[str, float]] = None,
) -> None:
    """持姿；任意关节 Overload 则同步实际位姿，不中断伺服。"""
    if not arm.goal:
        return
    try:
        arm.hold_tick()
    except Exception as exc:
        if "Overload" not in str(exc):
            raise
        print(f"[live] hold Overload，改跟实际关节: {exc}")
        try:
            synced = _sync_goal_from_actual(
                arm, g_hold, forward_arm=forward_arm, keep_forward_wrist=True
            )
            print(
                f"[live] goal←actual lift={synced.get('shoulder_lift', 0):.0f} "
                f"wrist_flex={synced.get('wrist_flex', 0):.0f}(朝前锁定)"
            )
        except Exception as exc2:
            print(f"[live] 同步实际位姿失败: {exc2}")


def _ensure_forward_wrist(
    arm: TicksArm,
    forward_arm: Dict[str, float],
    g_hold: float,
    *,
    lock_elbow: bool,
) -> None:
    """伺服前把腕部拉回观察位朝前，避免俯视丢框。"""
    actual = arm.read()
    wf = float(actual.get("wrist_flex", 0))
    target_wf = float(forward_arm["wrist_flex"])
    if abs(wf - target_wf) < 40.0:
        if arm.goal:
            arm.goal["wrist_flex"] = target_wf
            arm.goal["wrist_roll"] = float(forward_arm["wrist_roll"])
            if lock_elbow:
                arm.goal["elbow_flex"] = float(forward_arm["elbow_flex"])
            arm.goal["gripper"] = float(g_hold)
        return
    pose = {j: float(actual.get(j, 2048)) for j in JOINTS}
    pose["wrist_flex"] = target_wf
    pose["wrist_roll"] = float(forward_arm["wrist_roll"])
    if lock_elbow:
        pose["elbow_flex"] = float(forward_arm["elbow_flex"])
    pose["gripper"] = float(g_hold)
    print(
        f"[live] 恢复夹爪朝前: wrist_flex {wf:.0f} → {target_wf:.0f}"
    )
    try:
        go_pose_strict(arm, "forward_wrist", pose)
    except Exception as exc:
        if "Overload" in str(exc):
            print(f"[warn] 恢复朝前过载，仅改 goal: {exc}")
            arm.goal = dict(pose)
        else:
            raise


def _servo_step_xy_forward(
    *,
    arm: TicksArm,
    fk: SO101FK,
    ik: SO101IK,
    forward_arm: Dict[str, float],
    limits: dict,
    g_hold: float,
    delta_xy: np.ndarray,
    z_hold: float,
    ik_tol: float,
    max_horiz_m: float,
    name: str,
    lock_wrist: bool = True,
    lock_elbow: bool = False,
    max_lift_delta_ticks: float = 200.0,
    settle_s: float = 0.10,
) -> Tuple[bool, float]:
    """水平微步：用实际 FK 算位移；解出后锁腕朝前（肘默认不锁，否则易原地踏步）。"""
    ticks = arm.read()
    seed = {n: float(ticks[n]) for n in JOINT_ORDER}
    # 当前位置必须用实际关节，禁止用“虚拟朝前腕”算 cur（否则目标漂、像不动）
    cur = fk.forward_ticks(seed)[:3, 3].copy()
    target = np.array(
        [cur[0] + float(delta_xy[0]), cur[1] + float(delta_xy[1]), float(z_hold)],
        dtype=float,
    )
    max_h = float(max_horiz_m)
    cur_h = float(np.hypot(cur[0], cur[1]))
    tgt_h = float(np.hypot(target[0], target[1]))
    # 已顶到工作半径时：禁止再外扩，只做切向/内收；超限则钳到圆上而非整步跳过
    if tgt_h > max_h > 1e-6:
        if cur_h >= max_h - 1e-4:
            # 当前已在边界：只保留切向 + 内收分量
            radial = np.array([cur[0], cur[1]], dtype=float) / max(cur_h, 1e-9)
            dxy = np.array([float(delta_xy[0]), float(delta_xy[1])], dtype=float)
            d_rad = float(np.dot(dxy, radial))
            d_tan = dxy - d_rad * radial
            if d_rad > 0:
                d_rad = 0.0  # 丢掉外扩
            dxy2 = d_tan + d_rad * radial
            target[0] = cur[0] + float(dxy2[0])
            target[1] = cur[1] + float(dxy2[1])
            tgt_h = float(np.hypot(target[0], target[1]))
            if tgt_h > max_h and tgt_h > 1e-9:
                target[0] *= max_h / tgt_h
                target[1] *= max_h / tgt_h
            print(
                f"[live] 已近半径上限 {max_h*1000:.0f}mm，"
                f"改切向/内收 Δ=({dxy2[0]*1000:+.1f},{dxy2[1]*1000:+.1f})mm"
            )
        else:
            scale = max_h / tgt_h
            target[0] *= scale
            target[1] *= scale
            print(
                f"[live] 目标半径 {tgt_h*1000:.0f}mm→钳制 {max_h*1000:.0f}mm"
            )
    # 钳制后位移过小则跳过
    step_mm = float(np.hypot(target[0] - cur[0], target[1] - cur[1])) * 1000.0
    if step_mm < 0.4:
        print(f"[live] 钳制后步进过小 {step_mm:.1f}mm，跳过")
        return False, 0.0

    # 种子：实际构型，但腕先拨到朝前，利于收敛
    seed_ik = dict(seed)
    if lock_wrist:
        seed_ik["wrist_flex"] = float(forward_arm["wrist_flex"])
        seed_ik["wrist_roll"] = float(forward_arm["wrist_roll"])
    if lock_elbow:
        seed_ik["elbow_flex"] = float(forward_arm["elbow_flex"])

    try:
        ticks0, _info = ik.solve_position(
            target,
            seed_ik,
            joint_limits=limits,
            pos_tol_m=float(ik_tol),
            max_iters=80,
        )
    except Exception as exc:
        print(f"[live] IK 跳过: {exc}")
        return False, 0.0

    # 锁腕参考用“当前 lift/pan + 朝前腕”，肘默认跟 IK，避免抬不起来
    lock_ref = dict(seed)
    lock_ref["wrist_flex"] = float(forward_arm["wrist_flex"])
    lock_ref["wrist_roll"] = float(forward_arm["wrist_roll"])
    if lock_elbow:
        lock_ref["elbow_flex"] = float(forward_arm["elbow_flex"])
    else:
        lock_ref["elbow_flex"] = float(ticks0.get("elbow_flex", seed["elbow_flex"]))

    ticks1 = _lock_place_joints(
        ticks0,
        lock_ref,
        lock_wrist=lock_wrist,
        lock_elbow=lock_elbow,
        max_lift_delta_ticks=max_lift_delta_ticks,
    )
    ticks2, _ = ik.solve_position(
        target,
        ticks1,
        joint_limits=limits,
        pos_tol_m=float(ik_tol),
        max_iters=50,
    )
    ticks2 = _lock_place_joints(
        ticks2,
        lock_ref,
        lock_wrist=lock_wrist,
        lock_elbow=lock_elbow,
        max_lift_delta_ticks=max_lift_delta_ticks,
    )
    pose = full_pose_from_ik(ticks2, g_hold, ticks)
    limited = clamp_pose(pose, limits)
    move_to(arm.bus, pose_i(limited), wait_s=0.0)
    arm.goal = dict(limited)
    arm.current_name = name
    if settle_s > 0:
        time.sleep(float(settle_s))
    _, now = ee_xyz_now(arm, fk)
    moved_mm = float(np.hypot(now[0] - cur[0], now[1] - cur[1])) * 1000.0
    return True, moved_mm


def _bump_joints_before_open(
    arm: TicksArm,
    g_hold: float,
    limits: dict,
    *,
    wrist_flex_delta: float = 0.0,
    shoulder_lift_delta: float = 0.0,
) -> None:
    """开爪前微调：先改 wrist_flex，再改 shoulder_lift（正常/超时共用）。"""

    def _apply_one(joint: str, delta: float, label: str) -> None:
        d = float(delta)
        if abs(d) < 1.0:
            return
        actual = arm.read()
        pose = {j: float(actual.get(j, 2048)) for j in JOINTS}
        if arm.goal:
            for j in JOINTS:
                if j in arm.goal:
                    pose[j] = float(arm.goal[j])
        old_v = float(pose.get(joint, actual.get(joint, 2048)))
        pose[joint] = old_v + d
        pose["gripper"] = float(g_hold)
        limited = clamp_pose(pose, limits)
        print(
            f"[place2] 开爪前{label} {joint} {old_v:.0f}→{limited[joint]:.0f} "
            f"(Δ={d:+.0f})"
        )
        try:
            go_pose_strict(arm, f"pre_open_{joint}", limited)
        except Exception as exc:
            if "Overload" in str(exc):
                print(f"[warn] 开爪前调{joint}过载，仅写 goal: {exc}")
                try:
                    arm.goal = dict(limited)
                    move_to(arm.bus, pose_i(limited), wait_s=0.0)
                except Exception as exc2:
                    print(f"[warn] 开爪前调{joint}写入失败: {exc2}")
            else:
                print(f"[warn] 开爪前调{joint}失败，继续: {exc}")

    # 先腕后肩
    _apply_one("wrist_flex", wrist_flex_delta, "①")
    _apply_one("shoulder_lift", shoulder_lift_delta, "②")


def save_red_debug(
    out_dir: Path,
    frame: np.ndarray,
    detector: RedPlacementFrameDetector,
    det: Optional[RedFrameDetection],
    tu: float,
    tv: float,
    tol_u: float,
    tol_v: float,
    tag: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    vis = detector.draw(frame, det, desired=(tu, tv))
    cv2.rectangle(
        vis,
        (int(tu - tol_u), int(tv - tol_v)),
        (int(tu + tol_u), int(tv + tol_v)),
        (0, 255, 0),
        2,
    )
    cv2.putText(
        vis,
        "tip target",
        (int(tu) + 10, max(16, int(tv) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
    path = out_dir / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    cv2.imwrite(str(path), vis)
    print(f"[saved] {path}")
    return path


def detect_red_and_save(
    *,
    detector: RedPlacementFrameDetector,
    camera: LiveCamera,
    mode: str,
    target_u: Optional[float],
    target_v: Optional[float],
    tol_u: float,
    tol_v: float,
    save_dir: Path,
    no_save: bool,
    tag: str,
    flush: int = 2,
) -> Tuple[Optional[RedFrameDetection], np.ndarray, float, float]:
    for _ in range(max(0, int(flush))):
        try:
            camera.grab()
        except Exception:
            break
    frame = camera.grab()
    tu, tv = gripper_tip_uv(frame.shape, target_u, target_v)
    det = detector.detect(frame, mode=mode)
    if not no_save:
        save_red_debug(save_dir, frame, detector, det, tu, tv, tol_u, tol_v, tag)
    return det, frame, tu, tv


def detect_red_live(
    *,
    detector: RedPlacementFrameDetector,
    camera: LiveCamera,
    target_u: Optional[float],
    target_v: Optional[float],
    tol_u: float,
    tol_v: float,
    save_dir: Path,
    no_save: bool,
    tag: str,
    do_save: bool,
    show: bool,
) -> Tuple[Optional[RedFrameDetection], np.ndarray, float, float, bool]:
    frame = camera.grab_latest() if hasattr(camera, "grab_latest") else camera.grab()
    tu, tv = gripper_tip_uv(frame.shape, target_u, target_v)
    det = detector.detect(frame, mode="track")
    show_ok = True
    if show:
        vis = detector.draw(frame, det, desired=(tu, tv))
        cv2.rectangle(
            vis,
            (int(tu - tol_u), int(tv - tol_v)),
            (int(tu + tol_u), int(tv + tol_v)),
            (0, 255, 0),
            1,
        )
        try:
            cv2.imshow("place2_live", vis)
            cv2.waitKey(1)
        except Exception:
            show_ok = False
    if do_save and not no_save:
        save_red_debug(save_dir, frame, detector, det, tu, tv, tol_u, tol_v, tag)
    return det, frame, tu, tv, show_ok


def open_loop_forward(
    *,
    arm: TicksArm,
    fk: SO101FK,
    ik: SO101IK,
    T_ee_cam: np.ndarray,
    observe_arm: Dict[str, float],
    limits: dict,
    g_hold: float,
    forward_m: float,
    left_m: float,
    route_points: int,
    ik_tol: float,
    max_horiz_m: float,
    max_z_drop_m: float,
    lock_wrist: bool,
    lock_elbow: bool,
    max_lift_delta_ticks: float,
) -> Optional[np.ndarray]:
    """沿相机光轴前伸；返回水平前进单位方向（供最终前探）。"""
    if forward_m <= 1e-6:
        print("[place2] forward_m=0，跳过开环前伸")
        return None
    ticks = arm.read()
    seed = {n: float(ticks[n]) for n in JOINT_ORDER}
    T_base_ee = fk.forward_ticks(seed)
    fwd = cam_forward_xy(T_base_ee, T_ee_cam)
    lat = np.array([-fwd[1], fwd[0]], dtype=float)
    nlat = float(np.linalg.norm(lat))
    if nlat > 1e-9:
        lat /= nlat
    else:
        lat[:] = 0.0
    delta = fwd * float(forward_m) + lat * float(left_m)
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        print("[place2] 开环位移过小，跳过")
        return fwd
    direction = delta / dist
    print(
        f"[place2] 开环前伸 {forward_m*1000:.0f} mm "
        f"+ 左偏 {left_m*1000:.1f} mm → 合 {dist*1000:.0f} mm "
        f"dir=({direction[0]:+.3f},{direction[1]:+.3f})"
    )
    advance_along_forward_place(
        arm=arm,
        fk=fk,
        ik=ik,
        limits=limits,
        g_hold=g_hold,
        observe_arm=observe_arm,
        forward_xy=direction,
        distance_m=dist,
        route_points=route_points,
        ik_tol=ik_tol,
        max_horiz_m=max_horiz_m,
        max_z_drop_m=max_z_drop_m,
        lock_wrist=lock_wrist,
        lock_elbow=lock_elbow,
        max_lift_delta_ticks=max_lift_delta_ticks,
        label="place2_forward",
    )
    return direction


def refine_until_red_aligned(
    *,
    arm: TicksArm,
    fk: SO101FK,
    ik: SO101IK,
    detector: RedPlacementFrameDetector,
    camera: LiveCamera,
    T_ee_cam: np.ndarray,
    limits: dict,
    g_hold: float,
    forward_arm: Dict[str, float],
    target_u: Optional[float],
    target_v: Optional[float],
    tol_u: float,
    tol_v: float,
    m_per_px: float,
    max_step_m: float,
    max_iters: int,
    ik_tol: float,
    max_horiz_m: float,
    save_dir: Path,
    no_save: bool,
    lock_wrist: bool = True,
    lock_elbow: bool = False,
    max_lift_delta_ticks: float = 200.0,
    approach_sign: float = -1.0,
    loop_s: float = 0.05,
    timeout_s: float = 45.0,
    stable_need: int = 3,
    live_preview: bool = False,
    save_every: int = 5,
) -> bool:
    detector.set_mode("track")
    # 对准全程夹爪朝前（观察位腕部），禁止 IK 下垂
    _ensure_forward_wrist(arm, forward_arm, g_hold, lock_elbow=lock_elbow)
    try:
        _sync_goal_from_actual(
            arm, g_hold, forward_arm=forward_arm, keep_forward_wrist=True
        )
    except Exception as exc:
        print(f"[live] 启动同步实际位姿失败: {exc}")
    _, place_xyz = ee_xyz_now(arm, fk)
    z_start = float(place_xyz[2])
    sign = float(approach_sign)
    if abs(sign) < 1e-6:
        sign = -1.0
    sign = 1.0 if sign > 0 else -1.0
    flipped = False
    stable = 0
    step_i = 0
    stall_n = 0
    best_err = 1e9
    deadline = time.monotonic() + float(timeout_s)
    max_cycles = max(1, int(max_iters) * 20)
    if live_preview and not _gui_available():
        print("[live] 当前 OpenCV 无 GUI，自动关闭预览")
        live_preview = False
    print(
        f"[live] 红框实时引导 z_start={z_start:.4f}m "
        f"wrist_flex={forward_arm['wrist_flex']:.0f}(朝前锁定) "
        f"lock_elbow={lock_elbow} lift_Δ≤{max_lift_delta_ticks:.0f} "
        f"tol=({tol_u:.0f},{tol_v:.0f}) max_step={max_step_m*1000:.0f}mm "
        f"loop={loop_s*1000:.0f}ms timeout={timeout_s:.0f}s "
        f"stable_need={stable_need} approach_sign={sign:+.0f}"
    )

    try:
        while time.monotonic() < deadline and step_i < max_cycles:
            _hold_tick_safe(arm, g_hold, forward_arm=forward_arm)
            step_i += 1
            do_save = (step_i % max(1, int(save_every)) == 0) or (step_i <= 2)
            det, frame, tu, tv, show_ok = detect_red_live(
                detector=detector,
                camera=camera,
                target_u=target_u,
                target_v=target_v,
                tol_u=tol_u,
                tol_v=tol_v,
                save_dir=save_dir,
                no_save=no_save,
                tag=f"live{step_i}",
                do_save=do_save,
                show=live_preview,
            )
            if live_preview and not show_ok:
                live_preview = False

            ticks = arm.read()
            seed = {n: float(ticks[n]) for n in JOINT_ORDER}
            # 相机位姿用实际关节，否则光轴方向错、误差映射错
            T_base_ee = fk.forward_ticks(seed)
            cur = T_base_ee[:3, 3].copy()

            if det is None:
                stable = 0
                if step_i % 10 == 1:
                    print(f"[live {step_i}] 未检测到红框，等待...")
                    if step_i % 20 == 1:
                        _ensure_forward_wrist(
                            arm, forward_arm, g_hold, lock_elbow=lock_elbow
                        )
                time.sleep(float(loop_s))
                continue

            du = float(det.center_u) - tu
            dv = float(det.center_v) - tv
            err0 = float(np.hypot(du, dv))
            in_win = abs(du) <= float(tol_u) and abs(dv) <= float(tol_v)
            if step_i % 3 == 1 or in_win:
                state = "窗口内" if in_win else "伺服中"
                print(
                    f"[live {step_i}] ring={det.area:.0f} "
                    f"du={du:+.1f} dv={dv:+.1f} |e|={err0:.1f} "
                    f"{state} sign={sign:+.0f}"
                )

            if in_win:
                stable += 1
                if stable >= int(stable_need):
                    print(f"[live] 连续 {stable} 帧在窗口内 → 对准完成")
                    return True
                time.sleep(float(loop_s))
                continue
            stable = 0

            T_base_cam = T_base_ee @ T_ee_cam
            delta, _lat, _fwd = pixel_error_to_approach_xy(
                du,
                dv,
                T_base_cam,
                m_per_px=m_per_px,
                max_step_m=max_step_m,
                approach_sign=sign,
            )
            n = float(np.linalg.norm(delta[:2]))
            min_step = min(float(max_step_m), 0.003)
            if 1e-6 < n < min_step:
                delta[:2] *= min_step / n

            prev_pose = snapshot_pose(arm)
            prev_pose["wrist_flex"] = float(forward_arm["wrist_flex"])
            prev_pose["wrist_roll"] = float(forward_arm["wrist_roll"])
            if lock_elbow:
                prev_pose["elbow_flex"] = float(forward_arm["elbow_flex"])
            prev_pose["gripper"] = float(g_hold)

            z_cmd = float(cur[2])
            try:
                ok_move, moved_mm = _servo_step_xy_forward(
                    arm=arm,
                    fk=fk,
                    ik=ik,
                    forward_arm=forward_arm,
                    limits=limits,
                    g_hold=g_hold,
                    delta_xy=delta,
                    z_hold=z_cmd,
                    ik_tol=ik_tol,
                    max_horiz_m=max_horiz_m,
                    name=f"live{step_i}",
                    lock_wrist=lock_wrist,
                    lock_elbow=lock_elbow,
                    max_lift_delta_ticks=max_lift_delta_ticks,
                    settle_s=max(0.06, float(loop_s)),
                )
            except Exception as exc:
                if "Overload" not in str(exc):
                    raise
                print(f"[live] 步进 Overload，同步实际后跳过本步: {exc}")
                try:
                    _sync_goal_from_actual(
                        arm, g_hold, forward_arm=forward_arm, keep_forward_wrist=True
                    )
                except Exception:
                    pass
                time.sleep(float(loop_s))
                continue
            if not ok_move:
                stall_n += 1
                if stall_n >= 4:
                    sign *= -1.0
                    stall_n = 0
                    print(
                        f"[live] 连续无法步进 → approach_sign → {sign:+.0f}"
                    )
                time.sleep(float(loop_s))
                continue

            # 步进后强制腕朝前（防止 IK/惯性带偏）
            if lock_wrist and arm.goal:
                arm.goal["wrist_flex"] = float(forward_arm["wrist_flex"])
                arm.goal["wrist_roll"] = float(forward_arm["wrist_roll"])
                arm.goal["gripper"] = float(g_hold)
                try:
                    move_to(arm.bus, pose_i(clamp_pose(arm.goal, limits)), wait_s=0.0)
                except Exception:
                    pass

            det2, _, tu2, tv2, show_ok2 = detect_red_live(
                detector=detector,
                camera=camera,
                target_u=target_u,
                target_v=target_v,
                tol_u=tol_u,
                tol_v=tol_v,
                save_dir=save_dir,
                no_save=True,
                tag=f"live{step_i}chk",
                do_save=False,
                show=live_preview,
            )
            if live_preview and not show_ok2:
                live_preview = False
            if det2 is not None:
                du2 = float(det2.center_u) - tu2
                dv2 = float(det2.center_v) - tv2
                err1 = float(np.hypot(du2, dv2))
                if step_i % 3 == 1:
                    print(
                        f"[live] Δxy=({delta[0]*1000:+.1f},{delta[1]*1000:+.1f})mm "
                        f"moved={moved_mm:.1f}mm |e| {err0:.1f}->{err1:.1f}"
                    )
                improved = err1 < best_err - 1.5
                if improved:
                    best_err = err1
                    stall_n = 0
                else:
                    stall_n += 1

                # 误差明显变差：立刻翻符号并回退
                if err1 > err0 * 1.08 and moved_mm >= 0.5:
                    sign *= -1.0
                    flipped = True
                    stall_n = 0
                    print(f"[live] 误差变差 → approach_sign → {sign:+.0f}")
                    try:
                        go_pose_strict(arm, f"live{step_i}_rb", prev_pose)
                    except Exception as exc:
                        if "Overload" in str(exc):
                            print(f"[live] 回退过载，改跟实际: {exc}")
                            _sync_goal_from_actual(
                                arm,
                                g_hold,
                                forward_arm=forward_arm,
                                keep_forward_wrist=True,
                            )
                        else:
                            raise
                # 卡住（连续多步几乎不改善）：翻符号，允许多次
                elif stall_n >= 4:
                    sign *= -1.0
                    stall_n = 0
                    flipped = True
                    print(
                        f"[live] 误差卡住 |e|≈{err1:.1f} → approach_sign → {sign:+.0f}"
                    )
                elif moved_mm < 0.8 and not flipped:
                    sign *= -1.0
                    flipped = True
                    print(f"[live] 几乎未动 → approach_sign → {sign:+.0f}")
            elif moved_mm < 1.0 and not flipped:
                sign *= -1.0
                flipped = True
                print(f"[live] 丢框/几乎未动 → approach_sign → {sign:+.0f}")

            time.sleep(float(loop_s))
    finally:
        if live_preview:
            try:
                cv2.destroyWindow("place2_live")
            except Exception:
                pass

    print("[live] 超时/达到最大循环仍未对准")
    return False



def main() -> int:
    parser = argparse.ArgumentParser(
        description="观察位持块 → 红框视觉伺服 → 下降开爪"
    )
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--og-config", type=Path, default=DEFAULT_RAG)
    parser.add_argument("--vision-config", type=Path, default=DEFAULT_PLACE)
    parser.add_argument("--port", default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--forward-m", type=float, default=None)
    parser.add_argument("--left-m", type=float, default=None)
    parser.add_argument("--route-points", type=int, default=None)
    parser.add_argument("--final-forward-probe-m", type=float, default=None)
    parser.add_argument("--final-forward-route-points", type=int, default=None)
    parser.add_argument("--place-descend-m", type=float, default=None)
    parser.add_argument("--place-descend-route-points", type=int, default=None)
    parser.add_argument("--skip-observe-move", action="store_true")
    parser.add_argument("--no-fine-align", action="store_true")
    parser.add_argument("--preclose-iters", type=int, default=None)
    parser.add_argument("--preclose-tol-u", type=float, default=None)
    parser.add_argument("--preclose-tol-v", type=float, default=None)
    parser.add_argument("--preclose-target-u", type=float, default=None)
    parser.add_argument("--preclose-target-v", type=float, default=None)
    parser.add_argument("--preclose-m-per-px", type=float, default=None)
    parser.add_argument("--preclose-max-step", type=float, default=None)
    parser.add_argument("--loop-s", type=float, default=None)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--stable-frames", type=int, default=None)
    parser.add_argument("--no-live-preview", action="store_true")
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument(
        "--xy-sign",
        type=float,
        default=-1.0,
        help="光轴前移符号：框在 tip 下方时默认 -1 前移",
    )
    parser.add_argument("--ik-tol", type=float, default=None)
    parser.add_argument("--max-horiz-m", type=float, default=None)
    parser.add_argument("--gripper-open", type=float, default=None)
    parser.add_argument("--gripper-close", type=float, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--no-save-image", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--release-on-exit", action="store_true")
    parser.add_argument(
        "--require-frame",
        action="store_true",
        help="侦察阶段未看到红框则中止（默认仍开环前伸）",
    )
    args = parser.parse_args()

    if not args.yes:
        print("Refusing motion: pass --yes")
        return 2

    cfg = merge_configs(args.og_config, args.vision_config)
    grasp_cfg = dict(cfg.get("grasp") or {})
    poses = dict(cfg.get("poses") or {})
    place_observe = dict(poses.get("place_observe") or {})
    place_parallel = dict(poses.get("place_parallel") or {})
    grasp_observe = dict(poses.get("grasp") or {})
    observe_pose = place_observe or grasp_observe
    observe_pose_src = "poses.place_observe" if place_observe else "poses.grasp"
    if not observe_pose:
        print("[error] 配置缺少 poses.place_observe / poses.grasp")
        return 2

    port = args.port or cfg.get("port", "/dev/ttyACM1")
    baud = int(cfg.get("baud") or DEFAULT_BAUD)
    motion = dict(cfg.get("motion") or {})
    limits = motion.get("joint_limits") or {}

    g_close = float(
        args.gripper_close
        if args.gripper_close is not None
        else grasp_cfg.get(
            "gripper_close",
            observe_pose.get("gripper", grasp_observe.get("gripper", 800)),
        )
    )
    g_open = float(
        args.gripper_open
        if args.gripper_open is not None
        else grasp_cfg.get("gripper_open", 1200)
    )
    observe_full = {j: float(observe_pose.get(j, 2048)) for j in JOINTS}
    observe_hold = dict(observe_full)
    observe_hold["gripper"] = g_close
    observe_arm = {n: float(observe_hold[n]) for n in JOINT_ORDER}
    # 对准阶段始终锁此腕部朝前；勿被前伸后的实际 ticks 覆盖
    forward_arm = dict(observe_arm)

    def _pick(*vals, default=None):
        for v in vals:
            if v is not None:
                return v
        return default

    forward_m = float(_pick(args.forward_m, grasp_cfg.get("forward_m"), default=0.08))
    left_m = float(_pick(args.left_m, grasp_cfg.get("left_m"), default=0.0))
    max_z_drop_m = float(grasp_cfg.get("place_max_z_drop_m", 0.012))
    lock_wrist = bool(grasp_cfg.get("place_lock_wrist", True))
    lock_elbow = bool(grasp_cfg.get("place_lock_elbow", True))
    max_lift_delta_ticks = float(grasp_cfg.get("place_max_lift_delta_ticks", 40))
    route_points = int(_pick(args.route_points, grasp_cfg.get("route_points"), default=4))
    final_probe = float(
        _pick(args.final_forward_probe_m, grasp_cfg.get("final_forward_probe_m"), default=0.0)
    )
    final_route_pts = int(
        _pick(
            args.final_forward_route_points,
            grasp_cfg.get("final_forward_route_points"),
            default=3,
        )
    )
    place_descend_m = float(
        _pick(args.place_descend_m, grasp_cfg.get("place_descend_m"), default=0.03)
    )
    place_descend_pts = int(
        _pick(
            args.place_descend_route_points,
            grasp_cfg.get("place_descend_route_points"),
            default=4,
        )
    )
    pre_open_wf_delta = float(grasp_cfg.get("pre_open_wrist_flex_delta", -500.0))
    pre_open_sl_delta = float(grasp_cfg.get("pre_open_shoulder_lift_delta", 300.0))
    ik_tol = float(_pick(args.ik_tol, grasp_cfg.get("ik_tol_m"), default=0.012))
    max_horiz_m = float(_pick(args.max_horiz_m, grasp_cfg.get("max_horiz_m"), default=0.34))

    target_u = _pick(args.preclose_target_u, grasp_cfg.get("preclose_target_u_px"))
    target_v = _pick(args.preclose_target_v, grasp_cfg.get("preclose_target_v_px"))
    if target_u is not None:
        target_u = float(target_u)
    if target_v is not None:
        target_v = float(target_v)
    tol_u = float(_pick(args.preclose_tol_u, grasp_cfg.get("preclose_tol_u_px"), default=28.0))
    tol_v = float(_pick(args.preclose_tol_v, grasp_cfg.get("preclose_tol_v_px"), default=32.0))
    m_per_px = float(
        _pick(args.preclose_m_per_px, grasp_cfg.get("preclose_m_per_px"), default=0.00018)
    )
    max_step_m = float(
        _pick(args.preclose_max_step, grasp_cfg.get("preclose_max_step_m"), default=0.015)
    )
    preclose_iters = int(
        _pick(args.preclose_iters, grasp_cfg.get("preclose_iters"), default=8)
    )
    loop_s = float(_pick(args.loop_s, grasp_cfg.get("place_live_loop_s"), default=0.05))
    timeout_s = float(
        _pick(args.timeout_s, grasp_cfg.get("place_live_timeout_s"), default=45.0)
    )
    stable_need = int(
        _pick(args.stable_frames, grasp_cfg.get("place_stable_frames"), default=3)
    )
    save_every = int(_pick(args.save_every, grasp_cfg.get("place_live_save_every"), default=5))
    live_preview = not bool(args.no_live_preview) and bool(
        grasp_cfg.get("place_live_preview", False)
    )

    he_path = Path(cfg.get("handeye", "output/handeye_ee_cam.yaml"))
    if not he_path.is_absolute():
        he_path = (_TTA / he_path).resolve()
    try:
        T_ee_cam = load_handeye(he_path)
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return 1

    camera_cfg = dict(cfg.get("camera") or {})
    if args.camera is not None:
        camera_cfg["index_or_path"] = args.camera
    camera_cfg.setdefault("width", 640)
    camera_cfg.setdefault("height", 480)
    camera_cfg.setdefault("fps", 30)
    camera_cfg["settle_frames"] = 2
    camera_cfg.setdefault("frame_timeout_s", 5.0)

    save_dir = args.save_dir or Path("output/place2")
    if not save_dir.is_absolute():
        save_dir = (_TTA / save_dir).resolve()

    red_cfg = merge_red_frame_cfg(cfg.get("red_frame") or {})
    detector = RedPlacementFrameDetector(red_cfg)
    fk = SO101FK.from_config(cfg) if isinstance(cfg.get("fk"), dict) else SO101FK()
    ik = SO101IK(fk)

    print_pose(f"放置观察位持块({observe_pose_src}):", observe_hold)
    tip_desc = (
        f"({target_u:.0f},{target_v:.0f})"
        if target_u is not None and target_v is not None
        else "(0.5w,0.55h)"
    )
    print(
        f"[info] og={args.og_config} vision={args.vision_config} "
        f"observe={observe_pose_src} forward={forward_m*1000:.0f}mm "
        f"descend={place_descend_m*1000:.0f}mm tip={tip_desc} "
        f"tol=({tol_u:.0f},{tol_v:.0f}) preview={live_preview}"
    )

    arm = TicksArm(port, baud, motion)
    camera: Optional[LiveCamera] = None
    try:
        try:
            arm.connect()
        except RuntimeError as exc:
            err = str(exc)
            if "Missing motor" in err or "id: model_number" in err:
                print(
                    f"[error] 舵机握手失败（常见于夹爪 id=6 过载保护掉线）:\n{exc}\n"
                    "请断电重启机械臂电源，确认 1–6 号舵机都在线后再运行。"
                )
                return 1
            raise
        camera = LiveCamera(camera_cfg)

        print(f"\n=== 1) 放置观察位持块（{observe_pose_src}） ===")
        actual0 = arm.read()
        g_now0 = float(actual0.get("gripper", g_close))
        g_hold = resolve_hold_gripper(g_now0, g_close, g_open)
        print(
            f"[place2] 爪值实际={g_now0:.0f} → 持块命令={g_hold:.0f} "
            f"(close={g_close:.0f})"
        )
        if args.skip_observe_move:
            arm.goal = {
                **{j: float(actual0.get(j, observe_full.get(j, 2048))) for j in JOINTS},
                "gripper": g_hold if abs(g_now0 - g_hold) <= 80 else g_now0,
            }
            arm.current_name = "observe_hold"
            g_hold = float(arm.goal["gripper"])
            observe_arm = {n: float(arm.goal[n]) for n in JOINT_ORDER}
        else:
            move_pose = dict(observe_full)
            move_pose["gripper"] = g_hold
            try:
                go_pose_strict(arm, "place_observe_hold", move_pose)
            except Exception as exc:
                if "Overload" in str(exc):
                    print(f"[warn] 回观察位时夹爪过载，改用实际爪值继续: {exc}")
                    actual1 = arm.read()
                    g_hold = resolve_hold_gripper(
                        float(actual1.get("gripper", g_hold)), g_close, g_open
                    )
                    move_pose["gripper"] = g_hold
                    if arm.goal:
                        arm.goal["gripper"] = g_hold
                else:
                    raise
            observe_arm = {n: float(move_pose[n]) for n in JOINT_ORDER}
        try:
            g_hold = float(arm.read().get("gripper", g_hold))
        except Exception:
            pass
        if arm.goal:
            arm.goal["gripper"] = g_hold
        print(f"[place2] 持块 gripper goal={g_hold:.0f}")

        print("\n=== 2) 侦察红框 → 开环前伸 ===")
        detector.set_mode("scout")
        det0, _, _, _ = detect_red_and_save(
            detector=detector,
            camera=camera,
            mode="scout",
            target_u=target_u,
            target_v=target_v,
            tol_u=tol_u,
            tol_v=tol_v,
            save_dir=save_dir,
            no_save=bool(args.no_save_image),
            tag="scout",
            flush=3,
        )
        if det0 is None:
            print("[place2] 侦察未看到红框")
            if args.require_frame:
                hold_forever_safe(arm, motion)
                return 1
            print("[place2] 仍继续开环前伸（可用 --require-frame 强制中止）")
        else:
            print(
                f"[place2] 红框 center=({det0.center_u:.0f},{det0.center_v:.0f}) "
                f"ring={det0.area:.0f} fill={det0.rect_fill:.2f} hole={det0.hole_ratio:.2f}"
            )

        forward_xy = None
        try:
            forward_xy = open_loop_forward(
                arm=arm,
                fk=fk,
                ik=ik,
                T_ee_cam=T_ee_cam,
                observe_arm=forward_arm,
                limits=limits,
                g_hold=g_hold,
                forward_m=forward_m,
                left_m=left_m,
                route_points=route_points,
                ik_tol=ik_tol,
                max_horiz_m=max_horiz_m,
                max_z_drop_m=max_z_drop_m,
                lock_wrist=lock_wrist,
                lock_elbow=lock_elbow,
                max_lift_delta_ticks=max_lift_delta_ticks,
            )
        except RuntimeError as exc:
            print(f"[warn] 开环前伸失败，按当前位置继续: {exc}")
            ticks = arm.read()
            seed = {n: float(ticks[n]) for n in JOINT_ORDER}
            forward_xy = cam_forward_xy(fk.forward_ticks(seed), T_ee_cam)

        # 前伸后同步 goal，但腕部仍锁观察位朝前
        try:
            ticks_now = arm.read()
            _sync_goal_from_actual(
                arm, g_hold, forward_arm=forward_arm, keep_forward_wrist=True
            )
            print(
                f"[place2] 前伸后实际 lift={ticks_now.get('shoulder_lift', 0):.0f} "
                f"wrist→{forward_arm['wrist_flex']:.0f}(朝前锁定)"
            )
        except Exception as exc:
            print(f"[warn] 前伸后同步失败: {exc}")

        do_fine_align = not bool(args.no_fine_align)
        if do_fine_align:
            print("\n=== 3) 实时视觉引导（对准红框中心，夹爪朝前） ===")
            ok = refine_until_red_aligned(
                arm=arm,
                fk=fk,
                ik=ik,
                detector=detector,
                camera=camera,
                T_ee_cam=T_ee_cam,
                limits=limits,
                g_hold=g_hold,
                forward_arm=forward_arm,
                target_u=target_u,
                target_v=target_v,
                tol_u=tol_u,
                tol_v=tol_v,
                m_per_px=m_per_px,
                max_step_m=max_step_m,
                max_iters=preclose_iters,
                ik_tol=ik_tol,
                max_horiz_m=max_horiz_m,
                save_dir=save_dir,
                no_save=bool(args.no_save_image),
                lock_wrist=lock_wrist,
                lock_elbow=lock_elbow,
                max_lift_delta_ticks=max_lift_delta_ticks,
                approach_sign=float(args.xy_sign),
                loop_s=loop_s,
                timeout_s=timeout_s,
                stable_need=stable_need,
                live_preview=live_preview,
                save_every=save_every,
            )
            if not ok:
                print("[place2] 视觉对准超时/未到位 → 自动开爪释放")
                if not args.no_open:
                    try:
                        _bump_joints_before_open(
                            arm,
                            g_hold,
                            limits,
                            wrist_flex_delta=pre_open_wf_delta,
                            shoulder_lift_delta=pre_open_sl_delta,
                        )
                        set_gripper(
                            arm,
                            g_open,
                            limits,
                            float(grasp_cfg.get("close_settle_s", 0.35)),
                        )
                        print(f"[place2] 超时开爪 -> {int(g_open)}")
                    except Exception as exc:
                        print(f"[warn] 超时开爪失败: {exc}")
                else:
                    print("[place2] --no-open，超时仍保持持块")
                hold_forever_safe(arm, motion)
                return 1
        else:
            print(
                "\n=== 3) --no-fine-align，跳过视觉到位判定 "
                "（仍会下降开爪；正式放置请勿使用） ==="
            )

        if final_probe > 1e-4 and forward_xy is not None:
            print(f"\n=== 4) 最终前探 {final_probe*1000:.0f} mm ===")
            try:
                advance_along_forward_place(
                    arm=arm,
                    fk=fk,
                    ik=ik,
                    limits=limits,
                    g_hold=g_hold,
                    observe_arm=forward_arm,
                    forward_xy=forward_xy,
                    distance_m=final_probe,
                    route_points=final_route_pts,
                    ik_tol=ik_tol,
                    max_horiz_m=max_horiz_m,
                    max_z_drop_m=max_z_drop_m,
                    lock_wrist=lock_wrist,
                    lock_elbow=lock_elbow,
                    max_lift_delta_ticks=max_lift_delta_ticks,
                    label="place2_final_probe",
                )
            except RuntimeError as exc:
                print(f"[warn] 最终前探失败，按当前位置继续: {exc}")

        if place_parallel:
            print("\n=== 5) 开爪前切到与地面平行位姿（poses.place_parallel） ===")
            parallel_pose = {j: float(place_parallel.get(j, 2048)) for j in JOINTS}
            # 持块：爪值跟当前持块，不强制 yaml 里可能缺省的 gripper
            parallel_pose["gripper"] = float(g_hold)
            try:
                go_pose_strict(arm, "place_parallel", parallel_pose)
            except Exception as exc:
                if "Overload" in str(exc):
                    print(f"[warn] 平行位姿时夹爪过载，改用实际爪值: {exc}")
                    try:
                        g_hold = float(arm.read().get("gripper", g_hold))
                    except Exception:
                        pass
                    parallel_pose["gripper"] = g_hold
                    if arm.goal:
                        arm.goal["gripper"] = g_hold
                else:
                    print(f"[error] 无法到达平行位姿，中止开爪: {exc}")
                    hold_forever_safe(arm, motion)
                    return 1
            print_pose("平行位姿持块:", parallel_pose)
        elif place_descend_m > 1e-4:
            print(f"\n=== 5) 对准后下降 {place_descend_m*1000:.0f} mm ===")
            ok_down = descend_ee_hold_xy(
                arm=arm,
                fk=fk,
                ik=ik,
                observe_arm=observe_arm,
                limits=limits,
                g_hold=g_hold,
                descend_m=place_descend_m,
                route_points=place_descend_pts,
                ik_tol=ik_tol,
                max_horiz_m=max_horiz_m,
                label="place2_descend",
            )
            if not ok_down:
                print("[warn] 下降失败，仍尝试开爪")
        else:
            print("\n=== 5) 无 place_parallel 且 place_descend_m=0，跳过 ===")

        if camera is not None and not bool(args.no_save_image):
            print("\n=== 6a) 开爪前拍照 ===")
            try:
                detect_red_and_save(
                    detector=detector,
                    camera=camera,
                    mode="track",
                    target_u=target_u,
                    target_v=target_v,
                    tol_u=tol_u,
                    tol_v=tol_v,
                    save_dir=save_dir,
                    no_save=False,
                    tag="before_open",
                    flush=3,
                )
            except Exception as exc:
                print(f"[warn] 开爪前拍照失败: {exc}")

        if not args.no_open:
            print(f"\n=== 6) 开爪放下 -> {int(g_open)} ===")
            try:
                _bump_joints_before_open(
                    arm,
                    g_hold,
                    limits,
                    wrist_flex_delta=pre_open_wf_delta,
                    shoulder_lift_delta=pre_open_sl_delta,
                )
                set_gripper(arm, g_open, limits, float(grasp_cfg.get("close_settle_s", 0.35)))
            except Exception as exc:
                print(f"[warn] 开爪写入失败，请手动确认夹爪: {exc}")
        else:
            print("\n=== 6) --no-open，保持持块 ===")

        hold_forever_safe(arm, motion)
        return 0
    except KeyboardInterrupt:
        print("\n用户中断")
        return 130
    finally:
        if camera is not None:
            camera.close()
        arm.disconnect(release_torque=bool(args.release_on_exit))
        if not args.release_on_exit:
            print("已断开串口；力矩未主动关闭")


if __name__ == "__main__":
    raise SystemExit(main())
