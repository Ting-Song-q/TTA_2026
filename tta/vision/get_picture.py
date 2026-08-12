#!/usr/bin/env python3
"""Windows 下用 OpenCV 对 USB Camera 拍照。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2


def open_camera(index: int) -> cv2.VideoCapture:
    # DirectShow 在 Windows 上更稳
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap


def list_cameras(max_index: int = 5) -> None:
    print("可用摄像头索引：")
    found = False
    for i in range(max_index):
        cap = open_camera(i)
        if not cap.isOpened():
            continue
        ok, _ = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  index={i}: {w}x{h}, read={'OK' if ok else 'FAIL'}")
        found = True
        cap.release()
    if not found:
        print("  未找到可用摄像头")


def capture_photo(index: int, output: Path, settle: int = 8) -> Path:
    cap = open_camera(index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 index={index}")

    # 丢弃前几帧，等自动曝光稳定
    frame = None
    for _ in range(settle):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"摄像头 index={index} 读帧失败")

    cap.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"保存失败: {output}")
    return output


def preview(index: int) -> None:
    cap = open_camera(index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 index={index}")

    print("预览中：按 s 拍照并退出，按 q/Esc 退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow(f"camera {index}", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            out = Path("captures") / f"usb_cam_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), frame)
            print(f"已保存: {out.resolve()}")
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="USB Camera 拍照")
    parser.add_argument("--list", action="store_true", help="列出可用摄像头索引")
    parser.add_argument("--preview", action="store_true", help="打开预览窗口")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="摄像头索引（你机器上 0/1 可用；默认 1，若不对可改成 0）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="D:/code/tta/tta/vision/1.jpg",
        help="输出图片路径，默认 captures/usb_cam_时间戳.jpg",
    )
    args = parser.parse_args()

    if args.list:
        list_cameras()
        return

    if args.preview:
        preview(args.index)
        return

    output = args.output or Path("captures") / f"usb_cam_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    path = capture_photo(args.index, output)
    print(f"拍照成功: {path.resolve()}")


if __name__ == "__main__":
    main()
