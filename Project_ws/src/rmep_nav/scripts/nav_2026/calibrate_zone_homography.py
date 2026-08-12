#!/usr/bin/python3
# coding=UTF-8

# BEGIN added: image-to-base_link homography calibration tool
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from config_loader import load_mission_config, load_yaml, save_yaml


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "mission_config.yaml"


def _parse_points(value):
    points = []
    for pair in value.split(";"):
        coordinates = [float(item.strip()) for item in pair.split(",")]
        if len(coordinates) != 2:
            raise argparse.ArgumentTypeError("points must use x,y;x,y format")
        points.append(coordinates)
    if len(points) < 4:
        raise argparse.ArgumentTypeError("at least four points are required")
    return points


def _capture_ros_frame(config, output):
    import rospy

    from vision.camera_capture import CameraCapture

    rospy.init_node("calibrate_zone_homography", anonymous=True)
    camera_yaml = config.get("vision", {}).get("camera_yaml")
    if camera_yaml:
        camera_yaml = str((SCRIPT_DIR / camera_yaml).resolve())
    camera = CameraCapture(config, camera_yaml=camera_yaml)
    frame = camera.get_frame()
    if frame is None:
        raise RuntimeError("camera frame unavailable")
    cv2.imwrite(str(output), frame)
    print("captured:", output)
    return frame


def _select_points(frame, count):
    selected = []
    display = frame.copy()

    def on_mouse(event, x, y, _flags, _data):
        if event != cv2.EVENT_LBUTTONDOWN or len(selected) >= count:
            return
        selected.append([float(x), float(y)])
        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            display,
            str(len(selected)),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    cv2.namedWindow("zone homography calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("zone homography calibration", on_mouse)
    while len(selected) < count:
        cv2.imshow("zone homography calibration", display)
        if cv2.waitKey(20) & 0xFF in (27, ord("q")):
            break
    cv2.destroyAllWindows()
    if len(selected) != count:
        raise RuntimeError("point selection cancelled")
    return selected


def _calibration_path(config):
    filename = config.get("zone_entry", {}).get(
        "calibration_file", "zone_calibration.yaml"
    )
    path = Path(filename)
    return path if path.is_absolute() else SCRIPT_DIR / path


def _compute(image_points, base_points):
    image = np.asarray(image_points, dtype=np.float32)
    base = np.asarray(base_points, dtype=np.float32)
    matrix, _ = cv2.findHomography(image, base, method=0)
    if matrix is None:
        raise RuntimeError("homography calculation failed")
    projected = cv2.perspectiveTransform(
        image.reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    error = math.sqrt(float(np.mean(np.sum((projected - base) ** 2, axis=1))))
    return matrix, error


def _reprojection_error(matrix, image_points, base_points):
    image = np.asarray(image_points, dtype=np.float32)
    base = np.asarray(base_points, dtype=np.float32)
    if len(image) != len(base) or len(image) == 0:
        raise ValueError("validation point counts differ")
    projected = cv2.perspectiveTransform(
        image.reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    return math.sqrt(float(np.mean(np.sum((projected - base) ** 2, axis=1))))


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate image pixels to base_link ground coordinates."
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--capture-output", type=Path, default=Path("/tmp/zone_calibration.jpg"))
    parser.add_argument("--image-points", type=_parse_points)
    parser.add_argument("--base-points", type=_parse_points, required=True)
    parser.add_argument("--validation-image-points", type=_parse_points)
    parser.add_argument("--validation-base-points", type=_parse_points)
    parser.add_argument("--max-error", type=float, default=0.03)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_mission_config(CONFIG_PATH)
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise RuntimeError("cannot read image: %s" % args.image)
    else:
        frame = _capture_ros_frame(config, args.capture_output)

    image_points = args.image_points or _select_points(
        frame, len(args.base_points)
    )
    if len(image_points) != len(args.base_points):
        raise RuntimeError("image and base point counts differ")
    matrix, error = _compute(image_points, args.base_points)
    validation_image_points = args.validation_image_points
    if args.validation_base_points and not validation_image_points:
        print("select independent validation points in the same listed order")
        validation_image_points = _select_points(
            frame, len(args.validation_base_points)
        )
    if validation_image_points and not args.validation_base_points:
        raise RuntimeError("validation base points are required")

    validation_error = None
    if validation_image_points:
        validation_error = _reprojection_error(
            matrix,
            validation_image_points,
            args.validation_base_points,
        )
    elif args.write and not args.force:
        raise RuntimeError(
            "independent validation points are required for --write; "
            "use --force only for temporary testing"
        )

    checked_error = validation_error if validation_error is not None else error
    if checked_error > args.max_error and not args.force:
        raise RuntimeError(
            "reprojection error %.4fm exceeds %.4fm"
            % (checked_error, args.max_error)
        )

    calibration = {
        "zone_entry": {
            "homography": {
                "enabled": True,
                "image_points": image_points,
                "base_points": args.base_points,
                "matrix": matrix.tolist(),
                "fit_error": error,
                "validation_image_points": validation_image_points or [],
                "validation_base_points": args.validation_base_points or [],
                "validation_error": validation_error,
            }
        }
    }
    print(yaml.safe_dump(calibration, allow_unicode=True, sort_keys=False))
    if args.write:
        path = _calibration_path(config)
        overlay = load_yaml(path)
        overlay.setdefault("zone_entry", {})["homography"] = calibration[
            "zone_entry"
        ]["homography"]
        save_yaml(path, overlay)
        print("saved:", path)
    else:
        print("dry run; add --write to save")


if __name__ == "__main__":
    main()
# END added: image-to-base_link homography calibration tool
