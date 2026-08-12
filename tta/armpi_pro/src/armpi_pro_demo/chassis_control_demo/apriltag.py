import argparse
import sys
import math
from pathlib import Path
import numpy as np

def _load_image(path_str):
    import cv2
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(str(p))
    img = cv2.imread(str(p))
    if img is None:
        raise RuntimeError("failed to read image")
    return img

class Result:
    def __init__(self):
        self.center_x = 0
        self.center_y = 0
        self.angle = 0
        self.data = None

def _map(v, a1, a2, b1, b2):
    if a2 == a1:
        return b1
    return (v - a1) * (b2 - b1) / (a2 - a1) + b1

def detect_and_annotate(img, size_m=(320, 240)):
    import cv2
    msg = Result()
    img_h, img_w = img.shape[:2]
    frame_resize = cv2.resize(img, size_m, interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
    dets = []
    used_backend = None
    try:
        from pupil_apriltags import Detector
        det = Detector(families="tag25h9")
        res = det.detect(gray)
        used_backend = "pupil_apriltags"
        for d in res:
            dets.append({
                "id": int(d.tag_id),
                "center": [float(d.center[0]), float(d.center[1])],
                "corners": [[float(c[0]), float(c[1])] for c in d.corners],
            })
    except Exception:
        dets = []
    publish_en = False
    id_smallest = None
    for d in dets:
        tag_id = d["id"]
        corners = np.rint(np.array(d["corners"], dtype=np.float32))
        for i in range(4):
            corners[i][0] = int(_map(corners[i][0], 0, size_m[0], 0, img_w))
            corners[i][1] = int(_map(corners[i][1], 0, size_m[1], 0, img_h))
        cv2.drawContours(img, [np.array(corners, np.int32)], -1, (0, 255, 255), 10)
        object_center_x = int(_map(d["center"][0], 0, size_m[0], 0, img_w))
        object_center_y = int(_map(d["center"][1], 0, size_m[1], 0, img_h))
        object_angle = int(math.degrees(math.atan2(corners[0][1] - corners[1][1], corners[0][0] - corners[1][0])))
        import cv2 as _cv
        fs = max(1.6, min(3.2, img_w / 600.0))
        _cv.putText(img, str(tag_id), (object_center_x - 10, object_center_y + 10), _cv.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), 5)
        if id_smallest is None or tag_id <= id_smallest:
            id_smallest = tag_id
            msg.center_x = object_center_x
            msg.center_y = object_center_y
            msg.angle = object_angle
            msg.data = id_smallest
    if id_smallest is not None:
        publish_en = True
    id_smallest = None
    return img, {
        "backend": used_backend,
        "publish_en": publish_en,
        "msg": {
            "center_x": msg.center_x,
            "center_y": msg.center_y,
            "angle": msg.angle,
            "data": msg.data,
        },
    }

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", default=None)
    p.add_argument("--resize", default="320x240")
    p.add_argument("--camera", action="store_true")
    p.add_argument("--ros", action="store_true")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()

def _bgr_to_ros_image(img_bgr):
    import numpy as _np
    from sensor_msgs.msg import Image as _Image
    img_rgb = img_bgr[:, :, ::-1]
    h, w = img_rgb.shape[:2]
    msg = _Image()
    msg.height = h
    msg.width = w
    msg.encoding = "rgb8"
    msg.step = w * 3
    msg.data = _np.ascontiguousarray(img_rgb).tobytes()
    return msg

def run_ros(args):
    import rospy
    from sensor_msgs.msg import Image
    import time
    last_result_holder = {"img": None}
    def cb(ros_image):
        import numpy as _np
        import cv2
        t0 = time.perf_counter()
        image = _np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=_np.uint8, buffer=ros_image.data)
        cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            w_h = args.resize.lower().split("x")
            size_m = (int(w_h[0]), int(w_h[1]))
        except Exception:
            size_m = (320, 240)
        result_img, meta = detect_and_annotate(cv2_img, size_m=size_m)
        last_result_holder["img"] = result_img
        t1 = time.perf_counter()
        if args.show:
            cv2.imshow("annotated", result_img)
            cv2.waitKey(1)
        rgb_image = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB).tobytes()
        ros_image.data = rgb_image
        image_pub.publish(ros_image)
        print("处理时间:", f"{(t1 - t0) * 1000.0:.2f} ms")
    rospy.init_node("apriltag_roc", anonymous=True)
    image_pub = rospy.Publisher("/visual_processing/image_result", Image, queue_size=1)
    rospy.Subscriber("/usb_cam/image_raw", Image, cb)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    if args.output and last_result_holder["img"] is not None:
        import cv2
        cv2.imwrite(args.output, last_result_holder["img"])
        print("已保存标注图片到:", args.output)

def run_ros_camera(args):
    import rospy
    from sensor_msgs.msg import Image
    import cv2
    import time
    rospy.init_node("apriltag_roc_cam", anonymous=True)
    pub = rospy.Publisher("/visual_processing/image_result", Image, queue_size=1)
    try:
        w_h = args.resize.lower().split("x")
        size_m = (int(w_h[0]), int(w_h[1]))
    except Exception:
        size_m = (320, 240)
    cap = cv2.VideoCapture(args.camera_index)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    rate = rospy.Rate(max(1, args.fps))
    try:
        while not rospy.is_shutdown():
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(args.camera_index)
                if args.width:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                if args.height:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                time.sleep(0.05)
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            annotated, meta = detect_and_annotate(frame, size_m=size_m)
            msg = _bgr_to_ros_image(annotated)
            pub.publish(msg)
            if args.show:
                cv2.imshow("annotated", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            rate.sleep()
    except KeyboardInterrupt:
        pass
    cap.release()
    cv2.destroyAllWindows()

def main():
    args = parse_args()
    image_path = args.image
    if args.ros or (not args.camera and not image_path):
        run_ros(args)
        return
    import cv2
    try:
        w_h = args.resize.lower().split("x")
        size_m = (int(w_h[0]), int(w_h[1]))
    except Exception:
        size_m = (320, 240)
    if args.camera:
        cap = cv2.VideoCapture(args.camera_index)
        if args.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            print("无法打开摄像头:", args.camera_index)
            sys.exit(1)
        last = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, meta = detect_and_annotate(frame, size_m=size_m)
            last = annotated
            if args.show:
                cv2.imshow("annotated", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()
        if args.output and last is not None:
            cv2.imwrite(args.output, last)
        return
    if image_path:
        img = _load_image(image_path)
        annotated, meta = detect_and_annotate(img, size_m=size_m)
        if args.output:
            cv2.imwrite(args.output, annotated)
        else:
            cv2.imwrite("apriltag_roc_output.jpg", annotated)
        print(meta)
        return

if __name__ == "__main__":
    main()

