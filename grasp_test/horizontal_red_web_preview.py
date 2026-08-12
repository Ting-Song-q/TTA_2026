 #!/usr/bin/env python3
# coding=UTF-8

"""Browser MJPEG preview of horizontal red detection.

Open on the robot LAN:
  http://<robot-ip>:8080/

Example:
  python3 horizontal_red_web_preview.py
  python3 horizontal_red_web_preview.py --port 8080 --no-move
"""

from __future__ import print_function

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import rospy

import horizontal_red_grasp_test as grasp


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class FrameHub(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None
        self._meta = "waiting for first frame"
        self._seq = 0

    def set_frame(self, bgr, meta=""):
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
            self._meta = meta
            self._seq += 1

    def get_jpeg(self):
        with self._lock:
            return self._jpeg, self._meta, self._seq


def _desired_center(config, frame):
    h, w = frame.shape[:2]
    grasp_cfg = config["vision"]["grasp"]
    backend = str(grasp_cfg.get("backend", "joint_trajectory")).lower()
    if backend == "factory_ik":
        factory = grasp_cfg.get("factory_ik", {})
        return float(factory.get("x_center", w / 2.0)), float(factory.get("y_center", h / 2.0))
    offset_u, offset_v = grasp_cfg["align"]["center_offset"]
    return w / 2.0 + float(offset_u), h / 2.0 + float(offset_v)


def _capture_loop(camera, detector, config, hub, rate_hz):
    period = 1.0 / max(1.0, float(rate_hz))
    while not rospy.is_shutdown():
        started = time.time()
        try:
            frame = camera.get_frame(discard=0)
        except Exception as exc:
            hub.set_frame(
                _placeholder("camera error: {}".format(exc)),
                meta="camera error",
            )
            rospy.sleep(0.2)
            continue

        if frame is None:
            hub.set_frame(_placeholder("no camera frame"), meta="no frame")
            rospy.sleep(0.1)
            continue

        detection = detector.detect(frame)
        desired = _desired_center(config, frame)
        visual = detector.draw_debug(frame, detection, desired)
        if detection is None:
            meta = "no target"
            cv2.putText(
                visual,
                "WEB PREVIEW | no red target",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )
        else:
            meta = "u={:.0f} v={:.0f} area={:.0f}".format(
                detection.center_u, detection.center_v, detection.area
            )
            cv2.putText(
                visual,
                "WEB PREVIEW | {}".format(meta),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (50, 220, 50),
                2,
            )
        hub.set_frame(visual, meta=meta)

        elapsed = time.time() - started
        remain = period - elapsed
        if remain > 0:
            rospy.sleep(remain)


def _placeholder(text, size=(640, 480)):
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.putText(img, text, (30, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img


def _make_handler(hub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            rospy.logdebug("http: " + fmt, *args)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body = (
                    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                    "<title>Red Grasp Preview</title>"
                    "<style>"
                    "body{margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center;}"
                    "img{max-width:100%;height:auto;background:#000;}"
                    ".bar{padding:10px;font-size:14px;}"
                    "</style></head><body>"
                    "<div class='bar'>ArmPi red detection preview — stream refreshes automatically</div>"
                    "<img src='/stream' alt='detection stream'/>"
                    "<div class='bar'>ROI / contour / center / desired cross / mask</div>"
                    "</body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/stream":
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last_seq = -1
                try:
                    while not rospy.is_shutdown():
                        jpeg, _meta, seq = hub.get_jpeg()
                        if jpeg is None or seq == last_seq:
                            time.sleep(0.03)
                            continue
                        last_seq = seq
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write("Content-Length: {}\r\n\r\n".format(len(jpeg)).encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
                return

            if path == "/status":
                _jpeg, meta, seq = hub.get_jpeg()
                body = ('{{"seq":{},"meta":"{}"}}'.format(seq, meta.replace('"', "'"))).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_error(404, "Not Found")

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description="Browser preview for horizontal red detection")
    parser.add_argument(
        "--config",
        default=str(grasp.DEFAULT_CONFIG_PATH),
        help="yaml config path",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    parser.add_argument("--rate", type=float, default=8.0, help="preview FPS")
    parser.add_argument(
        "--no-move",
        action="store_true",
        help="do not move arm to observe pose; only stream camera + detection",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rospy.init_node("horizontal_red_web_preview", anonymous=True, log_level=rospy.INFO)

    config = grasp.load_config(args.config)
    # Preview needs fresh frames quickly; do not discard a long settle burst each tick.
    config["vision"]["settle_frames"] = 1
    config["vision"]["save_debug"] = False

    if not args.no_move:
        try:
            runner = grasp.HorizontalGraspRunner(config)
            runner.startup_open_gripper(dry_run=False)
            runner.move_arm_to_observe_pose(dry_run=False)
            rospy.sleep(0.5)
        except Exception as exc:
            rospy.logwarn("move to observe pose failed (%s); continue streaming anyway", exc)

    camera = grasp.CameraCapture(config=config)
    detector = grasp.HorizontalRedDetector(config)
    if not camera.wait_for_stream():
        rospy.logerr("camera stream not ready; check usb_cam topics")
        return 1

    hub = FrameHub()
    hub.set_frame(_placeholder("starting preview..."))

    worker = threading.Thread(
        target=_capture_loop,
        args=(camera, detector, config, hub, args.rate),
        daemon=True,
    )
    worker.start()

    handler = _make_handler(hub)
    host = args.host
    port = int(args.port)
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        if getattr(exc, "errno", None) == 98 or "Address already in use" in str(exc):
            rospy.logerr(
                "port %d already in use. Kill the old preview, or pick another port:\n"
                "  sudo lsof -i :%d\n"
                "  kill <PID>\n"
                "  # or:\n"
                "  python3 horizontal_red_web_preview.py --port %d",
                port,
                port,
                port + 1,
            )
        else:
            rospy.logerr("failed to bind HTTP server on %s:%d: %s", host, port, exc)
        return 1

    rospy.loginfo(
        "web preview ready: http://%s:%d/  (LAN: http://<robot-ip>:%d/)",
        host,
        port,
        port,
    )
    rospy.loginfo("press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        rospy.loginfo("web preview stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
