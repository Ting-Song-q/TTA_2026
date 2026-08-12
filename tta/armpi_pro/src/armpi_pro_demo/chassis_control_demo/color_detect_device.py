#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os
import sys
import time
import rospy
from sensor_msgs.msg import Image


select_min_area = 100
select_max_area = None

range_rgb = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
}


hsv_range_list = {
    'red': [
        {'min': [0, 70, 50], 'max': [10, 255, 255]},
        {'min': [170, 70, 50], 'max': [180, 255, 255]},
    ],
    'green': {'min': [35, 43, 46], 'max': [77, 255, 255]},
    'blue': {'min': [90, 30, 30], 'max': [140, 255, 255]},
}


def color_detect(img, color):
    global select_min_area
    global select_max_area

    

    area_max = 0
    area_max_contour = None
    img_copy = img.copy()
    frame_hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)

    if color in hsv_range_list:
        hr = hsv_range_list[color]
        if isinstance(hr, list):
            m = None
            for r in hr:
                tmp = cv2.inRange(frame_hsv, tuple(r['min']), tuple(r['max']))
                m = tmp if m is None else cv2.bitwise_or(m, tmp)
            frame_mask = m
        else:
            frame_mask = cv2.inRange(frame_hsv, tuple(hr['min']), tuple(hr['max']))
    eroded = cv2.erode(frame_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    filtered = []
    for c in contours:
        a = abs(cv2.contourArea(c))
        if a < select_min_area:
            continue
        if select_max_area is not None and a > select_max_area:
            continue
        filtered.append(c)

    if filtered:
        area_max_contour = max(filtered, key=lambda c: abs(cv2.contourArea(c)))
        area_max = abs(cv2.contourArea(area_max_contour))

    if area_max and area_max > select_min_area:
        (centerx, centery), radius = cv2.minEnclosingCircle(area_max_contour)
        cv2.circle(img, (int(centerx), int(centery)), int(radius) + 5, range_rgb[color], 2)

    return img


def color_detect_with_time(img, color):
    t0 = time.perf_counter()
    r = color_detect(img, color)
    t1 = time.perf_counter()
    return r, (t1 - t0) * 1000.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('image', nargs='?', default=None)
    p.add_argument('-c', '--color', default=None, choices=['red', 'green', 'blue'])
    p.add_argument('--camera', action='store_true')
    p.add_argument('--ros', action='store_true')
    p.add_argument('--camera-index', type=int, default=0)
    p.add_argument('--width', type=int, default=None)
    p.add_argument('--height', type=int, default=None)
    p.add_argument('--hsv-min', nargs=3, type=int, default=None)
    p.add_argument('--hsv-max', nargs=3, type=int, default=None)
    p.add_argument('--hsv-min2', nargs=3, type=int, default=None)
    p.add_argument('--hsv-max2', nargs=3, type=int, default=None)
    p.add_argument('-o', '--output', default=None)
    p.add_argument('--show', action='store_true')
    p.add_argument('--min-area', type=int, default=100)
    p.add_argument('--max-area', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    image_path = args.image
    color = args.color
    if not color:
        color = input('输入颜色名称(red/green/blue): ').strip()
        if color not in ['red', 'green', 'blue']:
            print('颜色输入无效，使用red')
            color = 'red'
    globals()['select_min_area'] = args.min_area
    globals()['select_max_area'] = args.max_area

    if args.hsv_min and args.hsv_max:
        if color == 'red' and args.hsv_min2 and args.hsv_max2:
            hsv_range_list[color] = [
                {'min': args.hsv_min, 'max': args.hsv_max},
                {'min': args.hsv_min2, 'max': args.hsv_max2},
            ]
        else:
            hsv_range_list[color] = {'min': args.hsv_min, 'max': args.hsv_max}

    if args.ros or (not args.camera and not image_path):
        last_result_holder = {'img': None}
        def image_callback(ros_image):
            image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
            cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            result_img, elapsed_ms = color_detect_with_time(cv2_img, color)
            last_result_holder['img'] = result_img
            print('处理时间:', f'{elapsed_ms:.2f} ms')
            if args.show:
                cv2.imshow('annotated', result_img)
                cv2.waitKey(1)
            rgb_image = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB).tostring()
            ros_image.data = rgb_image
            image_pub.publish(ros_image)
        rospy.init_node('color_detect_device', anonymous=True)
        image_pub = rospy.Publisher('/visual_processing/image_result', Image, queue_size=1)
        rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()
        if args.output and last_result_holder['img'] is not None:
            cv2.imwrite(args.output, last_result_holder['img'])
            print('已保存标注图片到:', args.output)
        return

    if args.camera:
        cap = cv2.VideoCapture(args.camera_index)
        if args.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            print('无法打开摄像头:', args.camera_index)
            sys.exit(1)
        last_result = None
        while True:
            ret, frame = cap.read()
            if not ret:
                print('摄像头读取失败')
                break
            result_img, elapsed_ms = color_detect_with_time(frame, color)
            last_result = result_img
            print('处理时间:', f'{elapsed_ms:.2f} ms')
            if args.show:
                cv2.imshow('annotated', result_img)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q') or k == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()
        if args.output and last_result is not None:
            cv2.imwrite(args.output, last_result)
            print('已保存标注图片到:', args.output)
    else:
        img = cv2.imread(image_path)
        if img is None:
            print('无法读取图片:', image_path)
            sys.exit(1)
        result_img, elapsed_ms = color_detect_with_time(img, color)
        out_path = args.output
        if not out_path:
            base, ext = os.path.splitext(os.path.basename(image_path))
            out_path = os.path.join(os.path.dirname(image_path), base + '_annotated' + (ext if ext else '.png'))
        cv2.imwrite(out_path, result_img)
        print('已保存标注图片到:', out_path)
        print('处理时间:', f'{elapsed_ms:.2f} ms')
        if args.show:
            cv2.imshow('annotated', result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
#1.apriltag identification
def apriltag_detect:
    return
            
            
            
            
            
            
            
            
            
            
            
            
            #2.path
#3.catch put,combine with 1.,2.

if __name__ == '__main__':
    main()
