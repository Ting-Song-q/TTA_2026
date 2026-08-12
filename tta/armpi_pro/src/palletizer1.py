#!/usr/bin/python3
# coding=utf8
import sys
import cv2
import math
import time
import rospy
import numpy as np
from threading import RLock, Thread

from std_srvs.srv import *
from std_msgs.msg import *
from sensor_msgs.msg import Image
from sensor.msg import Led
from hiwonder_servo_msgs.msg import MultiRawIdPosDur, ServoStateList
from chassis_control.msg import SetVelocity

# 导入项目库
from armpi_pro import PID
from armpi_pro import Misc
from armpi_pro import bus_servo_control
from armpi_pro import apriltag
from kinematics import ik_transform

class SmartPalletizerTask:
    def __init__(self):
        rospy.init_node('smart_palletizer_task', log_level=rospy.INFO)
        
        self.lock = RLock()
        self.ik = ik_transform.ArmIK()
        
        # 舵机状态字典
        self.servo_positions = {}
        
        # 发布者
        self.joints_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
        self.set_velocity = rospy.Publisher('/chassis_control/set_velocity', SetVelocity, queue_size=1)
        self.buzzer_pub = rospy.Publisher('/sensor/buzzer', Float32, queue_size=1)
        self.rgb_pub = rospy.Publisher('/sensor/rgb_led', Led, queue_size=1)
        self.image_pub = rospy.Publisher('/visual_processing/image_result', Image, queue_size=1)
        
        # 参数
        self.size_m = (320, 240)
        self.img_w = 640
        self.img_h = 480
        
        # 检测状态
        self.mode = 'None' 
        self.target_color = 'None'
        self.detect_result = None
        
        # PID控制器
        self.x_pid = PID.PID(P=0.06, I=0, D=0)
        self.y_pid = PID.PID(P=0.00003, I=0, D=0)
        
        # 摄像头订阅者
        self.running = True
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        self.servo_state_sub = rospy.Subscriber('/servo_controllers/port_id_1/servo_states', ServoStateList, self.servo_state_callback)
        
        # 任务变量
        self.centreX = 320
        self.centreY = 410
        self.x_dis = 500
        self.y_dis = 0.15
        
        self.range_rgb = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
        }
        
        rospy.sleep(1.0)
        self.init_robot()
        
        # 启动任务线程
        self.task_thread = Thread(target=self.run_task)
        self.task_thread.setDaemon(True)
        self.task_thread.start()

    def init_robot(self):
        rospy.loginfo("正在初始化机器人...")
        self.stop_chassis()
        self.set_rgb('black')
        self.reset_arm()
    
    def reset_arm(self):
        target = self.ik.setPitchRanges((0, 0.15, 0.10), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1500, (
                (1, 150), (2, 500), 
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, 500)
            ))
        rospy.sleep(1.5)

    def set_rgb(self, color):
        if color not in self.range_rgb:
            r,g,b = 0,0,0
        else:
            r,g,b = self.range_rgb[color][2], self.range_rgb[color][1], self.range_rgb[color][0]
        led = Led()
        led.index = 0
        led.rgb.r = r; led.rgb.g = g; led.rgb.b = b
        self.rgb_pub.publish(led)
        led.index = 1
        self.rgb_pub.publish(led)

    def stop_chassis(self):
        self.set_velocity.publish(0, 90, 0)

    def move_chassis(self, speed, direction, omega):
        self.set_velocity.publish(speed, direction, omega)

    def servo_state_callback(self, msg):
        for state in msg.servo_states:
            self.servo_positions[state.id] = state.position

    # --- 图像处理 ---
    def getAreaMaxContour(self, contours, hierarchy=None, is_black=False):
        contour_area_temp = 0
        contour_area_max = 0
        area_max_contour = None
        
        for i, c in enumerate(contours):
            # 基础面积筛选
            area = math.fabs(cv2.contourArea(c))
            if area < 100: continue # 忽略太小的
            
            # 寻找最大面积
            if area > contour_area_max:
                contour_area_max = area
                area_max_contour = c
                
        return area_max_contour, contour_area_max

    def color_detect(self, img, color):
        if color == 'None':
            return img, None
        img_copy = img.copy()
        img_h, img_w = img.shape[:2]
        frame_resize = cv2.resize(img_copy, self.size_m, interpolation=cv2.INTER_NEAREST)
        frame_hsv = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2HSV)
        result = None
        
        # HSV范围 (包含黑色)
        hsv_ranges = {
            'red': [((0, 43, 46), (10, 255, 255)), ((156, 43, 46), (180, 255, 255))],
            'green': [((35, 43, 46), (77, 255, 255))],
            'blue': [((100, 43, 46), (124, 255, 255))],
            'black': [((0, 0, 0), (180, 255, 65))] # V值恢复到46 (标准参考值)
        }
        
        if color in hsv_ranges:
            mask = np.zeros(frame_hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in hsv_ranges[color]:
                mask |= cv2.inRange(frame_hsv, np.array(lower), np.array(upper))
            
            # 形态学处理
            if color == 'black':
                # ROI 裁剪：屏蔽掉画面底部区域，防止识别到自己的黑色爪子
                # 假设爪子主要出现在画面底部 1/3
                h, w = mask.shape
                roi_h = int(h * 0.7) # 只保留上部 70%
                mask[roi_h:, :] = 0  # 将底部 30% 强制涂黑(忽略)
                
                # 黑色特殊处理：不需要空心检测，直接找最大黑色块
                eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))) 
                dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))) 
                
                # 使用 RETR_EXTERNAL 找外轮廓即可，不再需要父子层级
                contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
                area_max_contour, area_max = self.getAreaMaxContour(contours)
            else:
                # 其他颜色保持原逻辑 (RETR_EXTERNAL 只找外轮廓)
                eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
                dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
                
                contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
                area_max_contour, area_max = self.getAreaMaxContour(contours)
            
            if area_max > 100:
                (centerx, centery), radius = cv2.minEnclosingCircle(area_max_contour)
                cx = int(Misc.map(centerx, 0, self.size_m[0], 0, img_w))
                cy = int(Misc.map(centery, 0, self.size_m[1], 0, img_h))
                
                # 画黄色轮廓
                scale_x = img_w / self.size_m[0]
                scale_y = img_h / self.size_m[1]
                scaled_contour = area_max_contour.copy()
                scaled_contour[:, 0, 0] = area_max_contour[:, 0, 0] * scale_x
                scaled_contour[:, 0, 1] = area_max_contour[:, 0, 1] * scale_y
                scaled_contour = scaled_contour.astype(np.int32)
                cv2.drawContours(img, [scaled_contour], -1, (0, 255, 255), 2)
                cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1) 
                
                # 颜色ID：红1 绿2 蓝3 黑4
                color_id = {'red':1, 'green':2, 'blue':3, 'black':4}.get(color, 0)
                result = {'center_x': cx, 'center_y': cy, 'data': color_id, 'area': area_max}
        return img, result

    def image_callback(self, ros_image):
        if not self.running: return
        try:
            image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
            cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e: return
        processed_img = cv2_img
        res = None
        with self.lock:
            if self.mode == 'color':
                processed_img, res = self.color_detect(cv2_img, self.target_color)
            self.detect_result = res
        try:
            ros_image.data = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB).tobytes()
            self.image_pub.publish(ros_image)
        except: pass

    # --- 动作原语 ---
    def turn_90_left(self):
        rospy.loginfo("左转 90度...")
        self.move_chassis(0, 0, 0.5) 
        rospy.sleep(2.2) 
        self.stop_chassis()
        rospy.sleep(0.5)

    def turn_90_right(self):
        rospy.loginfo("右转 90度...")
        self.move_chassis(0, 0, -0.5) 
        rospy.sleep(2.2) 
        self.stop_chassis()
        rospy.sleep(0.5)
        
    def turn_180_right(self):
        rospy.loginfo("右转 180度...")
        self.move_chassis(0, 0, -0.5)
        rospy.sleep(4.4) 
        self.stop_chassis()
        rospy.sleep(0.5)

    def turn_180_left(self):
        rospy.loginfo("左转 180度...")
        self.move_chassis(0, 0, 0.5)
        rospy.sleep(4.4) 
        self.stop_chassis()
        rospy.sleep(0.5)

    # --- 任务逻辑 ---
    def intelligent_grasp(self, color):
        rospy.loginfo(f"开始智能抓取: {color}")
        self.set_rgb(color)
        
        bus_servo_control.set_servos(self.joints_pub, 1000, ((1, 120),)) 
        rospy.sleep(0.5)
        
        self.x_dis = 500
        self.y_dis = 0.15
        
        target = self.ik.setPitchRanges((0, 0.15, 0.03), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1500, (
                (1, 120), (2, 500), 
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, self.x_dis)
            ))
        rospy.sleep(1.5)
        
        self.x_pid.clear()
        self.y_pid.clear()
        
        stable_track_count = 0
        grasp_ready = False
        offset_y = 0
        color_buf = []
        start_time = time.time()
        
        while not grasp_ready:
            if time.time() - start_time > 20: 
                rospy.logwarn("Grasp timed out")
                return False
                
            res = None
            with self.lock:
                res = self.detect_result
                
            if res:
                cx = res['center_x']
                cy = res['center_y']
                
                if abs(cx - self.centreX) < 10: self.x_pid.SetPoint = cx
                else: self.x_pid.SetPoint = self.centreX
                self.x_pid.update(cx)
                dx = self.x_pid.output
                self.x_dis += int(dx)
                self.x_dis = max(200, min(800, self.x_dis))
                
                if abs(cy - self.centreY) < 10: self.y_pid.SetPoint = cy
                else: self.y_pid.SetPoint = self.centreY
                self.y_pid.update(cy)
                dy = self.y_pid.output
                self.y_dis += dy
                self.y_dis = max(0.12, min(0.28, self.y_dis))
                
                target = self.ik.setPitchRanges((0, round(self.y_dis, 4), 0.03), -180, -180, 0)
                if target:
                    servo_data = target[1]
                    bus_servo_control.set_servos(self.joints_pub, 20, (
                        (3, servo_data['servo3']), (4, servo_data['servo4']),
                        (5, servo_data['servo5']), (6, self.x_dis)
                    ))
                    if len(target) > 2:
                         offset_y = Misc.map(target[2], -180, -150, -0.04, 0.03)
                
                chassis_moving = False
                # Hysteresis 迟滞逻辑
                if self.x_dis > 750: 
                    self.move_chassis(20, 0, 0)
                    chassis_moving = True
                elif self.x_dis < 250:
                    self.move_chassis(20, 180, 0)
                    chassis_moving = True
                
                if not chassis_moving:
                    if self.y_dis > 0.27: 
                        self.move_chassis(20, 90, 0)
                        chassis_moving = True
                    elif self.y_dis < 0.13:
                        self.move_chassis(20, 270, 0)
                        chassis_moving = True
                
                if not chassis_moving:
                    self.stop_chassis()
                else:
                    stable_track_count = 0 

                if not chassis_moving and abs(dx) < 2 and abs(dy) < 0.003:
                    stable_track_count += 1
                    if stable_track_count > 10:
                        color_buf.append(res['data'])
                        if len(color_buf) > 5:
                            grasp_ready = True
                else:
                    stable_track_count = 0
                    color_buf = []
            else:
                rospy.sleep(0.05)
            rospy.sleep(0.05)
            
        # 抓取动作
        rospy.loginfo("执行抓取动作...")
        #这个self.y_dis+offset_y-0.01是为了调整抓取高度，防止抓取到箱子的底部，这里的-0.08的绝对值可以适应的减少，根据实际情况调整
        target = self.ik.setPitchRanges((0, round(self.y_dis + offset_y + 0.01, 4), -0.06), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1000, (
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, self.x_dis)
            ))
        rospy.sleep(1.5)
        
        bus_servo_control.set_servos(self.joints_pub, 500, ((1, 550),)) 
        rospy.sleep(0.8)
        
        bus_servo_control.set_servos(self.joints_pub, 1500, (
            (1, 550), (2, 500), (3, 80), (4, 825), (5, 625), (6, 500)
        ))
        rospy.sleep(1.5)

        # 视觉检测
        grasped = False
        check_start = time.time()
        while time.time() - check_start < 1.0: 
            res = None
            with self.lock:
                res = self.detect_result
            if res:
                area = res.get('area', 0)
                cx = res.get('center_x', 0)
                cy = res.get('center_y', 0)
                if area > 1000 and cy > 300 and abs(cx - 320) < 150:
                    rospy.loginfo(f"视觉抓取检测成功: Area={area}, Pos=({cx},{cy})")
                    grasped = True
                    break
            rospy.sleep(0.1)
            
        if not grasped:
            rospy.logwarn("视觉抓取检测失败")
            bus_servo_control.set_servos(self.joints_pub, 500, ((1, 150),))
            rospy.sleep(0.5)
            self.reset_arm()
            return False

        return True

    def intelligent_place(self, target_color, place_z):
        rospy.loginfo(f"开始放置: 目标={target_color}, 高度Z={place_z}")
        
        self.x_dis = 500
        self.y_dis = 0.15
        pre_z = place_z + 0.1
        if pre_z > 0.15: pre_z = 0.15
        
        target = self.ik.setPitchRanges((0, 0.15, pre_z), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1500, (
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, self.x_dis)
            ))
        rospy.sleep(1.5)
        
        self.x_pid.clear()
        self.y_pid.clear()
        
        with self.lock:
            self.mode = 'color'
            self.target_color = target_color
        
        aligned = False
        stable_count = 0
        offset_y = 0
        start_time = time.time()
        
        last_seen_x = 320
        last_seen_cy = 410
        last_seen_time = 0
        
        target_id_map = {'red':1, 'green':2, 'blue':3, 'black':4}
        target_id_val = target_id_map.get(target_color, 0)
        
        while not aligned:
            if time.time() - start_time > 20: 
                rospy.logwarn("Place alignment timed out")
                break
                
            res = None
            with self.lock:
                res = self.detect_result
            
            if res and res['data'] == target_id_val:
                cx = res['center_x']
                cy = res['center_y']
                
                # X轴修正 (Servo 6 微调 + 车身粗调)
                # 1. 使用PID更新 Servo 6 (x_dis) 以实现精确对齐
                if abs(cx - self.centreX) < 10: self.x_pid.SetPoint = cx
                else: self.x_pid.SetPoint = self.centreX
                self.x_pid.update(cx)
                dx = self.x_pid.output
                self.x_dis += int(dx)
                self.x_dis = max(200, min(800, self.x_dis))

                # 2. 车身修正 (如果偏差太大，或者Servo 6偏离中心太多)
                err_x = self.centreX - cx
                vx = 0
                # 只有当偏差较大时才动车身，小偏差交给机械臂
                if abs(err_x) > 15: 
                    # 降低车身移动速度，尤其是放置时要缓慢 (原来是 min 30, max 15)
                    # 现在改为 min 20, max 10
                    vx = int(min(20, max(10, abs(err_x) * 0.1)))
                    if err_x > 0: dir_x = 180
                    else: dir_x = 0
                    self.move_chassis(vx, dir_x, 0)
                else:
                    self.stop_chassis()
                    
                # Y轴修正
                target_cy = 410
                self.y_pid.SetPoint = target_cy
                self.y_pid.update(cy)
                dy = self.y_pid.output
                self.y_dis += dy
                self.y_dis = max(0.12, min(0.28, self.y_dis))
                
                # 车身前后移动逻辑 (确保机械臂处于最佳工作范围)
                chassis_moving = False
                if vx == 0: # 只有X轴对齐差不多了才调整Y轴车身，避免耦合震荡
                    if self.y_dis > 0.25: # 太远了，往前挪
                        self.move_chassis(20, 90, 0)
                        chassis_moving = True
                    elif self.y_dis < 0.15: # 太近了，往后挪
                        self.move_chassis(20, 270, 0)
                        chassis_moving = True
                
                if chassis_moving:
                    stable_count = 0
                    last_seen_x = cx
                    last_seen_cy = cy
                    last_seen_time = time.time()
                    rospy.sleep(0.05)
                    continue # 还在动车，先不执行机械臂动作

                target = self.ik.setPitchRanges((0, round(self.y_dis, 4), pre_z), -180, -180, 0)
                if target:
                    servo_data = target[1]
                    bus_servo_control.set_servos(self.joints_pub, 20, (
                        (3, servo_data['servo3']), (4, servo_data['servo4']),
                        (5, servo_data['servo5']), (6, self.x_dis)
                    ))
                    # 计算运动学补偿 (与抓取逻辑保持一致)
                    if len(target) > 2:
                         offset_y = Misc.map(target[2], -180, -150, -0.04, 0.03)
                    
                # 多帧稳定性确认 (要求X轴和Y轴都稳定)
                # dx是Servo6的PID输出，代表X轴需要的修正量
                if vx == 0 and not chassis_moving and abs(dy) < 0.003 and abs(dx) < 2: 
                     stable_count += 1
                     if stable_count > 5:
                         rospy.loginfo(f"对齐完成 (Stable) Pos:({cx},{cy}) Servo6:{self.x_dis}")
                         rospy.sleep(0.1) 
                         aligned = True
                         self.stop_chassis()
                         break
                else:
                     stable_count = 0
                     
                last_seen_x = cx
                last_seen_cy = cy
                last_seen_time = time.time()
                
            else:
                # 遮挡冲刺逻辑
                time_since_lost = time.time() - last_seen_time
                if time_since_lost < 1.0:
                    err_x_last = self.centreX - last_seen_x
                    if abs(err_x_last) > 5:
                        dash_speed = 20
                        dash_time = abs(err_x_last) * 0.005 
                        dash_time = max(0.1, min(0.8, dash_time))
                        
                        if err_x_last > 0: dash_dir = 180 
                        else: dash_dir = 0
                        
                        rospy.loginfo(f"遮挡冲刺: Time={dash_time:.2f}s")
                        self.move_chassis(dash_speed, dash_dir, 0)
                        rospy.sleep(dash_time)
                        self.stop_chassis()
                    
                    aligned = True
                    break
                else:
                    self.stop_chassis()
                    rospy.sleep(0.05)
                
            rospy.sleep(0.05)
            
        # 执行放置 (柔性下压策略)
        # 1. 计算压实高度 (比目标低 1mm，之前是2mm，防止压力过大崩塌)
        press_z = place_z - 0.001
        rospy.loginfo(f"执行放置 (Target Z={place_z}, Press Z={press_z:.4f})...")
        
        target = self.ik.setPitchRanges((0, round(self.y_dis + offset_y, 4), press_z), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1500, ( # 动作放慢到1.5s
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, self.x_dis)
            ))
        rospy.sleep(1.8) # 充分等待稳定
        
        # 2. 缓慢松爪
        bus_servo_control.set_servos(self.joints_pub, 800, ((1, 150),)) # 速度800ms，缓慢张开
        rospy.sleep(0.8)
        
        # 安全撤离：先垂直抬起一段距离，防止爪子扫倒积木
        rospy.loginfo("安全撤离：垂直抬起...")
        safe_z = place_z + 0.08 # 抬起8cm
        target = self.ik.setPitchRanges((0, round(self.y_dis, 4), safe_z), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1000, (
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, self.x_dis)
            ))
        rospy.sleep(1.0)
        
        # --- 堆叠成功检测 ---
        # 抬起后，检查下方的目标颜色面积
        # 如果堆叠成功，原来的目标（比如下面的红色积木）应该被挡住大部分，面积会显著减小
        # 或者如果是放在黑框上，我们无法直接判断“挡住”，但可以检查是否还能看到原来的大面积色块
        # 这里的逻辑是：如果放置成功，视野里应该能看到刚放下去的积木(当前抓的颜色)，或者原来底下的颜色变少了
        # 但最简单的检测是：检测“底座颜色”的面积。如果面积依然很大且完整，说明可能没放准掉下去了。
        
        # 切换到检测底座颜色
        self.reset_arm() # 先归位，让摄像头视野更好
        rospy.sleep(1.0)
        
        check_success = True
        if target_color != 'black': # 黑框太大不好判断，主要判断叠在红/蓝积木上的情况
             rospy.loginfo(f"检查堆叠结果 (底座: {target_color})...")
             # 此时机械臂已归位，视野开阔
             with self.lock:
                 res = self.detect_result
             
             if res and res['data'] == target_id_val:
                 area = res['area']
                 rospy.loginfo(f"底座颜色面积: {area}")
                 
                 # 阈值判断：如果底座面积依然很大，认为堆叠失败
                 # 正常红/蓝积木的面积大约在 2000-3000 左右(近距离)
                 # 如果被遮挡，面积应该会显著变小或者形状改变
                 # 这里假设：如果还能检测到完整的底座且面积 > 2500，则判定为失败
                 if area > 2500: 
                     rospy.logwarn("堆叠检测：底座面积过大，判定为未遮挡/失败")
                     return False
                 else:
                     rospy.loginfo("堆叠检测：底座面积减小，判定为成功")
                     return True
             else:
                # 根本没检测到底座颜色了（完全遮挡），也算成功
                rospy.loginfo("堆叠检测：底座被完全遮挡，判定为成功")
                return True
        
        return True

    def run_task(self):
        rospy.loginfo("比赛任务开始：码垛搬运")
        
        # 任务序列：(抓取颜色, 放置参照物颜色, 放置高度Z)
        tasks = [
            ('red', 'black', -0.05), # 抓红，放黑框
            ('blue', 'red', -0.02),  # 抓蓝，放红上
            ('green', 'blue', 0.01)  # 抓绿，放蓝上
        ]
        
        # (1) 移动到货物区
        rospy.loginfo("(1) 移动到货物摆放区...")
        self.move_chassis(50, 90, 0)
        rospy.sleep(5.0) 
        self.stop_chassis()
        rospy.sleep(0.5)

        # 初始左转，面向货物区 (只执行一次)
        self.turn_90_left()
        
        for i, (grasp_color, place_target, place_z) in enumerate(tasks):
            rospy.loginfo(f"--- 任务 {i+1}: 抓 {grasp_color} -> 放 {place_target} 上 ---")
            
            # (2)/(3)/(4) 搜索积木 (左侧)
            # 移除靠近动作：左转90度后，直接看是否能检测到，不需要先直行
            # self.move_chassis(50, 90, 0) # 靠近 (移除)
            # rospy.sleep(1.0)
            # self.stop_chassis()
            rospy.sleep(0.5)
            
            # 开启识别
            with self.lock: 
                self.mode = 'color'
                self.target_color = grasp_color
            
            # 抓取 (5次尝试)
            grasp_success = False
            search_actions = ['center', 'left', 'right', 'center', 'left']
            
            for i_try, pos in enumerate(search_actions):
                rospy.loginfo(f"尝试抓取 {grasp_color} (位置: {pos})...")
                if self.intelligent_grasp(grasp_color):
                    rospy.loginfo(f"成功抓取 {grasp_color}")
                    grasp_success = True
                    break
                else:
                    if i_try < len(search_actions) - 1:
                        next_pos = search_actions[i_try + 1]
                        if pos == 'center' and next_pos == 'left':
                            self.move_chassis(30, 180, 0)
                            rospy.sleep(0.6)
                        elif pos == 'left' and next_pos == 'right':
                            self.move_chassis(30, 0, 0)
                            rospy.sleep(1.2)
                        elif pos == 'right' and next_pos == 'center':
                            self.move_chassis(30, 180, 0)
                            rospy.sleep(0.6)
                        elif pos == 'center' and next_pos == 'left':
                             self.move_chassis(30, 180, 0)
                             rospy.sleep(0.6)
                        self.stop_chassis()
                        rospy.sleep(0.5)

            if grasp_success:
                # 抓到后动作
                rospy.loginfo("抓取后后退...")
                self.move_chassis(50, 270, 0)
                rospy.sleep(1.0)
                self.stop_chassis()
                rospy.sleep(0.5)
                
                # 运送到码垛区 (右侧)
                self.turn_180_right()
                
                # 靠近
                rospy.loginfo(f"寻找 {place_target} 并放置...")
                # 移除靠近动作：右转180度后，直接开始对齐，不需要先直行
                # self.move_chassis(50, 90, 0)
                # rospy.sleep(1.0)
                # self.stop_chassis()
                rospy.sleep(0.5)
                
                # 放置
                place_success = False
                for i_place in range(3): # 最多尝试3次
                    if self.intelligent_place(place_target, place_z):
                        rospy.loginfo(f"成功放置 {grasp_color}")
                        place_success = True
                        break
                    else:
                        rospy.logwarn(f"放置 {grasp_color} 失败 (检测到未堆叠成功)，尝试重试 {i_place+1}/3...")
                        # 如果失败，说明没叠上（可能掉旁边了，也可能没放准）
                        # 此时需要重新调整位置，或者稍作退后再次尝试
                        self.move_chassis(20, 270, 0) # 后退一点
                        rospy.sleep(1.0)
                        self.stop_chassis()
                        rospy.sleep(0.5)
                        self.move_chassis(20, 90, 0) # 再前进回来，重新对齐
                        rospy.sleep(1.0)
                        self.stop_chassis()
                
                if not place_success:
                    rospy.logwarn(f"放置 {grasp_color} 最终失败")
                
                # 放置后后退
                rospy.loginfo("放置后后退...")
                self.move_chassis(50, 270, 0)
                rospy.sleep(1.0)
                self.stop_chassis()
                rospy.sleep(0.5)
                    
                if i < len(tasks) - 1:
                    # 按照用户要求：左转180度 (直接面向左侧货物区)
                    self.turn_180_left()
                    
                    # 前往下一工位 (如果需要)
                    rospy.loginfo("前往下一工位 (右移)...")
                    self.move_chassis(50, 0, 0) # 向右平移
                    rospy.sleep(1.5) # 假设需要移一段距离
                    self.stop_chassis()
                else:
                    # 最后一次任务，直接左转90度，面向前方
                    rospy.loginfo("最后一次任务，直接左转90度...")
                    self.turn_90_left()
            else:
                rospy.logwarn(f"抓取 {grasp_color} 失败")
                # 即使失败也要保持队形
                if i < len(tasks) - 1:
                     rospy.loginfo("抓取失败，跳过放置，前往下一工位...")
                     self.move_chassis(50, 0, 0) # 向右平移(假设面向左)
                     rospy.sleep(1.5)
                     self.stop_chassis()
                else:
                     # 最后一次任务失败，右转90度，面向前方
                     rospy.loginfo("最后一次任务失败，右转90度...")
                     self.turn_90_right()
            
        # (5) 返回停放区
        rospy.loginfo("(5) 返回停放区 (修正朝向后退)...")
        # 此时已经面向前方
        
        # 然后倒车7秒
        rospy.loginfo("倒车7秒...")
        self.move_chassis(50, 270, 0) 
        rospy.sleep(7.0)
        self.stop_chassis()
        
        # 加分项调整：无需指向箭头，只需完成初始化
        rospy.loginfo("到达停放区，执行最终复位...")
        self.init_robot() # 机械臂归位，关灯，停车
            
        rospy.loginfo("任务全部完成")
        self.stop_chassis()
        
    def start(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        task = SmartPalletizerTask()
        task.start()
    except rospy.ROSInterruptException:
        pass
