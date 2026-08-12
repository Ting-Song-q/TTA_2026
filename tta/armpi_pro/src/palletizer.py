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
    def getAreaMaxContour(self, contours):
        contour_area_temp = 0
        contour_area_max = 0
        area_max_contour = None
        for c in contours:
            contour_area_temp = math.fabs(cv2.contourArea(c))
            if contour_area_temp > contour_area_max:
                contour_area_max = contour_area_temp
                if contour_area_temp > 10:
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
            'black': [((0, 0, 0), (180, 255, 46))] 
        }
        
        if color in hsv_ranges:
            mask = np.zeros(frame_hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in hsv_ranges[color]:
                mask |= cv2.inRange(frame_hsv, np.array(lower), np.array(upper))
            
            # 形态学处理优化：针对空心黑框，需要保护细边框不被腐蚀断裂
            if color == 'black':
                # 黑色边框可能较细，先微量腐蚀去噪，再大力膨胀连接断点
                eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))) # 轻微腐蚀
                # 使用较大的核进行膨胀，确保空心框的边能连上，同时也能填充内部细小空隙(如果有)
                # 注意：如果中间白色区域很大，膨胀不会把它填满，只会让框变粗，这对中心计算无影响
                dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))) 
            else:
                # 其他实心积木保持原有逻辑
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
        target = self.ik.setPitchRanges((0, round(self.y_dis + offset_y - 0.02, 4), -0.08), -180, -180, 0)
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
                
                # X轴修正
                err_x = self.centreX - cx
                vx = 0
                if abs(err_x) > 2:
                    vx = int(min(30, max(15, abs(err_x) * 0.15)))
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
                
                target = self.ik.setPitchRanges((0, round(self.y_dis, 4), pre_z), -180, -180, 0)
                if target:
                    servo_data = target[1]
                    bus_servo_control.set_servos(self.joints_pub, 20, (
                        (3, servo_data['servo3']), (4, servo_data['servo4']),
                        (5, servo_data['servo5']), (6, self.x_dis)
                    ))
                    
                if vx == 0 and abs(dy) < 0.005: 
                     rospy.loginfo("对齐完成")
                     rospy.sleep(0.1) 
                     aligned = True
                     self.stop_chassis()
                     break
                     
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
            
        # 执行放置
        rospy.loginfo(f"执行放置 (Z={place_z})...")
        target = self.ik.setPitchRanges((0, round(self.y_dis, 4), place_z), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1000, (
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, self.x_dis)
            ))
        rospy.sleep(1.5)
        bus_servo_control.set_servos(self.joints_pub, 500, ((1, 150),)) # 张开
        rospy.sleep(0.8)
        
        self.reset_arm()
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
        
        for i, (grasp_color, place_target, place_z) in enumerate(tasks):
            rospy.loginfo(f"--- 任务 {i+1}: 抓 {grasp_color} -> 放 {place_target} 上 ---")
            
            # (2)/(3)/(4) 搜索积木 (左侧)
            self.turn_90_left()
            self.move_chassis(50, 90, 0) # 靠近
            rospy.sleep(1.0)
            self.stop_chassis()
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
                self.move_chassis(50, 90, 0)
                rospy.sleep(1.0)
                self.stop_chassis()
                rospy.sleep(0.5)
                
                # 放置
                if self.intelligent_place(place_target, place_z):
                    rospy.loginfo(f"成功放置 {grasp_color}")
                else:
                    rospy.logwarn(f"放置 {grasp_color} 失败")
                
                # 放置后后退
                rospy.loginfo("放置后后退...")
                self.move_chassis(50, 270, 0)
                rospy.sleep(1.0)
                self.stop_chassis()
                rospy.sleep(0.5)
                    
                # 回正
                self.turn_90_left()
                
                # 下一工位
                if i < len(tasks) - 1:
                     rospy.loginfo("前往下一工位...")
                     self.move_chassis(50, 90, 0)
                     rospy.sleep(1.0)
                     self.stop_chassis()
            else:
                rospy.logwarn(f"抓取 {grasp_color} 失败")
                self.turn_90_right() # 回正
                rospy.sleep(1.0)
            
        # (5) 返回停放区
        rospy.loginfo("(5) 返回停放区 (后退7秒)...")
        self.move_chassis(50, 270, 0) # 后退
        rospy.sleep(7.0) # 修改：严格按照要求退后7秒
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
