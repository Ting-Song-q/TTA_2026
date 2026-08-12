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
from hiwonder_servo_msgs.msg import MultiRawIdPosDur
from chassis_control.msg import SetTranslation, SetVelocity

# 导入项目库
from armpi_pro import PID
from armpi_pro import Misc
from armpi_pro import bus_servo_control
from armpi_pro import apriltag
from kinematics import ik_transform

class VerifyRedBlockTask:
    def __init__(self):
        rospy.init_node('verify_red_block_task', log_level=rospy.INFO)
        
        self.lock = RLock()
        self.ik = ik_transform.ArmIK()
        
        # 发布者
        self.joints_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
        self.set_velocity = rospy.Publisher('/chassis_control/set_velocity', SetVelocity, queue_size=1)
        self.buzzer_pub = rospy.Publisher('/sensor/buzzer', Float32, queue_size=1)
        self.rgb_pub = rospy.Publisher('/sensor/rgb_led', Led, queue_size=1)
        self.image_pub = rospy.Publisher('/visual_processing/image_result', Image, queue_size=1)
        
        # 参数
        self.color_range_list = rospy.get_param('/lab_config_manager/color_range_list', {})
        self.img_w = 640
        self.img_h = 480
        self.size_m = (320, 240)
        
        # AprilTag检测器
        self.detector = apriltag.Detector(searchpath=apriltag._get_demo_searchpath())
        
        # 检测状态
        self.mode = 'None' # 'None', 'color', 'apriltag'
        self.target_color = 'None'
        self.target_tag_id = None
        self.detect_result = None # {'center_x': int, 'center_y': int, 'data': any}
        self.updated = False
        
        # PID控制器
        self.x_pid = PID.PID(P=0.06, I=0, D=0)
        self.y_pid = PID.PID(P=0.00003, I=0, D=0)
        
        # 摄像头订阅者
        self.running = True
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        
        # 任务状态
        self.running = True
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
        # 将机械臂移动到初始行驶位置
        target = self.ik.setPitchRanges((0, 0.15, 0.10), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1500, (
                (1, 200), (2, 500), 
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, servo_data['servo6'])
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
        # 按要求使用HSV
        frame_hsv = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2HSV)
        
        result = None
        
        # 为了鲁棒性硬编码HSV范围
        hsv_ranges = {
            'red': [
                ((0, 43, 46), (10, 255, 255)),
                ((156, 43, 46), (180, 255, 255))
            ],
            'green': [((35, 43, 46), (77, 255, 255))],
            'blue': [((100, 43, 46), (124, 255, 255))]
        }

        if color in hsv_ranges:
            mask = np.zeros(frame_hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in hsv_ranges[color]:
                mask |= cv2.inRange(frame_hsv, np.array(lower), np.array(upper))
            
            eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
            dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
            contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
            area_max_contour, area_max = self.getAreaMaxContour(contours)

            if area_max > 100:
                (centerx, centery), radius = cv2.minEnclosingCircle(area_max_contour)
                cx = int(Misc.map(centerx, 0, self.size_m[0], 0, img_w))
                cy = int(Misc.map(centery, 0, self.size_m[1], 0, img_h))
                rad = int(Misc.map(radius, 0, self.size_m[0], 0, img_w))
                
                cv2.circle(img, (cx, cy), rad+5, self.range_rgb[color], 2)
                
                # 根据颜色分配ID以兼容intelligent_grasp逻辑
                # 红: 1, 绿: 2, 蓝: 3
                color_id = 0
                if color == 'red': color_id = 1
                elif color == 'green': color_id = 2
                elif color == 'blue': color_id = 3
                
                result = {'center_x': cx, 'center_y': cy, 'data': color_id}
                
        return img, result

    def apriltag_detect(self, img, target_id):
        img_copy = img.copy()
        img_h, img_w = img.shape[:2]
        frame_resize = cv2.resize(img_copy, self.size_m, interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray, return_image=False)
        
        result = None
        id_smallest = None
        
        if len(detections) != 0:
            for i, detection in enumerate(detections):
                tag_id = int(detection.tag_id)
                
                # 按ID过滤
                if target_id is not None and tag_id != target_id:
                    continue
                
                corners = np.rint(detection.corners)
                for i in range(4):
                    corners[i][0] = int(Misc.map(corners[i][0], 0, self.size_m[0], 0, img_w))
                    corners[i][1] = int(Misc.map(corners[i][1], 0, self.size_m[1], 0, img_h))
                
                cv2.drawContours(img, [np.array(corners, np.int)], -1, (0, 255, 255), 6)
                object_center_x = int(Misc.map(detection.center[0], 0, self.size_m[0], 0, img_w))
                object_center_y = int(Misc.map(detection.center[1], 0, self.size_m[1], 0, img_h))
                
                cv2.putText(img, str(tag_id), (object_center_x - 10, object_center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 255, 255], 6)
                
                # 如果发现多个，选择最小的ID（或者只选择目标的那个）
                if id_smallest is None or tag_id <= id_smallest:
                    id_smallest = tag_id
                    result = {'center_x': object_center_x, 'center_y': object_center_y, 'data': tag_id}
                    
        return img, result

    def image_callback(self, ros_image):
        if not self.running:
            return
            
        try:
            image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
            cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            rospy.logerr(f"Image decode error: {e}")
            return

        processed_img = cv2_img
        res = None

        with self.lock:
            if self.mode == 'color':
                processed_img, res = self.color_detect(cv2_img, self.target_color)
            elif self.mode == 'apriltag':
                processed_img, res = self.apriltag_detect(cv2_img, self.target_tag_id)
            
            self.detect_result = res
            self.updated = True

        # 发布结果图像
        try:
            rgb_image = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB).tobytes()
            ros_image.data = rgb_image
            self.image_pub.publish(ros_image)
        except Exception as e:
            pass

    def move_to_cargo_area(self):
        rospy.loginfo("移动到货物区...")
        self.move_chassis(50, 90, 0)
        rospy.sleep(3.0)
        self.stop_chassis()
        rospy.sleep(0.5)

    #search_color函数
    def search_color(self, color):
        rospy.loginfo(f"正在搜索 {color} 物块（机械臂向左前）...")
        with self.lock:
            self.mode = 'color'
            self.target_color = color
            self.detect_result = None
        
        # 将机械臂底座（舵机6）转向左前（625）
        # 500是中间，750是正左。625是左前45度。
        
        found = False
        start_time = time.time()
        
        # 将机械臂保持在左前位置以便搜索
        bus_servo_control.set_servos(self.joints_pub, 1500, ((6, 625),))
        rospy.sleep(1.0)
        
        # 静态检查
        with self.lock:
            if self.detect_result:
                rospy.loginfo(f"Found {color} at start!")
                found = True
            
        if not found:
             # 如果初步扫描未找到，底盘向前移动搜索
             rospy.loginfo(f"{color} not found in scan. Moving forward to search...")
             
             # 向前移动 1.5s
             self.move_chassis(50, 90, 0)
             t_start = time.time()
             while time.time() - t_start < 1.5:
                 with self.lock:
                     if self.detect_result:
                         rospy.loginfo(f"Found {color} while moving forward!")
                         found = True
                         break
                 rospy.sleep(0.05)
             self.stop_chassis()
             
             # 如果还没找到，向后移动 3s
             if not found:
                 rospy.loginfo(f"{color} still not found. Moving backward to search...")
                 self.move_chassis(50, 270, 0)
                 t_start = time.time()
                 while time.time() - t_start < 3.0:
                     with self.lock:
                         if self.detect_result:
                             rospy.loginfo(f"Found {color} while moving backward!")
                             found = True
                             break
                     rospy.sleep(0.05)
                 self.stop_chassis()
            
        return found

    #align_and_grasp函数
    def align_and_grasp(self, color):
        rospy.loginfo(f"正在对齐以抓取 {color}...")
        self.set_rgb(color)
        
        # 注意：底盘现在是静止的。我们使用机械臂底座（ID 6）来居中物体。
        
        # 1. 机械臂抓取逻辑
        bus_servo_control.set_servos(self.joints_pub, 800, ((1, 120),)) # 张开爪子
        rospy.sleep(0.5)
        
        # 移动到悬停位置（机械臂向左前 625）
        self.x_dis = 625
        
        target = self.ik.setPitchRanges((0, 0.15, 0.03), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1500, (
                (1, 120), (2, 500), 
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, self.x_dis) # 强制ID 6为625
            ))
        rospy.sleep(1.5)
        
        self.x_pid.clear()
        self.y_pid.clear()
        self.y_dis = 0.15
        
        stable_count = 0
        
        # 稳定性检查变量（相当于position_en）
        last_x = 0
        last_y = 0
        stable_pos_count = 0
        position_en = False
        
        # 最终抓取触发变量
        stable_track_count = 0
        arm_move = False
        offset_y = 0
        color_buf = []
        
        # 对齐超时逻辑
        align_start_time = time.time()
        
        # 阶段1：底盘粗对齐
        rospy.loginfo("阶段1：底盘粗对齐...")
        chassis_aligned = False
        
        while not chassis_aligned:
            # 超时检查
            if time.time() - align_start_time > 30:
                rospy.logwarn("Chassis alignment timed out!")
                return False
                
            res = None
            with self.lock:
                res = self.detect_result
                
            if res:
                cx = res['center_x']
                cy = res['center_y']
                
                # 逻辑修改：
                # 机械臂现在朝向左前 (45度)。摄像头也是朝向左前。
                # 图像坐标系：
                # X轴 (cx): 左右 -> 对应 实际空间的 "左前-右后" 轴线上的偏移
                # Y轴 (cy): 远近 -> 对应 实际空间的 "右前-左后" 轴线上的距离
                
                # 为了简化控制，我们使用分解速度。
                # 但更简单的方法是：保持底盘朝向不变 (0度)，计算需要在X和Y方向平移的分量。
                
                # 实际上，如果机械臂转了45度，图像的X轴对应的是切向运动，Y轴对应的是径向运动。
                # 径向（靠近/远离）：沿着 45度 方向移动。
                # 切向（左右偏移）：沿着 135度 方向移动。
                
                # 1. Y轴误差 (远近) -> 径向移动
                # cy 小 (远) -> 需要靠近 -> 向左前移动 (45度)
                # cy 大 (近) -> 需要远离 -> 向右后移动 (225度)
                
                # 2. X轴误差 (左右) -> 切向移动
                # cx 小 (图像左) -> 目标在左侧 -> 需要向左移动 -> 向左后移动 (135度)
                # cx 大 (图像右) -> 目标在右侧 -> 需要向右移动 -> 向右前移动 (315度)
                
                vx_local = 0
                vy_local = 0
                
                # 处理 X 轴 (切向)
                err_x = self.centreX - cx
                if err_x > 20: # 偏左
                    # 向左平移 (相对于摄像头的左) -> 实际是向左后 (135度)
                    # move_chassis(speed, direction)
                    # 我们可以直接叠加向量。
                    # 或者简单点，一次只修一个轴。
                    move_dir = 135
                    speed = 25
                elif err_x < -20: # 偏右
                    # 向右平移 -> 实际是向右前 (315度)
                    move_dir = 315
                    speed = 25
                else:
                    move_dir = 0
                    speed = 0
                
                # 处理 Y 轴 (径向)
                # 如果X轴已经在调了，先不管Y轴，或者叠加？
                # 为了稳定，优先调距离 (Y轴)，还是优先调居中？
                # 通常一起调。
                
                # 让我们计算全局的 Vx, Vy
                # 目标速度向量 V_cam = (v_x_img, v_y_img)
                # v_x_img: 修正 cx。 err_x > 0 -> 需要向左移 -> v_x_img < 0
                # v_y_img: 修正 cy。 cy < target (远) -> 需要向前移 -> v_y_img > 0
                
                # 映射到底盘坐标系：
                # V_robot = Rotation_matrix(-45 deg) * V_cam
                # 因为机械臂向左转了，相对于车头是 +90度? 不，左前是 +45度?
                # 之前的代码：左侧是 90度移动是向前。
                # 之前的定义：90度是向前，0度是向右，180是向左。
                # 左前应该是 135度 (90+45) 还是 45度?
                # 机械臂 ID 6: 500=中间, 750=左(90度), 1000=后(180度)
                # 所以 625 大概是 左 45度。
                # 也就是 90(前) + 45 = 135度方向。
                
                # 所以径向（靠近物体）是朝 135度 移动。
                # 切向（修正左右）是朝 135-90=45度 或 135+90=225度 移动。
                
                # 修正：
                # 径向 (Y轴):
                # 太远 (cy small) -> 靠近 -> 朝 135度 移动
                # 太近 (cy big) -> 远离 -> 朝 315度 移动
                
                # 切向 (X轴):
                # 偏左 (cx small, err>0) -> 像左移 -> 朝 135+90 = 225度 移动
                # 偏右 (cx big, err<0) -> 向右移 -> 朝 135-90 = 45度 移动
                
                final_speed = 0
                final_dir = 0
                
                if cy < 150: # 太远
                    final_speed = 25
                    final_dir = 135
                elif cy > 250: # 太近
                    final_speed = 25
                    final_dir = 315
                elif err_x > 20: # 偏左
                    final_speed = 25
                    final_dir = 225
                elif err_x < -20: # 偏右
                    final_speed = 25
                    final_dir = 45
                else:
                    # 对齐完成
                    rospy.loginfo("底盘对齐完成！")
                    self.stop_chassis()
                    chassis_aligned = True
                    continue
                
                self.move_chassis(final_speed, final_dir, 0)
                    
            else:
                self.stop_chassis()
                rospy.sleep(0.05)
            
            rospy.sleep(0.05)
            
        # 阶段2：机械臂微调与抓取
        rospy.loginfo("阶段2：机械臂微调...")
        
        while not arm_move:
            # 超时检查
            if time.time() - align_start_time > 60:
                rospy.logwarn("Arm alignment timed out!")
                return False

            res = None
            with self.lock:
                res = self.detect_result
                
            if res:
                cx = res['center_x']
                cy = res['center_y']
                color_id = res['data']
                
                # --- 稳定性检查 ---
                if not position_en:
                    dx_pos = abs(cx - last_x)
                    dy_pos = abs(cy - last_y)
                    last_x = cx
                    last_y = cy
                    if dx_pos < 3 and dy_pos < 3:
                        stable_pos_count += 1
                        if stable_pos_count == 10:
                            stable_pos_count = 0
                            position_en = True
                    else:
                        stable_pos_count = 0
                else:
                    # --- 追踪逻辑 ---
                    
                    # 1. X轴修正 (使用底盘微调，不旋转ID 6)
                    # 目标：cx = centreX (320)
                    vx_micro = 0
                    err_x = self.centreX - cx
                    if abs(err_x) > 15:
                        speed_micro = int(min(30, max(15, abs(err_x) * 0.1)))
                        # 偏左 (err>0) -> 向左移 -> 225度
                        # 偏右 (err<0) -> 向右移 -> 45度
                        if err_x > 0:
                            vx_micro = speed_micro
                            dir_x = 225
                        else:
                            vx_micro = speed_micro
                            dir_x = 45
                        
                        self.move_chassis(vx_micro, dir_x, 0)
                    else:
                        self.stop_chassis()
                    
                    # 2. Y轴修正 (使用机械臂伸缩)
                    self.y_pid.SetPoint = self.centreY
                    self.y_pid.update(cy)
                    dy = self.y_pid.output
                    self.y_dis += dy
                    self.y_dis = max(0.12, min(0.28, self.y_dis))
                    
                    # 移动机械臂 (ID 6 保持 self.x_dis = 625 不变)
                    target = self.ik.setPitchRanges((0, round(self.y_dis, 4), 0.03), -180, -180, 0)
                    if target:
                        servo_data = target[1]
                        bus_servo_control.set_servos(self.joints_pub, 20, (
                            (3, servo_data['servo3']), (4, servo_data['servo4']),
                            (5, servo_data['servo5']), (6, self.x_dis)
                        ))
                        
                        if len(target) > 2:
                            offset_y = Misc.map(target[2], -180, -150, -0.04, 0.03)
                    
                    # --- 稳定性追踪检查 ---
                    if vx_micro == 0 and abs(dy) < 0.003:
                        stable_track_count += 1
                        if stable_track_count == 10:
                            stable_track_count = 0
                            color_buf.append(color_id)
                            if len(color_buf) == 5:
                                mean_num = np.mean(color_buf)
                                if mean_num == 1.0 or mean_num == 2.0 or mean_num == 3.0:
                                    target_map = {'red': 1, 'green': 2, 'blue': 3}
                                    if target_map.get(color) == int(mean_num):
                                        rospy.loginfo(f"Grasp Confirmed: Mean Color ID {mean_num}")
                                        arm_move = True
                                    else:
                                        rospy.logwarn(f"Color Mismatch! Target: {color}, Detected Mean: {mean_num}")
                                        color_buf = []
                                else:
                                    color_buf = []
                    else:
                        stable_track_count = 0
                        color_buf = []
            else:
                 rospy.sleep(0.05)
                        
            rospy.sleep(0.05)
        
        # 执行抓取
        rospy.loginfo("Executing Grasp...")
        target = self.ik.setPitchRanges((0, round(self.y_dis + offset_y, 4), -0.08), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1500, (
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, self.x_dis)
            ))
        rospy.sleep(1.5)
        bus_servo_control.set_servos(self.joints_pub, 800, ((1, 450),)) # 闭合
        rospy.sleep(0.8)
        
        # 抬起机械臂
        bus_servo_control.set_servos(self.joints_pub, 2000, (
            (1, 450), (2, 500), (3, 80), (4, 825), (5, 625), (6, self.x_dis)
        ))
        rospy.sleep(1.5)
        return True

    def find_and_place_tag(self, tag_id):
        rospy.loginfo(f"正在搜索 AprilTag {tag_id}（机械臂向右前）...")
        with self.lock:
            self.mode = 'apriltag'
            self.target_tag_id = tag_id
            self.detect_result = None
            
        # 将机械臂底座（舵机6）向右前转 (375)
        # 500是中间(90度前)，250是右(0度)，375是右前(45度)
        bus_servo_control.set_servos(self.joints_pub, 2000, ((6, 375),))
        rospy.sleep(2.0)
        
        # 等待检测
        start_time = time.time()
        found = False
        while time.time() - start_time < 3:
            with self.lock:
                if self.detect_result and self.detect_result['data'] == tag_id:
                    found = True
                    break
            rospy.sleep(0.1)
            
        if not found:
             # 尝试通过轻微移动底盘来搜索
             rospy.logwarn(f"Tag {tag_id} 未立即找到，正在调整底盘...")
             
             # 搜索序列：沿45度方向移动搜索
             search_moves = [
                 (30, 45, 0),    # 向右前平移 (视野前方)
                 (30, 225, 0),   # 向左后平移 (视野后方)
                 (30, 135, 0),   # 向左前平移 (视野左侧)
                 (30, 315, 0)    # 向右后平移 (视野右侧)
             ]
             
             for move in search_moves:
                 rospy.loginfo(f"搜索移动：{move}")
                 self.move_chassis(*move)
                 
                 # 移动中检测 (持续1秒)
                 t_start = time.time()
                 while time.time() - t_start < 1.0:
                      with self.lock:
                        if self.detect_result and self.detect_result['data'] == tag_id:
                            found = True
                            break
                      rospy.sleep(0.05)
                 
                 self.stop_chassis()
                 if found: break
                 
                 rospy.sleep(0.5) # 稍微停顿
             
             if not found:
                 rospy.logwarn(f"Tag {tag_id} 仍然未找到！")
                 return False
             else:
                 rospy.loginfo(f"Tag {tag_id} 在调整后找到！")
        
        # 对齐（使用底盘控制X，伸展控制Y）
        rospy.loginfo(f"正在对齐到 Tag {tag_id}...")
        self.x_pid.clear()
        self.y_pid.clear()
        
        # 固定ID 6在右前侧 (375)
        self.x_dis = 375
        
        # 移动到放置高度（高于抓取）
        target = self.ik.setPitchRanges((0, 0.15, 0.05), -180, -180, 0)
        if target:
             servo_data = target[1]
             bus_servo_control.set_servos(self.joints_pub, 1500, (
                 (3, servo_data['servo3']), (4, servo_data['servo4']),
                 (5, servo_data['servo5']), (6, self.x_dis) # 强制ID 6
             ))
        rospy.sleep(1.0)

        stable_track_count = 0
        placed = False
        
        # 允许的Y轴误差范围 (Tag距离)
        y_min_dist = 0.12
        y_max_dist = 0.28
        
        while not placed:
            res = None
            with self.lock:
                res = self.detect_result
                
            if res and res['data'] == tag_id:
                cx = res['center_x']
                cy = res['center_y']
                
                # --- 追踪逻辑（底盘微调 + 机械臂Y轴伸缩） ---
                
                # 1. X轴修正 (使用底盘微调，不旋转ID 6)
                # 目标：cx = centreX (320)
                # 机械臂朝向 45度 (右前)
                # cx < 320 (图像左) -> 目标在视野左侧 -> 向左平移 (相对于45度) -> 45 + 90 = 135度
                # cx > 320 (图像右) -> 目标在视野右侧 -> 向右平移 (相对于45度) -> 45 - 90 = 315度 (-45)
                
                vx_micro = 0
                err_x = self.centreX - cx
                if abs(err_x) > 15:
                    speed_micro = int(min(30, max(15, abs(err_x) * 0.1)))
                    if err_x > 0: # cx < 320 (Left)
                        vx_micro = speed_micro
                        dir_x = 135
                    else: # cx > 320 (Right)
                        vx_micro = speed_micro
                        dir_x = 315
                    
                    self.move_chassis(vx_micro, dir_x, 0)
                else:
                    self.stop_chassis()
                
                # 2. Y轴修正 (使用机械臂伸缩 + 底盘辅助)
                self.y_pid.SetPoint = self.centreY
                self.y_pid.update(cy)
                dy = self.y_pid.output
                self.y_dis += dy
                
                # 如果超出了机械臂的伸缩范围，使用底盘辅助平移
                # 视野方向是 45度
                # 太远 (Need Approach) -> Move 45
                # 太近 (Need Retreat) -> Move 225
                
                chassis_y_move = False
                if self.y_dis > y_max_dist: # 太远，向靠近Tag
                     self.move_chassis(25, 45, 0)
                     chassis_y_move = True
                elif self.y_dis < y_min_dist: # 太近，远离Tag
                     self.move_chassis(25, 225, 0)
                     chassis_y_move = True
                
                # 限制机械臂伸展范围
                self.y_dis = max(y_min_dist, min(y_max_dist, self.y_dis))
                
                # 移动机械臂
                target = self.ik.setPitchRanges((0, round(self.y_dis, 4), 0.05), -180, -180, 0)
                if target:
                    servo_data = target[1]
                    bus_servo_control.set_servos(self.joints_pub, 20, (
                        (3, servo_data['servo3']), (4, servo_data['servo4']),
                        (5, servo_data['servo5']), (6, self.x_dis)
                    ))
                
                # --- 稳定性检查 ---
                # 底盘必须停止 (X轴对齐完成 & Y轴无需底盘辅助) 且 机械臂伸缩稳定
                if vx_micro == 0 and not chassis_y_move and abs(dy) < 0.003:
                    stable_track_count += 1
                    if stable_track_count == 20: # 确保它真的稳定
                        placed = True
                else:
                    stable_track_count = 0
            else:
                 rospy.sleep(0.05)
            
            # 需要超时或手动中断？
            rospy.sleep(0.05)
            
        # 执行放置
        rospy.loginfo("正在放置物块...")
        # 放下机械臂
        target = self.ik.setPitchRanges((0, round(self.y_dis + 0.04, 4), -0.05), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1000, (
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, self.x_dis)
            ))
        rospy.sleep(1.5)
        
        # 张开爪子
        bus_servo_control.set_servos(self.joints_pub, 500, ((1, 150),))
        rospy.sleep(0.8)
        
        # 抬起机械臂
        bus_servo_control.set_servos(self.joints_pub, 1000, (
            (1, 150), (2, 500), (3, 80), (4, 825), (5, 625), (6, 500)
        ))
        rospy.sleep(1.5)
        self.reset_arm()
        return True

    def run_task(self):
        rospy.loginfo("验证任务开始：抓取红色物块 -> 放置到 Tag 1")
        
        # 1. 初始化与姿态
        self.init_robot()
        
        # 2. 移动到工作区（向前3秒）
        rospy.loginfo("正在移动到工作区...")
        self.move_chassis(60, 90, 0) # 慢速
        rospy.sleep(5.0) 
        self.stop_chassis()
        rospy.sleep(0.5)
        
        # 3. 抓取红色物块
        target_color = 'red'
        target_tag = 1
        
        rospy.loginfo(f"开始执行：抓取 {target_color} -> 放置到 Tag {target_tag}")
        
        success = False
        while not success:
            if self.search_color(target_color):
                if self.align_and_grasp(target_color):
                    if self.find_and_place_tag(target_tag):
                        rospy.loginfo("验证任务成功完成！")
                        success = True
                        break
                    else:
                        rospy.logwarn("放置失败，重试流程...")
                else:
                    rospy.logwarn("抓取失败，重试流程...")
            else:
                rospy.logwarn("未找到红色物块，重试流程...")
            
            rospy.sleep(1.0)
            
        rospy.loginfo("程序结束")
        self.stop_chassis()

    def start(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        task = VerifyRedBlockTask()
        task.start()
    except rospy.ROSInterruptException:
        pass
