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

class LaneTransportTask:
    def __init__(self):
        rospy.init_node('lane_transport_task', log_level=rospy.INFO)
        
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
        
        # AprilTag检测器
        # 使用 tag25h9
        # 注意：armpi_pro.apriltag 的 Detector 构造函数接受 options 对象
        try:
            options = apriltag.DetectorOptions(families='tag25h9')
            self.detector = apriltag.Detector(options, searchpath=apriltag._get_demo_searchpath())
            rospy.loginfo("AprilTag Detector initialized with families='tag25h9'")
        except Exception as e:
            rospy.logerr(f"Failed to initialize detector with options: {e}. Fallback to default.")
            self.detector = apriltag.Detector(searchpath=apriltag._get_demo_searchpath())
        
        # 检测状态
        self.mode = 'None' 
        self.target_color = 'None'
        self.target_tag_id = None
        self.detect_result = None
        
        # PID控制器 (参考 intelligent_grasp_node)
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

    #物理复位，机器初始化
    def init_robot(self):
        rospy.loginfo("正在初始化机器人...")
        self.stop_chassis()
        self.set_rgb('black')
        self.reset_arm()
    
    #机械臂初始化  reset_arm
    def reset_arm(self):
        # 归位
        target = self.ik.setPitchRanges((0, 0.15, 0.10), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1500, (
                (1, 150), (2, 500), # 爪子张开 (150)，保持放松
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

    #停止命令
    def stop_chassis(self):
        self.set_velocity.publish(0, 90, 0)

    #移动命令，也就是self.set_velocity.publish(speed, direction, omega)
    def move_chassis(self, speed, direction, omega):
        # direction: 90=前, 180=左, 270=后, 0=右
        self.set_velocity.publish(speed, direction, omega)

    def servo_state_callback(self, msg):
        for state in msg.servo_states:
            self.servo_positions[state.id] = state.position

    def get_servo_position(self, servo_id):
        return self.servo_positions.get(servo_id, None)

#下面是图像处理部分
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
        hsv_ranges = {
            'red': [((0, 43, 46), (10, 255, 255)), ((156, 43, 46), (180, 255, 255))],
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
                
                # 改为画轮廓：需要将轮廓点从 size_m 映射回 img_w/img_h
                # area_max_contour 的形状是 (N, 1, 2)
                # 我们创建一个新的轮廓数组，把每个点坐标都放大
                scale_x = img_w / self.size_m[0]
                scale_y = img_h / self.size_m[1]
                
                # 使用 numpy 广播机制快速缩放
                scaled_contour = area_max_contour.copy()
                scaled_contour[:, 0, 0] = area_max_contour[:, 0, 0] * scale_x
                scaled_contour[:, 0, 1] = area_max_contour[:, 0, 1] * scale_y
                scaled_contour = scaled_contour.astype(np.int32)
                
                # 画出轮廓 (线宽2) - 使用黄色 (0, 255, 255) 以高对比度显示
                cv2.drawContours(img, [scaled_contour], -1, (0, 255, 255), 2)
                
                # 顺便画个中心点，方便看对齐情况 - 还是用本色画中心点，或者也用黄色
                cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1) 
                
                color_id = {'red':1, 'green':2, 'blue':3}.get(color, 0)
                result = {'center_x': cx, 'center_y': cy, 'data': color_id, 'area': area_max}
        return img, result



    def apriltag_detect(self, img, target_id):
        img_copy = img.copy()
        img_h, img_w = img.shape[:2]
        frame_resize = cv2.resize(img_copy, self.size_m, interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray, return_image=False)
        result = None
        if len(detections) != 0:
            for detection in detections:
                tag_id = int(detection.tag_id)
                if target_id is not None and tag_id != target_id: continue
                
                object_center_x = int(Misc.map(detection.center[0], 0, self.size_m[0], 0, img_w))
                object_center_y = int(Misc.map(detection.center[1], 0, self.size_m[1], 0, img_h))
                cv2.putText(img, str(tag_id), (object_center_x, object_center_y), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 255, 255], 6)
                
                # 绘制方框
                corners = detection.corners
                for i in range(4):
                    j = (i + 1) % 4
                    pt1 = (int(Misc.map(corners[i][0], 0, self.size_m[0], 0, img_w)),
                           int(Misc.map(corners[i][1], 0, self.size_m[1], 0, img_h)))
                    pt2 = (int(Misc.map(corners[j][0], 0, self.size_m[0], 0, img_w)),
                           int(Misc.map(corners[j][1], 0, self.size_m[1], 0, img_h)))
                    cv2.line(img, pt1, pt2, [0, 255, 255], 2)
                
                result = {'center_x': object_center_x, 'center_y': object_center_y, 'data': tag_id}
                break # 找到一个就返回
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
            elif self.mode == 'apriltag':
                processed_img, res = self.apriltag_detect(cv2_img, self.target_tag_id)
            self.detect_result = res
        try:
            ros_image.data = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB).tobytes()
            self.image_pub.publish(ros_image)
        except: pass

    #下面是右转（turn_90_right）/左转（turn_90_left）/180度旋转（turn_180_right）的逻辑：
    # --- 动作原语 ---
    def turn_90_left(self):
        rospy.loginfo("左转 90度...")
        # 假设 omega=1.5 rad/s 对应某PWM，需调优
        # 这里的 omega 单位通常是 rad/s 或 比例
        # 之前的代码用 0.5 旋转 1.5s 约 90度? 
        # integrated_transport: 0.5 * 1.5s = ?
        self.move_chassis(0, 0, 0.5) # 正omega为左转
        rospy.sleep(2.2) # 估算值，需实测微调
        self.stop_chassis()
        rospy.sleep(0.5)



    
    def turn_90_right(self):
        rospy.loginfo("右转 90度...")
        self.move_chassis(0, 0, -0.5) # 负omega为右转
        rospy.sleep(2.2) # 估算值，需实测微调
        self.stop_chassis()
        rospy.sleep(0.5)
        
    def turn_180_right(self):
        rospy.loginfo("右转 180度...")
        self.move_chassis(0, 0, -0.5)
        rospy.sleep(4.4) # 2倍90度时间，需实测微调
        self.stop_chassis()
        rospy.sleep(0.5)

    # --- 任务逻辑 ---
    # 智能抓取
    def intelligent_grasp(self, color):
        """
        参考 intelligent_grasp_node.py 的逻辑
        使用 机械臂底座(ID 6) 进行X轴对齐，机械臂伸缩(ID 3/4/5) 进行Y轴对齐
        前提：底盘已经面向物体且大致对齐
        """
        rospy.loginfo(f"开始智能抓取: {color}")
        self.set_rgb(color)
        
        # 1. 准备姿态 (参考 intelligent_grasp_node initMove)
        bus_servo_control.set_servos(self.joints_pub, 1000, ((1, 120),)) # 张开爪子 (120)
        rospy.sleep(0.5)
        
        # 机械臂移动到中间寻找位置 (ik target: 0, 0.15, 0.03)
        self.x_dis = 500
        self.y_dis = 0.15
        
        target = self.ik.setPitchRanges((0, 0.15, 0.05), -180, -180, 0)
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
            if time.time() - start_time > 20: # 超时
                rospy.logwarn("Grasp timed out")
                return False
                
            res = None
            with self.lock:
                res = self.detect_result
                
            if res:
                cx = res['center_x']
                cy = res['center_y']
                
                # PID 更新
                # X轴 (ID 6)
                if abs(cx - self.centreX) < 10:
                    self.x_pid.SetPoint = cx
                else:
                    self.x_pid.SetPoint = self.centreX
                self.x_pid.update(cx)
                dx = self.x_pid.output
                self.x_dis += int(dx)
                self.x_dis = max(200, min(800, self.x_dis)) # 限位
                
                # Y轴 (Distance)
                if abs(cy - self.centreY) < 10:
                    self.y_pid.SetPoint = cy
                else:
                    self.y_pid.SetPoint = self.centreY
                self.y_pid.update(cy)
                dy = self.y_pid.output
                self.y_dis += dy
                self.y_dis = max(0.12, min(0.28, self.y_dis)) # 限位
                
                # 执行机械臂移动
                target = self.ik.setPitchRanges((0, round(self.y_dis, 4), 0.04), -180, -180, 0)
                if target:
                    servo_data = target[1]
                    bus_servo_control.set_servos(self.joints_pub, 20, (
                        (3, servo_data['servo3']), (4, servo_data['servo4']),
                        (5, servo_data['servo5']), (6, self.x_dis)
                    ))
                    if len(target) > 2:
                         offset_y = Misc.map(target[2], -180, -150, -0.04, 0.03)
                
                # 底盘辅助对齐 (Chassis Auto-Alignment)
                chassis_moving = False
                
                # 修改：放宽底盘触发阈值，防止震荡 (Hysteresis)
                # 降低移动速度 (30 -> 20)
                
                # 1. X轴对齐 (舵机6范围: 0-1000, 舒适区扩大到: 250-750)
                if self.x_dis > 750: # 机械臂过右 -> 底盘右移
                    self.move_chassis(20, 0, 0)
                    chassis_moving = True
                elif self.x_dis < 250: # 机械臂过左 -> 底盘左移
                    self.move_chassis(20, 180, 0)
                    chassis_moving = True
                
                # 2. Y轴对齐 (距离范围: 0.12-0.28, 舒适区扩大到: 0.13-0.27)
                if not chassis_moving: # 优先X轴，避免同时移动太复杂
                    if self.y_dis > 0.27: # 机械臂过伸 -> 底盘前进
                        self.move_chassis(20, 90, 0)
                        chassis_moving = True
                    elif self.y_dis < 0.13: # 机械臂过缩 -> 底盘后退
                        self.move_chassis(20, 270, 0)
                        chassis_moving = True
                
                if not chassis_moving:
                    self.stop_chassis()
                else:
                    stable_track_count = 0 # 移动时重置稳定计数

                # 稳定性检查
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
                # 如果丢失目标，也许需要底盘微调？
                # 这里简单处理：如果完全丢失，尝试小范围摆动ID6或底盘
                rospy.sleep(0.05)
                
            rospy.sleep(0.05)
            
        # 执行抓取 (完全参考 intelligent_grasp_node.py move 函数中的逻辑)
        rospy.loginfo("执行抓取动作...")
        
                # 1. 机械臂向下伸 (target: 0, y_dis + offset_y, -0.08)
        # y_dis 稍微减小一点，确保爪子能够很好的抓住
        target = self.ik.setPitchRanges((0, round(self.y_dis + offset_y -0.01, 4), -0.08), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1000, (
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, self.x_dis)
            ))
        rospy.sleep(1.5)
        
        # 2. 闭合机械爪 (550) - 稍微紧一点
        bus_servo_control.set_servos(self.joints_pub, 500, ((1, 550),)) 
        rospy.sleep(0.8)
        
        # 3. 机械臂抬起来 (固定姿态)
        bus_servo_control.set_servos(self.joints_pub, 1500, (
            (1, 550), (2, 500), (3, 80), (4, 825), (5, 625), (6, 500)
        ))
        rospy.sleep(1.5)

        # 4. 视觉抓取检测
        # 抬起后，如果抓住了物块，物块应该在摄像头视野内且面积较大
        # 且位置应该在下方居中位置 (因为爪子在摄像头下方)
        grasped = False
        check_start = time.time()
        while time.time() - check_start < 1.0: # 检查1秒
            res = None
            with self.lock:
                res = self.detect_result
            
            if res:
                area = res.get('area', 0)
                cx = res.get('center_x', 0)
                cy = res.get('center_y', 0)
                
                # 综合判断条件：
                # 1. 面积足够大 (抓在手里离得很近)
                # 2. 位置靠下 (cy > 300) - 假设480高
                # 3. 位置居中 (abs(cx - 320) < 150) - 假设640宽
                
                if area > 1000 and cy > 300 and abs(cx - 320) < 150:
                    rospy.loginfo(f"视觉抓取检测成功: Area={area}, Pos=({cx},{cy})")
                    grasped = True
                    break
            rospy.sleep(0.1)
            
        if not grasped:
            rospy.logwarn("视觉抓取检测失败 (未检测到手中物块)")
            # 松开并返回失败
            bus_servo_control.set_servos(self.joints_pub, 500, ((1, 150),))
            rospy.sleep(0.5)
            self.reset_arm()
            return False

        return True



    # 智能放置
    def intelligent_place(self, tag_id):
        """
        放置逻辑：
        1. 保持低姿态 (Z=0.05)。
        2. 识别Tag并进行对齐 (X轴底盘, Y轴机械臂)。
        3. 对齐完成后直接放置。
        """
        rospy.loginfo(f"开始放置到 Tag {tag_id}")
        
        # 1. 低位姿态 (Z=0.05)
        self.x_dis = 500
        self.y_dis = 0.15
        target = self.ik.setPitchRanges((0, 0.15, 0.03), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1500, (
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, self.x_dis)
            ))
        rospy.sleep(1.5)
        
        self.x_pid.clear()
        self.y_pid.clear()
        
        aligned = False
        start_time = time.time()
        
        # 记录最后一次看到的Tag信息
        last_seen_x = 320 # 默认中心
        last_seen_cy = 410 # 默认目标点
        last_seen_time = 0
        
        while not aligned:
            if time.time() - start_time > 20: 
                rospy.logwarn("Place alignment timed out")
                break
                
            res = None
            with self.lock:
                res = self.detect_result
            
            if res and res['data'] == tag_id:
                cx = res['center_x']
                cy = res['center_y']
                
                # 1. X轴修正 (底盘)
                err_x = self.centreX - cx
                vx = 0
                # 只要有误差就修正，不再设置死区阈值
                # 限制最小速度以保证电机能转动 (假设最小有效PWM对应速度是10-15)
                if abs(err_x) > 2: # 仅保留极小的抖动过滤
                    vx = int(min(30, max(15, abs(err_x) * 0.15))) # 提高一点比例系数
                    if err_x > 0: dir_x = 180
                    else: dir_x = 0
                    self.move_chassis(vx, dir_x, 0)
                else:
                    self.stop_chassis()
                    
                # 2. Y轴修正 (机械臂)
                # 目标像素点 target_cy 设定为 410 (低姿态Z=0.03时的标定值，这里Z=0.05也差不多)
                target_cy = 410
                self.y_pid.SetPoint = target_cy
                self.y_pid.update(cy)
                dy = self.y_pid.output
                self.y_dis += dy
                self.y_dis = max(0.12, min(0.28, self.y_dis))
                
                # 移动机械臂 (Z保持0.05)
                target = self.ik.setPitchRanges((0, round(self.y_dis, 4), 0.03), -180, -180, 0)
                if target:
                    servo_data = target[1]
                    bus_servo_control.set_servos(self.joints_pub, 20, (
                        (3, servo_data['servo3']), (4, servo_data['servo4']),
                        (5, servo_data['servo5']), (6, self.x_dis)
                    ))
                    
                if vx == 0 and abs(dy) < 0.005: 
                     rospy.loginfo("Tag对齐完成")
                     # 利用惯性再延时0.1s，让车再滑行一点点，或者让系统再稳定确认一下
                     rospy.sleep(0.1) 
                     
                     aligned = True
                     self.stop_chassis()
                     break
                     
                # 记录最后一次看到的Tag信息，用于丢失后的盲走
                last_seen_x = cx
                last_seen_cy = cy
                last_seen_time = time.time()
                
            else:
                # 没找到Tag
                if time.time() - last_seen_time < 1.0:
                    # 如果是刚丢的，说明是被机械臂遮挡了
                    # 按照用户思路：取最后一张图的偏差，执行最后一次修正移动（开环冲刺）
                    
                    err_x_last = self.centreX - last_seen_x
                    rospy.loginfo(f"Tag遮挡，执行最后冲刺修正 (Last Err={err_x_last})")
                    
                    # 1. 计算最后需要的修正量
                    # 假设：误差 100 像素 对应 约 0.5 秒的移动 (速度30)
                    # 这个比例系数需根据实际调节: time = abs(err) * k
                    # 限制最大冲刺时间，防止暴走
                    
                    if abs(err_x_last) > 5: # 只有误差大于阈值才冲刺
                        dash_speed = 20
                        # 简单的P控制转时间：像素误差 -> 移动时间
                        # 经验值：100px 偏差大概对应车身横移 3-5cm，需要 0.3-0.5s
                        dash_time = abs(err_x_last) * 0.005 
                        dash_time = max(0.1, min(0.8, dash_time)) # 限制在 0.1s - 0.8s 之间
                        
                        if err_x_last > 0: dash_dir = 180 # 目标在左，车往左
                        else: dash_dir = 0 # 目标在右，车往右
                        
                        rospy.loginfo(f"冲刺: Speed={dash_speed}, Dir={dash_dir}, Time={dash_time:.2f}s")
                        self.move_chassis(dash_speed, dash_dir, 0)
                        rospy.sleep(dash_time)
                        self.stop_chassis()
                    else:
                        rospy.loginfo("偏差极小，无需冲刺")
                        self.stop_chassis()
                    
                    # 冲刺完成后，直接认为对齐成功
                    aligned = True
                    break
                else:
                    self.stop_chassis()
                    rospy.sleep(0.05)
                
            rospy.sleep(0.05)
            
        # 放下
        rospy.loginfo(f"执行放置 (y_dis={self.y_dis:.4f})...")
        # 放置高度 Z=-0.05
        target = self.ik.setPitchRanges((0, round(self.y_dis, 4), -0.05), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1000, (
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, self.x_dis)
            ))
        rospy.sleep(1.5)
        bus_servo_control.set_servos(self.joints_pub, 500, ((1, 150),)) # 张开
        rospy.sleep(0.8)
        
        # 抬起归位
        self.reset_arm()
        return True



    #核心运行逻辑，按照任务顺序执行
    def run_task(self):
        rospy.loginfo("比赛任务开始：停放区 -> 货物区(左) -> 目标区(右) -> 停放区")
        
        # 任务顺序：红->Tag1, 蓝->Tag2, 绿->Tag3
        tasks = [('red', 0), ('blue', 1), ('green', 2)]
        
        # (1) 启动小车程序，将小车从停放区移动到货物摆放区
        rospy.loginfo("(1) 从停放区移动到货物摆放区...")
        self.move_chassis(50, 90, 0)
        start_drive_time = 5.0 # 记录初始前进时间 (修改为5s)
        rospy.sleep(start_drive_time) 
        self.stop_chassis()
        rospy.sleep(0.5)
        
        for i, (color, tag_id) in enumerate(tasks):
            rospy.loginfo(f"--- 任务 {i+1}: 抓取 {color} -> 放置到 Tag {tag_id} ---")
            
            # 这里的逻辑假设：
            # 每次放置完后，小车都处于"货物摆放区"的通道上，准备进行下一次搜索。
            # 如果三个物块在同一个位置（堆叠或紧邻），则不需要额外的前进。
            # 如果物块分散在不同工位，则需要前进。
            # 根据题目 "(3) 小车返回货物摆放区"，暗示是往返或原地操作。
            # 假设场景：
            # 巷道左边是货物架，右边是目标架。
            # 小车停在中间。
            
            # (2)/(3)/(4) 搜索积木
            rospy.loginfo(f"正在搜索 {color} 积木 (左侧)...")
            
            # 左转 90度 面向货物区
            self.turn_90_left()
            
            # 左转后，小车可能离物块还有一定距离（在巷道中间）
            # 需要向前移动一点，以便机械臂能够着物块
            rospy.loginfo("左转后，向前靠近物块...")
            self.move_chassis(50, 90, 0)
            rospy.sleep(1.0) # 假设向前1秒能靠近
            self.stop_chassis()
            rospy.sleep(0.5)
            
            # 开启颜色识别
            with self.lock: 
                self.mode = 'color'
                self.target_color = color
            
            # 执行抓取
            grasp_success = False
            # 搜索策略：中 -> 左 -> 右 -> 中 -> 左
            search_actions = ['center', 'left', 'right', 'center', 'left']
            
            for i_try, pos in enumerate(search_actions):
                rospy.loginfo(f"尝试抓取 {color} (位置: {pos})...")
                if self.intelligent_grasp(color):
                    rospy.loginfo(f"成功抓取 {color} 积木")
                    grasp_success = True
                    break
                else:
                    if i_try < len(search_actions) - 1:
                        next_pos = search_actions[i_try + 1]
                        rospy.logwarn(f"抓取失败，移动到 {next_pos} 继续搜索...")
                        
                        if pos == 'center' and next_pos == 'left':
                            # 左移
                            self.move_chassis(30, 180, 0)
                            rospy.sleep(0.6)
                        elif pos == 'left' and next_pos == 'right':
                            # 左 -> 右 (跨度大)
                            self.move_chassis(30, 0, 0)
                            rospy.sleep(1.2)
                        elif pos == 'right' and next_pos == 'center':
                            # 右 -> 中
                            self.move_chassis(30, 180, 0)
                            rospy.sleep(0.6)
                        elif pos == 'center' and next_pos == 'left': # 第5次
                             self.move_chassis(30, 180, 0)
                             rospy.sleep(0.6)
                            
                        self.stop_chassis()
                        rospy.sleep(0.5)

            if grasp_success:
                # 抓取后，为了转身不撞到架子，先退回到巷道中心
                rospy.loginfo("抓取后后退...")
                self.move_chassis(50, 270, 0)
                rospy.sleep(1.0) # 后退同样的时间
                self.stop_chassis()
                rospy.sleep(0.5)
                
                # 运送到目标位置区 (右边)
                # 右转 180度
                self.turn_180_right()
                
                # 右转后，可能也需要向前靠近 Tag (如果 Tag 在墙上)
                rospy.loginfo("右转后，向前靠近 Tag...")
                self.move_chassis(50, 90, 0)
                rospy.sleep(1.0)
                self.stop_chassis()
                rospy.sleep(0.5)
                
                # 放置到 Tag (得分20分)
                rospy.loginfo(f"正在运送到 Tag {tag_id} (右侧)...")
                with self.lock:
                    self.mode = 'apriltag'
                    self.target_tag_id = tag_id
                    
                if self.intelligent_place(tag_id):
                    rospy.loginfo(f"成功放置到 Tag {tag_id}")
                else:
                    rospy.logwarn(f"放置 {color} 失败")
                
                # 放置后，退回巷道中心
                rospy.loginfo("放置后后退...")
                self.move_chassis(50, 270, 0)
                rospy.sleep(1.0)
                self.stop_chassis()
                rospy.sleep(0.5)
                    
                # 放置完成后，返回货物摆放区（即回到巷道中间并面向前方/货物区）
                # 题目说 "小车返回货物摆放区"，意味着回到初始状态准备下一次抓取
                # 当前面向右侧(目标区)，需要左转 90度 回到巷道正向，或者左转180度直接面向货物区？
                # 如果下一个物块在同一个位置，直接左转180度面向货物区最快。
                # 如果需要移动到下一个工位，则左转90度回正 -> 前进 -> 左转90度。
                
                # 假设所有物块在同一区域，或者小车需要在巷道中微调
                # 为了通用性，我们先回正 (面向巷道前方)
                self.turn_90_left()
                
                # 如果不是最后一个任务，可能需要调整位置?
                # 题目隐含意思是往返跑? 或者就在原地?
                # "小车返回货物摆放区" -> 可能是指从 Tag区(右) 回到 货物区(左) 的动作
                # 我们的逻辑是：右侧放完 -> 回正 -> (可选前进) -> 左转抓取
                
                if i < len(tasks) - 1:
                     rospy.loginfo("准备进行下一轮任务...")
                     # 如果需要前进到下一个工位，在这里添加代码
                     self.move_chassis(50, 90, 0)
                     rospy.sleep(1.0)
                     self.stop_chassis()
            else:
                rospy.logwarn(f"抓取 {color} 失败 (已重试5次)")
                # 即使失败也要转回去，保持状态一致
                self.turn_90_right() 
                
            rospy.sleep(1.0)
            
        # (5) 小车返回停放区
        rospy.loginfo("(5) 所有任务完成，返回停放区...")
        
        # 最后一次放置完成后，小车已经退回到了巷道中心，且面向前方(Turn 90 Left后)
        # 如果要车头朝前倒着开回去：
        # 1. 确保车头是朝前的 (已经是了)
        # 2. 执行后退动作
        
        rospy.loginfo("车头朝前，向后行驶返回停放区...")
        self.move_chassis(50, 270, 0) # 270是向后
        rospy.sleep(start_drive_time+1) # 使用与开始时相同的时间
        self.stop_chassis()
        
        rospy.loginfo("已到达停放区")
        self.stop_chassis()
        
    def start(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        task = LaneTransportTask()
        task.start()
    except rospy.ROSInterruptException:
        pass
