#!/usr/bin/python3
# coding=UTF-8

from fabric2 import Connection
import rospy
from geometry_msgs.msg import Twist, Quaternion, PoseWithCovarianceStamped
from cv_bridge import CvBridge 
from tf.transformations import quaternion_from_euler,euler_from_quaternion
import tf
import actionlib
from actionlib_msgs.msg import *
from sensor_msgs.msg import LaserScan,Image
from move_base_msgs.msg import MoveBaseAction,MoveBaseGoal
import math
import numpy as np
import os
import cv2
import time
import copy
import threading
from threading import Thread
from ultralytics import YOLO
from pathlib import Path
import glob
from queue import Queue
import csv
import torch

def filter(boxes, window_x=0.4, window_y=0.4, box_x=4, box_y=3, window_span=1.2, box_span=1.25):
    """
    返回3个值,第一个值为该照片对应房间是否存在含识别卡的箱子,
    第二个值为GOOD识别卡数量,第三个值为BAD识别卡数量
    """
    rec_list = [boxes.xywhn.numpy(), boxes.cls.numpy()]
    
    if len(rec_list[0]) == 0:
        # 如果没有识别到物体,返回False
        return False, 0, 0

    BOX_VALID = False
    CENTRAL_BIAS = 100
    CENTRAL_OBJECT_INDEX = 0
    BOX_BAD_COUNT = 0
    BOX_GOOD_COUNT = 0

    # 2. 如果识别到物体,获取离中心点最近的,位于识别Window窗体内的识别卡
    for index, (obj_box, obj_cls) in enumerate(zip(rec_list[0], rec_list[1])):
        obj_central_x, obj_central_y = obj_box[0], obj_box[1]

        # 判断识别卡中心点是否位于Window窗体内
        if (0.5 - window_x / 2) < obj_central_x < (0.5 + window_x / 2) and \
        (0.5 - window_y / 2) < obj_central_y < (0.5 + window_y / 2):
            BOX_VALID = True

            # 判断识别卡与照片中心点距离
            my_distance = (obj_central_x - 0.5) ** 2 + (obj_central_y - 0.5) ** 2

            # 获取离中心点最近的物体
            if my_distance < CENTRAL_BIAS:
                CENTRAL_BIAS = my_distance
                CENTRAL_OBJECT_INDEX = index

    if BOX_VALID:
        # 获取中心点最近的物体的坐标和尺寸
        central_obj_box = rec_list[0][CENTRAL_OBJECT_INDEX]
        BOX_XYWHN = [
            central_obj_box[0],
            central_obj_box[1],
            central_obj_box[2] * box_x * box_span,
            central_obj_box[3] * box_y * box_span
        ]

        for obj_box, obj_cls in zip(rec_list[0], rec_list[1]):
            if (BOX_XYWHN[0] - BOX_XYWHN[2] / 2) < obj_box[0] < (BOX_XYWHN[0] + BOX_XYWHN[2] / 2) and \
            (BOX_XYWHN[1] - BOX_XYWHN[3] / 2) < obj_box[1] < (BOX_XYWHN[1] + BOX_XYWHN[3] / 2):
                if -0.01 < obj_cls - 1 < 0.01:
                    BOX_GOOD_COUNT += 1
                elif -0.01 < obj_cls < 0.01:
                    BOX_BAD_COUNT += 1
                else:
                    print("???????我希望永远都不会看到这个输出")

        return True, BOX_GOOD_COUNT, BOX_BAD_COUNT

    return False, 0, 0

class Model:
    def __init__(self, model_path, model_input_path,pic_recognize_count,num_workers=4):
        # self.model = YOLO(model_path)
        # this will bring some threading issues
        self.pic_recognize_lock = threading.Lock()
        self.pic_recognize_count = pic_recognize_count

        self.model_path = model_path
        self.model_input_path = model_input_path
        self.model_input_buffer = Queue()
        self.model_output_buffer = Queue()

        self.condition = threading.Condition()
        self.threads_pool = []
        for i in range(0,num_workers):
            worker = Thread(target=self.work)
            worker.daemon = True
            worker.start()
            self.threads_pool.append(worker)
            

        self.PLANE_1 = False
        self.PLANE_2 = False
        self.PLANE_FINISH = False
        self.CAR_RECOGNIZE = False
        self.CAR_FINISH = False

    def plane_run(self):
        while True:
            time.sleep(3)
            check = self.plane_task()
            if check:
                with self.condition:
                    self.condition.notify_all()
        
            if self.PLANE_FINISH:
                print("finish plane task")
                return True
    
    def car_run(self):
        while True:
            time.sleep(5)
            if self.CAR_RECOGNIZE:
                self.CAR_RECOGNIZE = False
                with self.condition:
                    self.condition.notify_all()

            if self.CAR_FINISH:
                print("finish car task")
                return True

    def work(self):
        while True:
            with self.condition:
                self.condition.wait()
                while not self.model_input_buffer.empty():
                    my_list = self.model_input_buffer.get()
                    img_name = my_list[0]
                    #print("recognize:!!!",img_name,my_list[1],type(my_list[1]))
                    good_counts,bad_counts = self.recognize(my_list[1])
                    csv_line = (img_name,good_counts,bad_counts)
                    with self.pic_recognize_lock:
                        self.pic_recognize_count -= 1
                    self.model_output_buffer.put(csv_line)
                    #print("APPEND A TXT LINE:",txt_line)
        


    def recognize(self,model_input):
        """_summary_

        Args:
            input (_type_): 可以是cv2.imread()返回的图片Mat(numpy ndarray数组),也可以是图片路径,还可以是一些别的东西.

        Returns:
            _type_: 推理结果list,例如['1 good 1 bad']
        """
        model = YOLO(self.model_path)
        results = model(model_input)
        # print("notice",type(results))
        res = results[0].boxes

        # if len(results) == 1 :
        #     pass
        #     #print("yesyesyes finish recognize recognize!")

        # else:
        #     print("it seems that you give me some wrong inputs.")

        ret,good_num,bad_num = filter(res)
        return good_num,bad_num
    


    # 仅供飞机使用的方法
    def plane_task(self):
        file_names = os.listdir(self.model_input_path)
        if 'finish4.txt' in file_names and self.PLANE_1 == False:
            print("FINISH 4 SIGNAL!!!")
            image_paths = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_2' + '*')))
            print("image_paths:",image_paths)
            for image_name in image_paths:
                image_result = cv2.imread(image_name)
                self.append(image_name,image_result)
                print("succesfully append an image to model_input_buffer")
            self.PLANE_1 = True
            #print(self.model_output_buffer.qsize())
            return True

        elif 'finish8.txt' in file_names and self.PLANE_2 == False:
            print("FINISH 8 SIGNAL!!!")
            image_paths = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_1' + '*')))
            print("image paths for finish 8:",image_paths)
            for image_name in image_paths:
                image_result = cv2.imread(image_name)
                self.append(image_name,image_result)
            self.PLANE_2 = True
            #print(self.model_output_buffer.qsize())
            return True 
        
        else:
            return False
        
            
        # if '1440x1088_1-top.txt' in file_names and self.top_1 == False:
        #     image_paths = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_1-top' + '*')))
        #     for i in image_paths:
        #         image = cv2.imread(i)
        #         self.append(image,i)
        #     self.top_1 = True
        #     print(self.model_output_buffer.qsize())
        #     return True

        # if '1440x1088_1-middle.txt' in file_names and self.middle_1 == False:
        #     image_paths = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_1-middle' + '*')))
        #     for i in image_paths:
        #         image = cv2.imread(i)
        #         self.append(image,i)
        #     self.middle_1 = True
        #     print(self.model_output_buffer.qsize())
        #     return True
        # return False


            
    def append(self,image_name,image):
        self.model_input_buffer.put([image_name,image])
        return True
    

    # def save(self):
    #     print("enter save def")
    #     while self.model_output_buffer.qsize() !=0:
    #         print(self.model_output_buffer.qsize())
    #         txt_line = self.model_output_buffer.get()
    #         print("this is a saving txt line?",txt_line)
    #         with open(self.model_output_path,'a') as f:
    #             f.write(','.join(txt_line) + '\n')
    #             print("Save Result:",txt_line)     

    #     return True

class CsvWriter:
    def __init__(self,output_file_path,car_model_buffer,plane_model_buffer):
        self.car_model_buffer = car_model_buffer
        self.plane_model_buffer = plane_model_buffer
        self.output_file_path = output_file_path
        
    def write(self):
        csv_file = self.output_file_path
        with open(csv_file,mode='w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('房间','好人','坏人'))
            while self.car_model_buffer.qsize() > 0:
                csv_line = self.car_model_buffer.get()
                writer.writerow(csv_line)
        
            while self.plane_model_buffer.qsize()>0:
                csv_line = self.plane_model_buffer.get()
                writer.writerow(csv_line)
        f.close()


class Nav:
    def __init__(self):
        rospy.init_node('nav', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.laser_subscriber = rospy.Subscriber('/scan', LaserScan, self.laser_callback,queue_size=1)
        self.image_raw = rospy.Subscriber("ep_cam/image_raw", Image, self.camera_cb, queue_size=1)
        self.cv_bridge = CvBridge()
        self.laser_data = None
        self.image = None
        self.take_picture_need = False
        self.image_lock = threading.Lock()

        self.plane_model = Model("/home/tta/Project_ws/src/rmep_nav/scripts/model/720car-1.pt",
        model_input_path='/home/tta/Project_ws/src/rmep_nav/scripts/image_from_flight/',
        pic_recognize_count=16)

        self.car_model = Model("/home/tta/Project_ws/src/rmep_nav/scripts/model/720car-2.pt",
        model_input_path = '/home/tta/Project_ws/src/rmep_nav/scripts/image_from_car/',
        pic_recognize_count=8)
        
        self.car_model_manager = Thread(target=self.car_model.car_run)
        self.car_model_manager.start()
        self.plane_model_manager = Thread(target=self.plane_model.plane_run)
        self.plane_model_manager.start()

        self.rate = rospy.Rate(150)
        self.tf_listener = tf.TransformListener() 
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        while(not self.move_base.wait_for_server(rospy.Duration(1.0))):
            continue
        self.nav_for_projecte()
        
    def nav_for_projecte(self):
        self.move_to_initial_position(0.75, 0.85)
        self.move_to_initial_position(0.75, 0.85)
        self.initial_pose_for_amcl(0,0,0,math.pi/2)
 
        self.is_go('/mnt/start.txt')
        rospy.sleep(5)
        print("isgo!!!!!!!!!!!!!!")

        a = self.make_move_base_goal(0,2.75,0)
        self.move(a,0,False)
        rospy.sleep(1)

        self.go_picture_point(2,2.75,0,'2-9')
        self.go_picture_point(3.5,2.75,0,'2-10')
        self.go_picture_point(4.75,2.75,0,'2-11')
        self.go_picture_point(5.75,2.75,0,'2-12')


        #
        self.car_model.CAR_RECOGNIZE = True
        #

        b = self.make_move_base_goal(7.25,2.4,0)
        self.move(b,0,True)
        rospy.sleep(0.5)
        self.turn_ang(-1,math.pi/2)
        rospy.sleep(0.5)
        self.adjust_pose(math.pi/2)
        rospy.sleep(0.5)


        self.has_arrive('/mnt/arrival.txt')
        self.is_go('/mnt/start_again.txt')

        self.turn_ang(1,math.pi)
        rospy.sleep(1)

        c = self.make_move_base_goal(7.25,0.45,math.pi)
        self.move(c,math.pi,False)
        rospy.sleep(1)


        self.go_picture_point(2.0,0.45,0,'1-9')
        self.go_picture_point(3.5,0.45,0,'1-10')
        self.go_picture_point(4.75,0.45,0,'1-11')
        self.go_picture_point(5.75,0.45,0,'1-12')

        #
        self.car_model.CAR_RECOGNIZE = True
        #
    
        self.turn_ang(1,math.pi)
        rospy.sleep(1)


        destinition = self.make_move_base_goal(0.25,0.25,math.pi/2)
        self.move(destinition,math.pi/2,False)
        rospy.sleep(1)

        self.has_arrive('/mnt/finish.txt')

        print("start to finish 2222222222222222222222222222222222222222222222222222222222222222")
        self.finish()


    def go_picture_point(self,x,y,yaw,image_name):
        point = self.make_move_base_goal(x,y,yaw)
        self.move(point,yaw,True)
        self.take_picture(image_name)
    

    def camera_cb(self, msg):
        with self.image_lock:
            if self.take_picture_need:
                self.image = self.cv_bridge.imgmsg_to_cv2(msg,"bgr8")
            else:
                return
        
    def take_picture(self, image_name):
        self.take_picture_need = True
        while self.image is None:
            continue
        rospy.sleep(1)

        image = copy.deepcopy(self.image)
        # image = self.image
        # / 有无必要使用deepcopy？
        dist_coeffs = np.array([-0.0444954569665007, -0.00201770323876866, 0, 0, -0.0159863040899372])
        intrinsic = np.array([[624.302683276108, 0, 632.376721286788],
                        [0, 623.915644264347, 371.420934824104],
                        [0, 0, 1]])
        corrected_image = cv2.undistort(image, intrinsic, dist_coeffs)     
        # 可以注释掉的deepcopy，理论上是为了防止脏写，但是OS或cv2包可能已经进行了维护   
        #
        cv2.imwrite(f'/home/tta/Project_ws/src/rmep_nav/scripts/image_from_car/{image_name}.jpg',corrected_image)
        # 

        self.car_model.append(image_name,corrected_image)
        self.take_picture_need = False

        time.sleep(0.5)
        self.image = None
        
    def has_arrive(self,path):
        hostname = '192.168.31.110'
        username = 'root'
        password = '123456'
        filepath = path

        conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})

        conn.run(f"touch {filepath}")

    def is_go(self,path):
        hostname = '192.168.31.110'
        username = 'root'
        password = '123456'
        filepath = path

        conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})

        while True:
            try:
                time.sleep(0.5)
                result = conn.run('test -e {}'.format(filepath), hide=True)
                if result.ok: 
                    conn.run(f'rm {filepath}')
                    conn.close()
                    break
            except Exception as e:
                print("continue to check")
                continue

    def initial_pose_for_amcl(self,x,y,z,yaw):
        initialpose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        initial_pose = PoseWithCovarianceStamped()

        initial_pose.header.stamp = rospy.Time.now()
        initial_pose.header.frame_id = "map"
        
        initial_pose.pose.pose.position.x = x
        initial_pose.pose.pose.position.y = y
        initial_pose.pose.pose.position.z = z
        
        quaternion = quaternion_from_euler(0, 0, yaw) 
        initial_pose.pose.pose.orientation.x = quaternion[0]
        initial_pose.pose.pose.orientation.y = quaternion[1]
        initial_pose.pose.pose.orientation.z = quaternion[2]
        initial_pose.pose.pose.orientation.w = quaternion[3]

        initial_pose.pose.covariance = [0.25, 0, 0, 0, 0, 0,
                                        0, 0.25, 0, 0, 0, 0,
                                        0, 0, 0.25, 0, 0, 0,
                                        0, 0, 0, 0.068, 0, 0,
                                        0, 0, 0, 0, 0.068, 0,
                                        0, 0, 0, 0, 0, 0.068]
        start_time = rospy.Time.now() 
        while (rospy.Time.now() - start_time).to_sec() < 0.5:
            initialpose_pub.publish(initial_pose)
            self.rate.sleep()
        rospy.sleep(1)
        print("initialpose success!")


    def laser_callback(self, laser_data):
        self.laser_data = laser_data

    def move_to_initial_position(self, goal_left, goal_back):
        self.align_wall()
        retries_y = 0
        retries_x = 0
        while not rospy.is_shutdown():
            if self.laser_data:
                laser_data = self.laser_data
                left_distance = self.get_distance(laser_data,-90)
                back_distance = self.get_distance(laser_data,0)
                if left_distance == float('inf') or back_distance == float('inf'):
                    continue
                if abs(left_distance - goal_left) > 0.02 and retries_y < 3:
                    error_y = abs(left_distance - goal_left)
                    speed = - 0.1 * (left_distance - goal_left) / abs(left_distance - goal_left)  
                    self.go_linear_y(speed,error_y)
                    retries_y += 1
                    rospy.sleep(1)
                    continue
                if abs(goal_back - back_distance) > 0.02 and retries_x < 3:
                    error_x = abs(goal_back - back_distance)
                    speed = 0.1 * (goal_back - back_distance) / abs(goal_back - back_distance)
                    self.go_linear_x(speed,error_x)
                    retries_x += 1
                    rospy.sleep(1)
                    continue
                # self.stop()
                break
            self.rate.sleep()
        self.align_wall()
        laser_data = self.laser_data
        left_distance = self.get_distance(laser_data,-90)
        back_distance = self.get_distance(laser_data,0)
        print(left_distance,back_distance)

    def get_distance(self, laser_data, angle):
        angle_in_rad = math.radians(angle)
        dis = laser_data.ranges[int((angle_in_rad - laser_data.angle_min) / laser_data.angle_increment)]
        return dis

    def align_wall(self):
        count = 0
        while not rospy.is_shutdown():
            if self.laser_data:
                try:
                    laser_data = self.laser_data
                    THETA = math.pi / 180 * 30
                    b = self.get_distance(laser_data, -90)
                    a = self.get_distance(laser_data, -120)
                    alpha = -math.atan((a * math.cos(THETA) - b) / (a * math.sin(THETA)))
                    if b == float('inf') or a == float('inf') or alpha == np.nan:
                        continue
                    ab = b * math.cos(alpha)
                    print(alpha, ab)
                    if abs(alpha) > 0.01 and count < 3:
                        self.turn_ang(1,alpha)
                        count += 1
                        rospy.sleep(1)
                    else:
                        self.stop()
                        rospy.sleep(1)
                        print("??????????????")
                        break
                except Exception as e:
                    continue
            # self.rate.sleep()

    def stop(self):
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        vel_msg.linear.y = 0
        self.velocity_publisher.publish(vel_msg)

    def turn_ang(self, ang_speed, goal_rotation):
        twist_ang = Twist()
        twist_ang.angular.z = -ang_speed if goal_rotation > 0 else ang_speed
        twist_ang.linear.x = 0
        twist_ang.linear.y = 0
        target_rotation_time = math.fabs(goal_rotation / ang_speed)
        turn_start_time = rospy.Time.now()
        while (rospy.Time.now() - turn_start_time).to_sec() < target_rotation_time:
            self.velocity_publisher.publish(twist_ang)
            self.rate.sleep()
        twist_ang.angular.z = 0
        self.velocity_publisher.publish(twist_ang)
        
    def go_linear_x(self,linear_speed,goal_distance):
        target_linear_time = math.fabs(goal_distance /linear_speed)
        twist_linear = Twist()
        twist_linear.linear.x = linear_speed
        twist_linear.linear.y = 0
        twist_linear.angular.z = 0
        go_start_time = rospy.Time.now() 
        while (rospy.Time.now() - go_start_time).to_sec() < target_linear_time:
            self.velocity_publisher.publish(twist_linear)
            self.rate.sleep()
        twist_linear.linear.x = 0
        self.velocity_publisher.publish(twist_linear)

    def go_linear_y(self,linear_speed,goal_distance):
        target_linear_time = math.fabs(goal_distance /linear_speed)
        twist_linear = Twist()
        twist_linear.linear.x = 0
        twist_linear.linear.y = linear_speed
        twist_linear.angular.z = 0
        go_start_time = rospy.Time.now() 
        while (rospy.Time.now() - go_start_time).to_sec() < target_linear_time:
            self.velocity_publisher.publish(twist_linear)
            self.rate.sleep()
        twist_linear.linear.y = 0
        self.velocity_publisher.publish(twist_linear)


    def move(self, goal, target_yaw, is_xy_fix, timeout=30, retries=3):
        attempt = 0
        while attempt < retries:
            attempt += 1
            if attempt > 1:
                laser_data = np.array(self.laser_data)
                min_index = np.argmin(laser_data)
                rad = math.radians(min_index * laser_data.angle_increment)
                error = math.pi - rad
                if error > 0:
                    self.turn_ang(0.8,abs(error))
                else:
                    self.turn_ang(-0.8,abs(error))
                rospy.sleep(0.5)
                self.go_linear_x(0.1,0.1)
                rospy.sleep(0.5)
                rate = rospy.Rate(20)
                while not rospy.is_shutdown():
                    try:
                        (current_postion,rotation) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0)) 
                        rate.sleep()
                        break
                    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                        continue
                _1, _2, yaw = euler_from_quaternion(rotation)
                self.initial_pose_for_amcl(current_postion[0],current_postion[1],0,yaw)

            self.move_base.send_goal(goal)
            success = self.move_base.wait_for_result(rospy.Duration(timeout))        
            if success:
                state = self.move_base.get_state()
                if state == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo("Goal reached successfully on attempt %d", attempt)
                    rospy.sleep(0.5)
                    self.adjust_pose(target_yaw)
                    rospy.sleep(0.5)
                    if is_xy_fix == False:
                        return
                    rate = rospy.Rate(20)
                    while not rospy.is_shutdown():
                        try:
                            (current_postion,rotation) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0)) 
                            rate.sleep()
                            break
                        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                            continue
                    current_position_x = current_postion[0]
                    current_position_y = current_postion[1]
                    error_x = current_position_x - goal.target_pose.pose.position.x
                    error_y = current_position_y - goal.target_pose.pose.position.y
                    if abs(error_x) > 0.025:
                        self.go_linear_x(-0.1*abs(error_x)/(error_x),abs(error_x))
                        rospy.sleep(0.5)
                    if abs(error_y) > 0.025:
                        if current_position_y > goal.target_pose.pose.position.y:
                            self.turn_ang(0.6,math.pi/2)
                            rospy.sleep(0.5)
                            self.go_linear_x(0.1,current_position_y - goal.target_pose.pose.position.y)
                            rospy.sleep(0.5)
                            self.turn_ang(-0.6,math.pi/2)
                            rospy.sleep(1)
                        else:
                            self.turn_ang(-0.6,math.pi/2)
                            rospy.sleep(0.5)
                            self.go_linear_x(0.1,-(current_position_y - goal.target_pose.pose.position.y))
                            rospy.sleep(0.5)
                            self.turn_ang(0.6,math.pi/2)
                            rospy.sleep(1)
                    self.adjust_pose(target_yaw)
                    rospy.sleep(0.5)
                    # while not rospy.is_shutdown():
                    #     try:
                    #         (current_postion2,rotation2) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0)) 
                    #         rate.sleep()
                    #         break
                    #     except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    #         continue
                    # current_position_x2 = current_postion2[0]
                    # current_position_y2 = current_postion2[1]
                    # print("is_xy_fix_after:",current_position_x2,current_position_y2)
                    return True
                else:
                    rospy.logwarn("Attempt %d: Goal failed with state: %d", attempt, state)
            else:
                rospy.logwarn("Attempt %d: Goal timed out after %d seconds", attempt, timeout)
            
            rospy.loginfo("Retrying... (attempt %d/%d)", attempt, retries)
        
        rospy.logerr("All attempts failed. Unable to reach the goal.")
        return False
    
    def make_move_base_goal(self, x, y, yaw):
        goal = MoveBaseGoal()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        q = quaternion_from_euler(0,0,yaw)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        return goal     

    def adjust_pose(self,target_yaw):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            try:
                (current_postion,current_rotation) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0))
                rate.sleep()
                break 
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

        current_position_x = current_postion[0]
        current_position_y = current_postion[1]

        _, _, current_yaw = euler_from_quaternion(current_rotation)
        error = current_yaw - target_yaw
        error = math.atan2(math.sin(error), math.cos(error))
        if abs(error) > 0.025:
            self.turn_ang(0.6*abs(error)/error,abs(error))


    
    def finish(self):
        print("START TO FINISH:")
        while True:
            time.sleep(1)
            print("CHECK CAR RECOGNIZE COUNT:",self.car_model.pic_recognize_count)
            if self.car_model.pic_recognize_count==0:
                self.car_model.CAR_FINISH = True
                break
        print("CAR TASK FINISH")
        
        # while True:
        #     time.sleep(1)
        #     if self.plane_model.PIC_RECOGNIZE_COUNT==0:
        #         break
        
        while True:
            
            time.sleep(1)
            print("CHECK PLANE RECOGNIZE COUNT:",self.plane_model.pic_recognize_count)
            if self.plane_model.pic_recognize_count==0:
                self.plane_model.PLANE_FINISH = True
                break
            print("PLANE TASK FINISH")
                
        #self.plane_model_manager.join()
        #print("plane model manager join")
        #self.car_model_manager.join()
        #print("car model manager join")

        print("FINISH CAR RECOGNITION!")

        csv_time = rospy.Time.now()
        write_path = f'/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result{csv_time}.csv'
        self.csv_writer = CsvWriter(write_path,
                                    self.car_model.model_output_buffer,
                                    self.plane_model.model_output_buffer)
        # print('FINISH PLANE RECOGNITION!')
        print("START TO SAVE RESULT AS CSV FILE")
        # self.car_model.save()
        # self.plane_model.save()
        self.csv_writer.write()
        print(f"SAVE AS {write_path}")
        return True

if __name__ == '__main__':
    nav = Nav()
    rospy.spin()


