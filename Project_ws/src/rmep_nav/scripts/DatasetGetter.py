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
# from Vision import Capture
import os
import cv2
import time
import copy
from threading import Thread
from ultralytics import YOLO
from pathlib import Path
import glob
from queue import Queue
class Model:
    def __init__(self, model_path, model_input_path, model_output_path):
        # self.model = YOLO(model_path)
        # this will bring some threading issues
        self.model_path = model_path
        self.model_input_path = model_input_path
        self.model_output_path = model_output_path
        self.model_input_buffer = Queue()
        self.model_output_buffer = Queue()
        self.middle_1 = False
        self.middle_2 = False
        self.top_1 = False
        self.top_2 = False
        self.threads_pool = []

    def plane_run(self):
        while True:
            time.sleep(5)
            check = self.plane_append()
            if check:
                plane_thread = Thread(target = self.start)
                self.threads_pool.append(plane_thread)
                plane_thread.start()
        
            if self.middle_1 and self.middle_2 and self.top_1 and self.top_2:
                return True
    
    def car_run(self):
        car_thread = Thread(target = self.car_model.start)
        self.threads_pool.append(car_thread)
        car_thread.start()

    def start(self):
        while not self.model_input_buffer.empty():
            my_list = self.model_input_buffer.get()
            img_name = my_list[0]
            print("recognize:!!!",img_name,my_list[1],type(my_list[1]))
            recognize_result = self.recognize(my_list[1])
            txt_line = [img_name, recognize_result]
            self.model_output_buffer.put(txt_line)
        return True


    def recognize(self,model_input):
        """_summary_

        Args:
            input (_type_): 可以是cv2.imread()返回的图片Mat(numpy ndarray数组),也可以是图片路径,还可以是一些别的东西.

        Returns:
            _type_: 推理结果list,例如['1 good 1 bad']
        """
        model = YOLO(self.model_path)
        results = model(model_input)


        if len(results) == 1 :
            print("yesyesyes finish recognize recognize!")

        else:
            print("it seems that you give me some wrong inputs.")


        for r in results:
            res = r.verbose()
            print(res)
        return res

    # 仅供飞机使用的方法
    def plane_append(self):
        file_names = os.listdir(self.model_input_path)
        if '1440x1088_2-top.txt' in file_names and self.top_2 == False:
            image_paths = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_2-top' + '*')))
            for i in image_paths:
                image = cv2.imread(i)
                self.append(image,i)
            self.top_2 = True
            print(self.model_output_buffer.qsize())
            return True

        if '1440x1088_2-middle.txt' in file_names and self.middle_2 == False:
            image_paths = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_2-middle' + '*')))
            for i in image_paths:
                image = cv2.imread(i)
                self.append(image,i)
            self.middle_2 = True
            print(self.model_output_buffer.qsize())
            return True 
            
        if '1440x1088_1-top.txt' in file_names and self.top_1 == False:
            image_paths = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_1-top' + '*')))
            for i in image_paths:
                image = cv2.imread(i)
                self.append(image,i)
            self.top_1 = True
            print(self.model_output_buffer.qsize())
            return True

        if '1440x1088_1-middle.txt' in file_names and self.middle_1 == False:
            image_paths = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_1-middle' + '*')))
            for i in image_paths:
                image = cv2.imread(i)
                self.append(image,i)
            self.middle_1 = True
            print(self.model_output_buffer.qsize())
            return True
        return False
    
    def car_append(self):
        pass
            
    def append(self,image,image_name):
        self.model_input_buffer.put([image,image_name])
        return True
    

    def save(self):
        while self.model_output_buffer.qsize !=0:
            txt_line = self.model_output_buffer.get()

            with open(self.model_output_path,'a') as f:
                f.write(','.join(txt_line) + '\n')
                print("Save Result:",txt_line)     

        return True

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, current):
        error = setpoint - current
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


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

        self.plane_model = Model("/home/tta/Project_ws/src/rmep_nav/scripts/model/plane-tiny76-best.pt",model_input_path='/home/tta/Project_ws/src/rmep_nav/scripts/image_from_flight/',model_output_path='/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result.txt')
        self.car_model = Model("/home/tta/Project_ws/src/rmep_nav/scripts/model/car-tiny76-best.pt",model_input_path = '/home/tta/Project_ws/src/rmep_nav/scripts/image_from_car/',model_output_path='/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result.txt')
        self.plane_thread = Thread(target = self.plane_model.plane_run)
        self.plane_thread.start()
        self.car_threads_pool = []

        self.rate = rospy.Rate(150)
        self.tf_listener = tf.TransformListener() 
        # self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        # while(not self.move_base.wait_for_server(rospy.Duration(1.0))):
        #     continue

        self.nav_for_projecte()
        # self.MakeDataSet()
        
    def nav_for_projecte(self):
        picture_count = 0
        while True:
            picture_count+=1
            time.sleep(10)
            print("try to get a picture")
            self.take_picture(f"data5-{picture_count}")
            print(f"data5-{picture_count} is captured!")

    def camera_cb(self, msg):
        if self.take_picture_need:
            self.image = self.cv_bridge.imgmsg_to_cv2(msg,"bgr8")
        else:
            return
        
    def take_picture(self, image_name):
        self.take_picture_need = True
        while self.image is None:
            continue
        rospy.sleep(1.5)

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
        #car_thread = Thread(target = self.car_model.start)
        #self.car_threads_pool.append(car_thread)
        #car_thread.start()
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



        

    def initial_pose_for_amcl(self):
        initialpose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        initial_pose = PoseWithCovarianceStamped()

        initial_pose.header.stamp = rospy.Time.now()
        initial_pose.header.frame_id = "map"
        
        initial_pose.pose.pose.position.x = 0.0
        initial_pose.pose.pose.position.y = 0.0
        initial_pose.pose.pose.position.z = 0.0
        
        quaternion = quaternion_from_euler(0, 0, math.pi/2) 
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
        print("success!")


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

    """
    laser info

    header: 
    seq: 3778
    stamp: 
        secs: 1719624923
        nsecs: 443701336
    frame_id: "laser"
    angle_min: -3.1415927410125732
    angle_max: 3.1415927410125732
    angle_increment: 0.005482709966599941
    time_increment: 0.00010589129669824615
    scan_time: 0.12135142832994461
    range_min: 0.15000000596046448
    range_max: 12.0
    ranges: "<array type: float32, length: 1147>"
    intensities: "<array type: float32, length: 1147>"
    """

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
                        self.turn_ang(0.5,alpha)
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


    def move(self, goal, target_yaw, timeout=30, retries=3):
        attempt = 0
        while attempt < retries:
            attempt += 1
            self.move_base.send_goal(goal)
            success = self.move_base.wait_for_result(rospy.Duration(timeout))        
            if success:
                state = self.move_base.get_state()
                if state == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo("Goal reached successfully on attempt %d", attempt)
                    rospy.sleep(0.5)
                    self.adjust_pose(target_yaw)
                    rospy.sleep(0.5)
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
                    if abs(current_position_x-goal.target_pose.pose.position.x) < 0.035:
                        self.go_linear_x(-0.1*abs(current_position_x-goal.target_pose.pose.position.x)/(current_position_x-goal.target_pose.pose.position.x),abs(current_position_x-goal.target_pose.pose.position.x))
                        rospy.sleep(0.5)
                    if abs(current_position_y - goal.target_pose.pose.position.y) < 0.035:
                        if current_position_y > goal.target_pose.pose.position.y:
                            self.turn_ang(0.5,math.pi/2)
                            rospy.sleep(0.5)
                            self.go_linear_x(0.1,current_position_y - goal.target_pose.pose.position.y)
                            rospy.sleep(0.5)
                            self.turn_ang(-0.5,math.pi/2)
                            rospy.sleep(0.5)
                        else:
                            self.turn_ang(-0.5,math.pi/2)
                            rospy.sleep(0.5)
                            self.go_linear_x(0.1,-(current_position_y - goal.target_pose.pose.position.y))
                            rospy.sleep(0.5)
                            self.turn_ang(0.5,math.pi/2)
                            rospy.sleep(0.5)
                    self.adjust_pose(target_yaw)
                    rospy.sleep(0.5)

                    while not rospy.is_shutdown():
                        try:
                            (current_postion2,rotation2) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0)) 
                            rate.sleep()
                            break
                        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                            continue
                    current_position_x2 = current_postion2[0]
                    current_position_y2 = current_postion2[1]
                    print("sssssssssssssssss",current_position_x2,current_position_y2)


                    return True
                else:
                    rospy.logwarn("Attempt %d: Goal failed with state: %d", attempt, state)
            else:
                rospy.logwarn("Attempt %d: Goal timed out after %d seconds", attempt, timeout)
            
            rospy.loginfo("Retrying... (attempt %d/%d)", attempt, retries)
        
        rospy.logerr("All attempts failed. Unable to reach the goal.")
        return False
    

    def cross_track_error(self, begin, end):
        rospy.sleep(1)
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
        print(current_position_x,current_position_y)

        start_x = begin[0]
        start_y = begin[1]
        target_x = end[0]
        target_y = end[1]

        dx = target_x - start_x
        dy = target_y - start_y

        px = current_position_x - start_x
        py = current_position_y - start_y

        cross_track_error = (dx * py - dy * px) / math.sqrt(dx ** 2 + dy ** 2)

        return cross_track_error

    
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
    

    def go_line_test1(self):
        a = self.make_move_base_goal(1.25,0,0)
        self.move(a,0)
        rospy.sleep(1)

        b = self.make_move_base_goal(2.75,0,0)
        self.move(b,0)
        rospy.sleep(1)
        c = self.make_move_base_goal(4.25,0,0)
        self.move(c,0)
        rospy.sleep(1)
        d = self.make_move_base_goal(5.25,0,0)
        self.move(d,0)
        rospy.sleep(1)
        e = self.make_move_base_goal(7.25,0,0)
        self.move(e,0)
        rospy.sleep(1)

    def detecte_box(self,speed_x,goal_distance_x,threshold):
        distance = goal_distance_x
        speed = speed_x
        time_total = abs(distance/speed)
        begin_time = None
        twist_linear = Twist()
        twist_linear.linear.x = speed
        twist_linear.linear.y = 0
        twist_linear.angular.z = 0
        go_start_time = rospy.Time.now()
        while (rospy.Time.now() - go_start_time).to_sec() < time_total:
            self.velocity_publisher.publish(twist_linear)
            left_distance = self.get_distance(self.laser_data,-90)
            if left_distance == float('inf'):
                self.rate.sleep()
                continue
            if left_distance < 1.25 + 0.2:
                if begin_time == None:
                    begin_time = rospy.Time.now()
                    self.rate.sleep()
                    continue
                else: 
                    if (rospy.Time.now() - begin_time).to_sec() >= threshold/speed:
                        self.stop()
                        begin_time = None
                        break
                    else:
                        self.rate.sleep()
                        continue
            else:
                if begin_time != None:
                    if (rospy.Time.now() - begin_time).to_sec() <= threshold/speed:
                        begin_time = None
                        self.rate.sleep()
                        continue
                    else:
                        begin_time = None
                        self.rate.sleep()
                        continue
                else:
                    self.rate.sleep()
                    continue

    def go_line_test2(self):
        a = self.make_move_base_goal(1.25,0,0)
        self.move(a,0)
        rospy.sleep(1)
        self.detecte_box(0.1,2,0.12)
        rospy.sleep(1)
        # while True:
        #     flag,image = self.vs.image_get()
        #     if flag:
        #         self.buffer_append(image)
        #         self.image_write(f'image_from_car/{1}.jpg',image)
        #         break

        
        # TODO 拍照

        b = self.make_move_base_goal(2.75,0,0)
        self.move(b,0)
        rospy.sleep(1)
        self.detecte_box(0.1,2,0.12)
        rospy.sleep(1)
        # TODO 拍照

        c = self.make_move_base_goal(4.25,0,0)
        self.move(c,0)
        rospy.sleep(1)
        self.detecte_box(0.1,2,0.12)
        rospy.sleep(1)
        # TODO 拍照

        d = self.make_move_base_goal(5.25,0,0)
        self.move(d,0)
        rospy.sleep(1)
        self.detecte_box(0.1,2,0.12)
        rospy.sleep(1)
        # TODO 拍照

        e = self.make_move_base_goal(7.25,0,0)
        self.move(e,0)
        rospy.sleep(1)

        

    def adjust_pose(self,target_yaw):
        rospy.sleep(1)
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
        print(current_position_x,current_position_y)

        _, _, current_yaw = euler_from_quaternion(current_rotation)
        error = current_yaw - target_yaw
        error = math.atan2(math.sin(error), math.cos(error))
        if abs(error) > 0.025:
            self.turn_ang(0.6*abs(error)/error,abs(error))


    def adjust_postion(self,begin,end):
        self.adjust_pose(0)
        rospy.sleep(1)
        error = self.cross_track_error(begin,end)
        if abs(error) > 0.025:
            if error > 0:
                self.turn_ang(0.5,math.pi/2)
                rospy.sleep(1)
                self.go_linear_x(0.1,abs(error))
                rospy.sleep(1)
                self.turn_ang(-0.5,math.pi/2)
            else:
                self.turn_ang(-0.5,math.pi/2)
                rospy.sleep(1)
                self.go_linear_x(0.1,abs(error))
                rospy.sleep(1)
                self.turn_ang(0.5,math.pi/2)
            # self.go_linear_y(0.2*error/abs(error),abs(error))
            rospy.sleep(1)
            self.adjust_pose(0)
        else:
            return
    
    def finish(self):
        for car_thread in self.car_model.threads_pool:
            car_thread.join()
        for plane_thread in self.car_model.threads_pool:
            plane_thread.join()
        self.car_model.save()
        self.plane_model.save()
        return True

    def go_line_test3_1(self):
        a = self.make_move_base_goal(1.25,0.45,0)
        self.move(a,0)
        rospy.sleep(1)
        self.adjust_postion((1.25,0.45),(5.25,0.45))
        rospy.sleep(1)
        count = 0
        begin_time = None
        treshold = 0.12
        speed = 0.1
        twist_linear = Twist()
        twist_linear.linear.x = speed
        twist_linear.linear.y = 0
        twist_linear.angular.z = 0
        while count < 4:
            self.velocity_publisher.publish(twist_linear)
            left_distance = self.get_distance(self.laser_data,-90)
            if left_distance == float('inf'):
                self.rate.sleep()
                continue
            if left_distance < 0.7 + 0.2:
                if begin_time == None:
                    begin_time = rospy.Time.now()
                    self.rate.sleep()
                    continue
                else: 
                    if (rospy.Time.now() - begin_time).to_sec() >= treshold/speed:
                        self.stop()
                        rospy.sleep(1)
                        self.adjust_postion((1.25,0.45),(5.25,0.45))
                        rospy.sleep(1)
                        time = rospy.Time.now()
                        self.take_picture(str(time))
                        while self.get_distance(self.laser_data,-90) < 0.7 + 0.2:
                            self.velocity_publisher.publish(twist_linear)
                            self.rate.sleep()
                            continue
                        begin_time = None 
                        count += 1
                        self.rate.sleep()
                        continue
                    else:
                        self.rate.sleep()
                        continue
            else:
                if begin_time != None:
                    if (rospy.Time.now() - begin_time).to_sec() <= treshold/speed:
                        begin_time = None
                        self.rate.sleep()
                        continue
                    else:
                        begin_time = None
                        self.rate.sleep()
                        continue
                else:
                    self.rate.sleep()
                    continue
        self.stop()
        rospy.sleep(1)
        e = self.make_move_base_goal(7.25,0.25,math.pi/2)
        self.move(e,math.pi/2)
        rospy.sleep(1)
        return


    def go_line_test3_2(self):
        a = self.make_move_base_goal(7.25,2.75,0)
        self.move(a,0)
        rospy.sleep(1)
        self.adjust_postion((1.25,2.75),(7.25,2.75))
        rospy.sleep(1)
        count = 0
        begin_time = None
        treshold = 0.12
        speed = -0.1
        twist_linear = Twist()
        twist_linear.linear.x = speed
        twist_linear.linear.y = 0
        twist_linear.angular.z = 0
        while count < 4:
            self.velocity_publisher.publish(twist_linear)
            left_distance = self.get_distance(self.laser_data,-90)
            if left_distance == float('inf'):
                self.rate.sleep()
                continue
            if left_distance < 0.7 + 0.25:
                if begin_time == None:
                    begin_time = rospy.Time.now()
                    self.rate.sleep()
                    continue
                else: 
                    if (rospy.Time.now() - begin_time).to_sec() >= abs(treshold/speed):
                        self.stop()
                        rospy.sleep(1)
                        self.adjust_postion((1.25,2.75),(7.25,2.75))
                        rospy.sleep(1)
                        time = rospy.Time.now()
                        self.take_picture(str(time))
                        while self.get_distance(self.laser_data,-90) < 0.7 + 0.25:
                            self.velocity_publisher.publish(twist_linear)
                            self.rate.sleep()
                            continue
                        begin_time = None 
                        count += 1
                        self.rate.sleep()
                        continue
                    else:
                        self.rate.sleep()
                        continue
            else:
                if begin_time != None:
                    if (rospy.Time.now() - begin_time).to_sec() <= abs(treshold/speed):
                        begin_time = None
                        self.rate.sleep()
                        continue
                    else:
                        begin_time = None
                        self.rate.sleep()
                        continue
                else:
                    self.rate.sleep()
                    continue
        self.stop()
        rospy.sleep(1)
        self.turn_ang(0.5,math.pi)
        rospy.sleep(1)
        e = self.make_move_base_goal(0,2.75,-math.pi/2)
        self.move(e,-math.pi/2)
        rospy.sleep(1)
        return


    def test_4(self):
        
        a = self.make_move_base_goal(2.0,0.45,0)
        self.move(a,0)
        rospy.sleep(1)
        time = rospy.Time.now()
        # self.take_picture(str(time))
        b = self.make_move_base_goal(3.5,0.45,0)
        self.move(b,0)
        rospy.sleep(1)
        time = rospy.Time.now()
        # self.take_picture(str(time))
        c = self.make_move_base_goal(4.75,0.45,0)
        self.move(c,0)
        rospy.sleep(1)
        time = rospy.Time.now()
        self.take_picture(str(time))
        d = self.make_move_base_goal(5.75,0.45,0)
        self.move(d,0)
        rospy.sleep(1)
        time = rospy.Time.now()
        self.take_picture(str(time))
        e = self.make_move_base_goal(7.25,0.05,math.pi/2)
        self.move(e,math.pi/2)
        rospy.sleep(1)
        self.has_arrive('/mnt/arrival.txt')
        self.is_go('/mnt/start_again.txt')
        f = self.make_move_base_goal(7.25,2.75,math.pi)
        self.move(f,math.pi)
        rospy.sleep(1)
        f2 = self.make_move_base_goal(0,2.75,0)
        self.move(f2,0)
        rospy.sleep(1)
        j = self.make_move_base_goal(2,2.75,0)
        self.move(j,0)
        rospy.sleep(1)
        time = rospy.Time.now()
        self.take_picture(str(time))
        i = self.make_move_base_goal(3.5,2.75,0)
        self.move(i,0)
        rospy.sleep(1)
        time = rospy.Time.now()
        self.take_picture(str(time))
        h = self.make_move_base_goal(4.75,2.75,0)
        self.move(h,0)
        rospy.sleep(1)
        time = rospy.Time.now()
        self.take_picture(str(time))
        g = self.make_move_base_goal(5.75,2.75,0)
        self.move(g,0)
        rospy.sleep(1)
        time = rospy.Time.now()
        self.take_picture(str(time))
        self.turn_ang(0.5,math.pi)
        rospy.sleep(1)
        f3 = self.make_move_base_goal(0,2.75,math.pi)
        self.move(f3,math.pi/2)
        rospy.sleep(1)
        self.turn_ang(0.5,math.pi)
        rospy.sleep(1)
        destinition = self.make_move_base_goal(0,0,math.pi/2)
        self.move(destinition,math.pi/2)




        

            

        


            





    # ????????????,????
    def go_line(self, begin, end):
        target_ang = math.atan2(end.target_pose.pose.position.y - begin.target_pose.pose.position.y,
                                end.target_pose.pose.position.x - begin.target_pose.pose.position.x)
        q = quaternion_from_euler(0,0,target_ang)
        begin.target_pose.pose.orientation.x = q[0]
        begin.target_pose.pose.orientation.y = q[1]
        begin.target_pose.pose.orientation.z = q[2]
        begin.target_pose.pose.orientation.w = q[3]
        begin.target_pose.header.frame_id = "map"
        begin.target_pose.header.stamp = rospy.Time.now()
        self.move(begin)
        while not rospy.is_shutdown():
            try:
                (current_postion,rotation) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0)) 
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            current_position_x = current_postion[0]
            current_position_y = current_postion[1]
            numerator = abs((end.target_pose.pose.position.y - begin.target_pose.pose.position.y) * current_position_x -
                        (end.target_pose.pose.position.x - begin.target_pose.pose.position.x) * current_position_y +
                        end.target_pose.pose.position.x * begin.target_pose.pose.position.y -
                        end.target_pose.pose.position.y * begin.target_pose.pose.position.x)
            denominator = math.sqrt((end.target_pose.pose.position.y - begin.target_pose.pose.position.y) ** 2 + (end.target_pose.pose.position.x - begin.target_pose.pose.position.x) ** 2)
            cross_track_error = numerator / denominator
            self.rate.sleep()

        pass

    # ????????????
    def stay_in_center(self):
        pass


if __name__ == '__main__':
    nav = Nav()
    rospy.spin()


