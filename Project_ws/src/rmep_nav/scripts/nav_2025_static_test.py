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
import re

def filter_boxes_car(boxes, span_factor=1.25, min_factor=0.2):
    CARD_SPAN_X = 4
    CARD_SPAN_Y = 3
    BOX_MIN_MORPH = 0.3 # 1:1
    BOX_MAX_MORPH = 1.5 # 3:2
    
    raw_list = [boxes.xywhn.numpy(), boxes.cls.numpy()]
    rbox_list = []
    rcls_list = []
    if len(boxes) == 0:
        # If no objects are detected, return False with zero counts
        return False, 0, 0

    BOX_VALID = False
    BOX_XYWHN = 0
    BOX_MAX_AREA = 0
    BOX_BAD_COUNT = 0
    BOX_GOOD_COUNT = 0
    CENTRAL_BIAS = 10

    # 得到最大识别框面积并过滤变形严重的识别框
    for obj_box, obj_cls in zip(raw_list[0], raw_list[1]):
        obj_width, obj_height = obj_box[2], obj_box[3]
        this_area = obj_width * obj_height
        if BOX_MAX_AREA < this_area:
            BOX_MAX_AREA = this_area
        if BOX_MIN_MORPH < (obj_width / obj_height) < BOX_MAX_MORPH:
            rbox_list.append(obj_box)
            rcls_list.append(obj_cls)
    
    index = 0
    CENTRAL_OBJECT_INDEX = 0

    # 小框过滤
    for obj_box, obj_cls in zip(rbox_list, rcls_list):
        obj_central_x , obj_central_y = obj_box[0],obj_box[1]
        obj_width , obj_height = obj_box[2],obj_box[3]
        my_distance = (obj_central_x-0.5)**2 + (obj_central_y-0.5)**2
        if (my_distance < CENTRAL_BIAS) and (obj_width * obj_height > BOX_MAX_AREA * min_factor):
            CENTRAL_BIAS = my_distance
            CENTRAL_OBJECT_INDEX = index
            BOX_VALID = True
        
        index += 1

    # 4倍滑动窗口检测
    if BOX_VALID:
        centroid_x = rbox_list[CENTRAL_OBJECT_INDEX][0]
        centroid_y = rbox_list[CENTRAL_OBJECT_INDEX][1]
        centroid_w = rbox_list[CENTRAL_OBJECT_INDEX][2]
        centroid_h = rbox_list[CENTRAL_OBJECT_INDEX][3]
        
        BOX_XYWHN = [
            centroid_x,
            centroid_y,
            centroid_w * CARD_SPAN_X * span_factor,
            centroid_h * CARD_SPAN_Y * span_factor
        ]
        
        for obj_box, obj_cls in zip(rbox_list, rcls_list):
            if (BOX_XYWHN[0] - BOX_XYWHN[2] / 2 < obj_box[0] < BOX_XYWHN[0] + BOX_XYWHN[2] / 2) and (BOX_XYWHN[1] - BOX_XYWHN[3] / 2 < obj_box[1] < BOX_XYWHN[1] + BOX_XYWHN[3] / 2):
                if -0.01 < obj_cls - 1 < 0.01:
                    BOX_GOOD_COUNT += 1
                elif -0.01 < obj_cls < 0.01:
                    BOX_BAD_COUNT += 1
                else:
                    print("我希望永远不会得到这个输出")
        
        return True, BOX_GOOD_COUNT, BOX_BAD_COUNT
    
    return False, 0, 0

    
def filter_boxes_plane(boxes,span_factor=1.25,layer=1,min_factor=0.2):
    CARD_SPAN_X = 4
    CARD_SPAN_Y = 3
    BOX_MIN_MORPH = 0.3 # 1:1
    BOX_MAX_MORPH = 1.5 # 3:2
    
    raw_list = [boxes.xywhn.numpy(), boxes.cls.numpy()]
    rbox_list = []
    rcls_list = []
    if len(boxes) == 0:
        # 1.如果没有识别到物体,返回False
        return False,0,0

    
    # 2.如果识别到物体,获取离中心点最近的,位于识别Window窗体内的识别卡
    else:
        BOX_VALID = False
        
        BOX_XYWHN = 0
        BOX_MAX_AREA = 0
        
        BOX_BAD_COUNT = 0
        BOX_GOOD_COUNT = 0
        
        CENTRAL_BIAS = 10
        CENTRAL_OBJECT_INDEX = 0
        
        # 1-1-过滤变形严重的识别框
        # 1-2-获取最大的识别框的面积
        for obj_box, obj_cls in zip(raw_list[0], raw_list[1]):
            obj_width , obj_height = obj_box[2],obj_box[3]
            this_area = obj_width  * obj_height
            if BOX_MAX_AREA < this_area:
                BOX_MAX_AREA = this_area
            #if True:
            if BOX_MIN_MORPH < (obj_width/obj_height) < BOX_MAX_MORPH:
                rbox_list.append(obj_box)
                rcls_list.append(obj_cls)
            
        
        index = 0
        
        # 2-1 小框过滤(mian)
        # 2-2 上下比例窗口过滤
        # 2-3 最近邻
        
        for obj_box, obj_cls in zip(rbox_list,rcls_list):
            obj_central_x , obj_central_y = obj_box[0],obj_box[1]
            obj_width , obj_height = obj_box[2],obj_box[3]
            
            print("TEST FOR RBOX_LIST:",rbox_list[0])
            centroid_x = rbox_list[CENTRAL_OBJECT_INDEX][0]
            centroid_y = rbox_list[CENTRAL_OBJECT_INDEX][1]
            centroid_w = rbox_list[CENTRAL_OBJECT_INDEX][2]
            centroid_h = rbox_list[CENTRAL_OBJECT_INDEX][3]
            
            if not (obj_width*obj_height > BOX_MAX_AREA*min_factor):
                print("小框过滤未通过")
            if obj_width*obj_height > BOX_MAX_AREA*min_factor:

                
                if layer   == 1:
                    my_distance = (obj_central_x-0.5)**2 + (obj_central_y)**2
                    CHECK_WINDOW_XYWHN = [0.5 ,  0.25 ,  centroid_w *14, centroid_h *10]
                    # if LAYER = 1 
                    # then CHECK_WINDOW = [0.5,0.3]
                    
                elif layer == 2:
                    my_distance = (obj_central_x-0.5)**2 + (1-obj_central_y)**2
                    CHECK_WINDOW_XYWHN = [0.5 ,  0.75 ,  centroid_w *14 , centroid_h*8]

                    # then CHECK_WINDOW = [0.5,0.75]
                    
                else:
                    my_distance = (obj_central_x-0.5)**2 + (obj_central_y-0.5)**2
                    CHECK_WINDOW_XYWHN = [0.5 ,  0.5,  0.4, 0.4]
                print("WINDOW:",CHECK_WINDOW_XYWHN)
                print("PICTURE:",rbox_list[CENTRAL_OBJECT_INDEX])
                if not (CHECK_WINDOW_XYWHN[0]-CHECK_WINDOW_XYWHN[2]/2 < obj_central_x < CHECK_WINDOW_XYWHN[0]+CHECK_WINDOW_XYWHN[2]/2):
                    print("横向检测框未通过") 
                if not (CHECK_WINDOW_XYWHN[1]-CHECK_WINDOW_XYWHN[3]/2 < obj_central_y < CHECK_WINDOW_XYWHN[1]+CHECK_WINDOW_XYWHN[3]/2):
                    print("纵向检测框未通过")
                if (my_distance < CENTRAL_BIAS) and (CHECK_WINDOW_XYWHN[0]-CHECK_WINDOW_XYWHN[2]/2 < obj_central_x < CHECK_WINDOW_XYWHN[0]+CHECK_WINDOW_XYWHN[2]/2) and (CHECK_WINDOW_XYWHN[1]-CHECK_WINDOW_XYWHN[3]/2 < obj_central_y < CHECK_WINDOW_XYWHN[1]+CHECK_WINDOW_XYWHN[3]/2):

                    CENTRAL_BIAS = my_distance
                    CENTRAL_OBJECT_INDEX = index
                    BOX_VALID = True
                    
            index = index + 1

        # 3-1 滑动窗口
        if BOX_VALID:
            # EXPAND 策略，BOX——XYWHN制作
            # 0:X 1:Y 2:W 3:H 
            # 归一化坐标检查
            # 只要物体的中心点在BOX内即可(BOX为物体中心点检测框)
            # 中央坐标， 4倍宽度坐标*span_factor,3倍高度坐标*span_factor

            
            BOX_XYWHN = [centroid_x,
            centroid_y,
            centroid_w*CARD_SPAN_X*span_factor,
            centroid_h*CARD_SPAN_Y*span_factor]
            
            

                
            
            for obj_box,obj_cls in zip(rbox_list, rcls_list):
                if not (BOX_XYWHN[0]-BOX_XYWHN[2]/2 <obj_box[0]< BOX_XYWHN[0]+BOX_XYWHN[2]/2):
                    print("滑动窗口横向未通过")
                if not (BOX_XYWHN[1]-BOX_XYWHN[3]/2 <obj_box[1]< BOX_XYWHN[1]+BOX_XYWHN[3]/2):
                    print("滑动窗口纵向未通过")
                    
                # 4倍滑动窗口
                if (BOX_XYWHN[0]-BOX_XYWHN[2]/2 <obj_box[0]< BOX_XYWHN[0]+BOX_XYWHN[2]/2) \
                    and (BOX_XYWHN[1]-BOX_XYWHN[3]/2 <obj_box[1]< BOX_XYWHN[1]+BOX_XYWHN[3]/2):
                    #and obj_width*obj_height > BOX_MAX_AREA*min_factor:
                    # 0:BAD 1:GOOD

                    if -0.01 < obj_cls -1 < 0.01:
                        BOX_GOOD_COUNT = BOX_GOOD_COUNT + 1
                        
                    elif -0.01 < obj_cls < 0.01:
                        BOX_BAD_COUNT = BOX_BAD_COUNT + 1
                    
                    else:
                        print("我希望永远都不会看到这个输出")
                    
            return True,BOX_GOOD_COUNT,BOX_BAD_COUNT
        return False, 0, 0
    
    
class Model:
    def __init__(self, device,model_path, model_input_path,pic_recognize_count,num_workers=4):
        # self.model = YOLO(model_path)
        # this will bring some threading issues
        self.device = device
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
            if check==1:
                with self.condition:
                    self.condition.notify_all()
        
            if check==2:
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
        results = model.predict(model_input,conf=0.75)
        # print("notice",type(results))
        res = results[0].boxes

        # if len(results) == 1 :
        #     pass
        #     #print("yesyesyes finish recognize recognize!")

        # else:
        #     print("it seems that you give me some wrong inputs.")
        # print(model_input,type(model_input))
        # if isinstance(model_input, str):
        #     #print("model_input是一个字符串")
        # else:
        #     #print("model_input不是一个字符串")
        
        try:
            if self.device=='plane':
                match = re.search(r'_(\d+)-(\d+)\.jpg$', model_input)
                if match:
                    var1 = match.group(1)
                    var2 = match.group(2)
                    if (-0.1<int(var2)<4.1):
                        ret,good_num,bad_num = filter_boxes_plane(res,layer=1)
                    elif (4.1<int(var2)<8.1):
                        ret,good_num,bad_num = filter_boxes_plane(res,layer=2)
            elif self.device=='car':
                 ret,good_num,bad_num = filter_boxes_car(res)
            return good_num,bad_num

        except Exception as e:
            print("err in recognize function.")

    


    # 仅供飞机使用的方法
    def plane_task(self):
        file_names = os.listdir(self.model_input_path)
        if 'finish4.txt' in file_names and self.PLANE_1 == False:
            print("FINISH 4 SIGNAL!!!")
            image_paths = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_1' + '*')))
            #print("image_paths:",image_paths)
            for image_name in image_paths:
                #image_result = cv2.imread(image_name)
                ##############################################
                self.append(image_name,image_name)
                print("succesfully append an image to model_input_buffer")
            self.PLANE_1 = True
            #print(self.model_output_buffer.qsize())
            return 1

        elif 'finish8.txt' in file_names and self.PLANE_2 == False:
            print("FINISH 8 SIGNAL!!!")
            image_paths = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_2' + '*')))
            #print("image paths for finish 8:",image_paths)
            for image_name in image_paths:
                #image_result = cv2.imread(image_name)
                #############################################
                self.append(image_name,image_name)
            self.PLANE_2 = True
            #print(self.model_output_buffer.qsize())
            return 2
        
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
    def __init__(self,car_model_buffer,plane_model_buffer,output_file_path):
        self.car_model_buffer = car_model_buffer
        self.plane_model_buffer = plane_model_buffer
        self.output_file_path = output_file_path
        with open(output_file_path,mode='w',newline='') as f:
            init_writer = csv.writer(f)
            init_writer.writerow(('房间','好人','坏人'))
        f.close()

        
    def write(self):
        csv_file = self.output_file_path
        with open(csv_file,mode='a',newline='') as f:
            this_writer = csv.writer(f)
            while self.car_model_buffer.qsize() > 0:
                csv_line = self.car_model_buffer.get()
                this_writer.writerow(csv_line)
                print("WRITE A CAR MODEL LINE:",csv_line)
        
            while self.plane_model_buffer.qsize()>0:
                model_tuple = self.plane_model_buffer.get()
                csv_line = [1,1,1]
                csv_line[0] = model_tuple[0]
                csv_line[1] = model_tuple[1]
                csv_line[2] = model_tuple[2]
                room_id = csv_line[0]
                match = re.search(r'(\d+)-(\d+)', room_id)

                if match:
                    # 将匹配到的部分用 '_' 连接
                    room_id = f"{match.group(1)}_{match.group(2)}"
                    print(room_id)
                    csv_line[0] = room_id
                else:
                    print("No match found")
                    
                this_writer.writerow(csv_line)
                print("WRITE A PLANE MODEL LINE:",csv_line)
        f.close()


class Nav:
    def __init__(self):
        rospy.init_node('nav', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # self.laser_subscriber = rospy.Subscriber('/scan', LaserScan, self.laser_callback,queue_size=1)
        # self.image_raw = rospy.Subscriber("ep_cam/image_raw", Image, self.camera_cb, queue_size=1)
        # self.cv_bridge = CvBridge()
        # self.laser_data = None
        # self.image = None
        # self.take_picture_need = False
        # self.image_lock = threading.Lock()

        # self.plane_model = Model('plane',
        #     "/home/tta/Project_ws/src/rmep_nav/scripts/model/720car-2.pt",
        # model_input_path='/home/tta/Project_ws/src/rmep_nav/scripts/image_from_flight/',
        # pic_recognize_count=16)

        # self.car_model = Model('car',
        #     "/home/tta/Project_ws/src/rmep_nav/scripts/model/720car-1.pt",
        # model_input_path = '/home/tta/Project_ws/src/rmep_nav/scripts/image_from_car/',
        # pic_recognize_count=8)
        
        # self.car_model_manager = Thread(target=self.car_model.car_run)
        # self.car_model_manager.start()
        # self.plane_model_manager = Thread(target=self.plane_model.plane_run)
        # self.plane_model_manager.start()
        # self.csv_time = rospy.Time.now()
        # self.csv_writer = CsvWriter(self.car_model.model_output_buffer,
        #                     self.plane_model.model_output_buffer,
        #                     f'/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result{self.csv_time}.csv')
    
        self.rate = rospy.Rate(100)
        # self.tf_listener = tf.TransformListener() 
        # self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        # while(not self.move_base.wait_for_server(rospy.Duration(1.0))):
        #     continue
        self.nav_for_projecte()
        
    def nav_for_projecte(self):
        print("nav start")
        
        speed=0.5

        self.go_linear_x(speed,2.5)
        rospy.sleep(1)
        self.go_linear_y(-speed,0.9)
        rospy.sleep(1)

        # #小车到达1区，给飞机发送指令起飞
        # self.is_go('/mnt/start.txt')
        # #飞机检索完1区，给小车回复启动
        # self.has_arrive('/mnt/finish.txt')

        print("1区驶往2区.......")

        self.go_linear_x(speed,2.35) #2.5理论
        rospy.sleep(1)
        self.go_linear_y(speed,1.8)
        rospy.sleep(1)

        # #飞机检索完2区，并已经成功降落，给小车回复启动
        # self.has_arrive('/mnt/finish.txt')
        print("2区驶往充电区......")

        self.go_linear_y(-speed,0.9)
        rospy.sleep(1)
        self.go_linear_x(speed,2)
        #充电区充电
        rospy.sleep(5)   

        #返航
        print("返航......") 
        self.go_linear_x(-speed,7)            


    


    def has_arrive(self,path):
        # hostname = '192.168.110.55'
        hostname = '192.168.31.110'
        username = 'root'
        password = '123456'
        filepath = path
        conn = None

        
        while True:
            try:
                conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})
                result = conn.run(f"touch {filepath}")

                if result.ok:
                    print("File created successfully.")
                    conn.close()
                    break
                else:
                    print("File created filed.Attempting to reconnect...")
                    if conn:
                        conn.close()
                    time.sleep(2)
                    continue

            except Exception as e:
                print("Attempting to reconnect...")
                if conn:
                    conn.close()
                time.sleep(2)


    def is_go(self,path):

        # hostname = '192.168.110.55'
        hostname = '192.168.31.110'
        username = 'root'
        password = '123456'
        filepath = path
        conn = None
        
        while True:
            try:
                conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})
                while True:
                    time.sleep(2)
                    result = conn.run('test -e {}'.format(filepath), hide=True)
                    if result.ok: 
                        print("fly succseefully")
                        conn.run(f'rm {filepath}')
                        conn.close()
                        return
                    else:
                        print("next check")
                        time.sleep(2)
            except Exception as e:
                print("Attempting to reconnect...")
                if conn:
                    conn.close()
                time.sleep(2)

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




if __name__ == '__main__':
    nav = Nav()
    rospy.spin()


