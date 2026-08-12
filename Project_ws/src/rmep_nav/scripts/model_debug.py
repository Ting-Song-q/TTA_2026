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
import ultralytics
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
        print("notice",type(results))
        #print(results)
        #print(results)
        print("try to print boxes:",results[0].boxes)

        if len(results) == 1 :
            pass
            #print("yesyesyes finish recognize recognize!")

        else:
            print("it seems that you give me some wrong inputs.")

        res = results[0].boxes
        
        ret,good_num,bad_num = filter(res)
        return good_num,bad_num
    


    # 仅供飞机使用的方法
    def plane_task(self):
        file_names = os.listdir(self.model_input_path)
        if 'finish4.txt' in file_names and self.PLANE_1 == False:
            image_paths = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_1' + '*')))
            for image_name in image_paths:
                image_result = cv2.imread(image_name)
                self.append(image_name,image_result)
            self.PLANE_1 = True
            #print(self.model_output_buffer.qsize())
            return True

        elif 'finish8.txt' in file_names and self.PLANE_2 == False:
            image_paths = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_2' + '*')))
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

        
            while self.plane_model_buffer.qsize()>0:
                csv_line = self.plane_model_buffer.get()
                writer.writerow(csv_line)
        f.close()


if __name__=='__main__':
        plane_model = Model("/home/tta/Project_ws/src/rmep_nav/scripts/model/720car-1.pt",
        model_input_path='/home/tta/Project_ws/src/rmep_nav/scripts/image_from_flight/',
        pic_recognize_count=16)

        plane_model_manager = Thread(target=plane_model.plane_run)
        plane_model_manager.start()
        while True:
            time.sleep(1)
            if plane_model.pic_recognize_count==15:
                plane_model.PLANE_FINISH = True
                break
        plane_model_manager.join()


        print("FINISH CAR RECOGNITION!")
        write_path = '/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result.csv'
        csv_writer = CsvWriter(write_path,
                                    '',
                                    plane_model.model_output_buffer)
        # print('FINISH PLANE RECOGNITION!')
        print("START TO SAVE RESULT AS CSV FILE")
        # self.car_model.save()
        # self.plane_model.save()
        csv_writer.write()
        print(f"SAVE AS {write_path}")