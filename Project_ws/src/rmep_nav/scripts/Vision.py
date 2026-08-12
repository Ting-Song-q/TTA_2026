#!/usr/bin/python3
# coding=UTF-8

import cv2
#from robomaster import robot
#from robomaster import camera
import numpy as np
import time
from ultralytics import YOLO

    
class Maths:
    def fix(img):
        dist_coeffs = np.array([-0.0444954569665007, -0.00201770323876866, 0, 0, -0.0159863040899372])
        intrinsic = np.array([[624.302683276108, 0, 632.376721286788],
                    [0, 623.915644264347, 371.420934824104],
                    [0, 0, 1]])
        corrected_image = cv2.undistort(img, intrinsic, dist_coeffs)
        return corrected_image
    
    
class Capture:
    def __init__(self,model_path=''):
        self.ep_robot = robot.Robot()
        self.ep_robot.initialize(conn_type='rndis')
        self.ep_camera = self.ep_robot.camera
        self.ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
        #print(ep_camera.video_stream_addr)
        self.buffer = []
        print("DynamicCap initialized successfully")
        self.model = YOLO(model_path)
        print("Model initialized successfully")
        self.model_recoginize_count = 0
        self.maths = Maths()
    
    
    def image_get(self,policy='newest'):
        """_summary_

        Args:
            policy (str, optional): _description_. Defaults to 'newest'.

        Returns:
            _type_: opencv格式(numpy ndarray)图像
        """
        img = self.ep_camera.read_cv2_image(timeout=3,strategy=policy)
        img = self.maths.fix(img)  
        return True,img
    
    
    def image_write(self,path,img):
        """_summary_

        Args:
            path (_type_): 图像路径,可以是绝对路径也可以是相对路径.注意,相对路径是取决于当前工作目录的相对路径,由主函数入口决定,即从哪个路径启动python xxx.py,如果使用了大量os库和threading库函数,可能会导致相对路径格式不清晰.
            img (_type_): opencv格式(numpy ndarray)图像

        Returns:
            _type_: opencv格式(numpy ndarray)图像
        """
        cv2.imwrite(path,img)
        print("successfully write an image")
        return True
        
        
    def buffer_append(self,img):
        self.buffer.append(img)
        return True
    
        
    def buffer_clear(self):
        self.buffer.clear()
        return True
    
    
    def buffer_get(self):
        return self.buffer
    
    
    def buffer_len(self):
        return len(self.buffer)
    
    
    def model_recognize(self,input,autosave=False):
        """_summary_

        Args:
            input (_type_): 可以是cv2.imread()返回的图片Mat(numpy ndarray数组),也可以是图片路径,还可以是一些别的东西.
            autosave (bool, optional): 打开自动保存功能,在当前目录下创建model_results文件夹,保存每一张图片的推理结果为单个txt. Defaults to False.

        Returns:
            _type_: 返回单张图片的推理结果,例如1 good 1 bad
        """
        results = self.model(input)  
        for r in results:
            res = r.verbose()
            print("Model Output:",res)
            if autosave:
                self.model_recoginize_count+=1
                r.save_txt(f"./model_results/model_result_{self.model_recoginize_count}.txt")
        return res
    
    def close(self):
        self.ep_camera.stop_video_stream()
        self.ep_robot.close()
        print("quit successfully!")
        
        
        
if __name__=="__main__":
    img = cv2.imread("{157}.jpg")
    print(img)
    dist_coeffs = np.array([-0.0444954569665007, -0.00201770323876866, 0, 0, -0.0159863040899372])
    corrected_image = cv2.undistort(img, np.array([[624.302683276108, 0, 632.376721286788],
                    [0, 623.915644264347, 371.420934824104],
                    [0, 0, 1]]), dist_coeffs)
    cv2.imwrite('corrected_image.jpg', corrected_image)