from ultralytics import YOLO
import re
import glob
import os
import csv
import datetime
import shutil


def copy_and_rename_csv(src_path):
    total_good = 0
    total_bad = 0
    row_count = 0

    with open(src_path, mode='r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            total_good += int(row[1])
            total_bad += int(row[2])
            row_count += 1
    f.close()

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    new_csv_name = f"{total_good+total_bad}-{row_count}-{timestamp}.csv"
    new_csv_path = os.path.join(os.path.dirname(src_path), new_csv_name)
    
    shutil.copy(src_path, new_csv_path)

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
    
    
    
car_image_paths = '/home/tta/Project_ws/src/rmep_nav/scripts/image_from_car/'
plane_image_paths = '/home/tta/Project_ws/src/rmep_nav/scripts/image_from_flight_hand/'
csv_folder_path = '/home/tta/Project_ws/src/rmep_nav/scripts/auto_csv/'

plane_images = sorted(glob.glob(os.path.join(plane_image_paths, '1440x1088' + '*')))
car_images = sorted(glob.glob(os.path.join(car_image_paths,  '*')))

model = YOLO("/home/tta/Project_ws/src/rmep_nav/scripts/model/720car-1.pt")


# 检查文件是否存在并根据需要更改文件名
def get_csv_filename(base_path, base_name):
    csv_path = os.path.join(base_path, base_name)
    if not os.path.exists(csv_path):
        return csv_path
    else:
        index = 1
        new_csv_path = os.path.join(base_path, f"{base_name.split('.')[0]}-{index}.csv")
        while os.path.exists(new_csv_path):
            index += 1
            new_csv_path = os.path.join(base_path, f"{base_name.split('.')[0]}-{index}.csv")
        return new_csv_path

csv_filename = get_csv_filename(csv_folder_path, 'auto.csv')

# 在这里增加一个方法，创建一个csv文件，文件名为auto.csv或auto-1.csv，内容为房间，好人，坏人
with open(csv_filename, mode='w', newline='') as f:
    init_writer = csv.writer(f)
    init_writer.writerow(('房间', '好人', '坏人'))
f.close()

for image_name in plane_images:
    if image_name.endswith('.jpg'):
        results = model.predict(image_name, conf=0.75)
        bxs = results[0].boxes
        room_name = re.search(r'_(\d+)-(\d+)\.jpg$', image_name)
        if room_name:
            var1 = room_name.group(1)
            var2 = room_name.group(2)
            if -0.1 < int(var2) < 4.1:
                ret, good_num, bad_num = filter_boxes_plane(bxs, layer=1)
            elif 4.1 < int(var2) < 8.1:
                ret, good_num, bad_num = filter_boxes_plane(bxs, layer=2)

            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow((f"{var1}-{var2}", good_num, bad_num))
            f.close()
print("finish plane recognize")

for car_image_name in car_images:
    print(car_image_paths)
    if car_image_name.endswith('.jpg'):
        
        car_results = model.predict(car_image_name, conf=0.75)
        bxs = car_results[0].boxes
        ret, good_num, bad_num = filter_boxes_car(bxs)
        print("try to enter csv")
        basename = os.path.basename(car_image_name)
        try:
            car_room_name = re.sub(r'\.jpg$', '', basename)
        except:
            car_room_name = basename
        try:
            with open(csv_filename, mode='a', newline='') as f:
                car_writer = csv.writer(f)
                car_writer.writerow((f"{car_room_name}", good_num, bad_num))
                print("write row",f"{car_room_name}: {good_num} good, {bad_num} bad")
            f.close()
        except:
            print("error entering csv")

print("finish car recognize")
copy_and_rename_csv(csv_filename)
print("copy and rename finished")
