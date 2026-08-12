#!/usr/bin/python3
# coding=UTF-8
from fabric2 import Connection
import time
import csv

def download_flight_result(remote_path, local_path):
    # 配置信息
    hostname = '192.168.50.22'
    username = 'forlinx'
    password = 'forlinx'
    conn = None

    while True:
        try:
            # 建立连接
            conn = Connection(host=hostname, user=username, 
                            connect_kwargs={"password": password})
            

                    # 方法1：直接使用cat命令读取CSV内容
            while True:
                cat_result = csv_result = conn.run(f'cat {remote_path}', hide=True)
                if cat_result.ok:
                    csv_content = csv_result.stdout
                    print("读取成功")
                    break
                else:
                    print("读取不成功，等待2秒后重试...")
                    time.sleep(2)
            
            # 将内容写入本地文件
            with open(local_path, 'w') as f:
                f.write(csv_content)

            print("CSV内容已成功读取并保存到本地")
            return    

                    
        except Exception as e:
            print(f"连接出错, 尝试重新连接...")
            if conn:
                conn.close()
            time.sleep(2)


def merge_csv_files(flight_file, car_file, output_file):
    # 读取result_flight.csv
    with open(flight_file, 'r', newline='', encoding='utf-8') as f_flight:
        flight_reader = csv.reader(f_flight)
        flight_data = list(flight_reader)
    
    # 读取result_car.csv
    with open(car_file, 'r', newline='', encoding='utf-8') as f_car:
        car_reader = csv.reader(f_car)
        car_data = list(car_reader)
    
    # 确保两个文件行数一致
    if len(flight_data) != len(car_data):
        raise ValueError("CSV files have different number of rows")
    
    # 创建合并后的数据
    merged_data = []
    
    # 处理标题行
    if flight_data and car_data:
        merged_data.append(flight_data[0])  # 使用任一文件的标题行
    
    # 合并数据行
    for i in range(1, len(flight_data)):
        flight_row = flight_data[i]
        car_row = car_data[i]
        
        # 确保列数一致
        if len(flight_row) != 4 or len(car_row) != 4:
            raise ValueError(f"Row {i} has inconsistent columns")
        
        # 处理I区水果种类：优先car，其次flight
        i_fruit = car_row[1] if car_row[1].strip() else flight_row[1]
        
        # 处理II区水果种类：优先car，其次flight
        ii_fruit = car_row[3] if car_row[3].strip() else flight_row[3]
        
        # 构建合并后的行（保留原始货架号格式）
        merged_row = [
            car_row[0],  # I区货架号（两个文件相同）
            i_fruit,
            car_row[2],  # II区货架号（两个文件相同）
            ii_fruit
        ]
        merged_data.append(merged_row)
    
    # 写入合并后的CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerows(merged_data)

if __name__ == "__main__":
    # 输入文件和输出文件配置
    flight_csv = "/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result_flight.csv"
    car_csv = "/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result_car.csv"
    output_csv = "/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result.csv"
    
    download_flight_result('/home/forlinx/catkin_ws/src/tta_m3e_rtsp/model_result/flight_result.csv',
                            '/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result_flight.csv')


    # 执行合并
    merge_csv_files(flight_csv, car_csv, output_csv)
    print(f"合并完成！结果已保存至: {output_csv}")