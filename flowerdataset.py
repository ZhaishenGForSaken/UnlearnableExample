import os
import shutil

# 设定你的数据集所在的路径
dataset_path = 'data/datasets/classification/flowers/flowers'

# 遍历数据集目录中的所有文件
for filename in os.listdir(dataset_path):
    if filename.endswith('.png'):  # 确认文件是JPG格式
        prefix = filename.split('_')[0]  # 分离出前缀XX
        new_folder_path = os.path.join(dataset_path, prefix)  # 创建新的文件夹路径

        if not os.path.exists(new_folder_path):  # 如果文件夹不存在，则创建
            os.makedirs(new_folder_path)

        # 移动文件到新的文件夹
        shutil.move(os.path.join(dataset_path, filename), os.path.join(new_folder_path, filename))
