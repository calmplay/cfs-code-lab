import os
import h5py
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm
import sys

# 设置日志格式
logging.basicConfig(level=logging.INFO,
                    format=' %(levelname)s - %(message)s')


def process_folder(input_folder):
    """处理父文件夹下所有子文件夹"""
    # 创建父文件夹路径的_trans后缀版本
    output_folder = f"{input_folder}_trans"
    os.makedirs(output_folder, exist_ok=True)

    # 遍历父文件夹下的所有子文件夹
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)

        if os.path.isdir(subfolder_path):
            # 输出HDF5文件的路径
            output_h5_path = os.path.join(output_folder, f"{subfolder}.h5")

            # 获取该子文件夹中的所有JPEG、PNG文件
            image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # 如果子文件夹没有JPEG文件，记录日志
            if not image_files:
                logging.warning(f"No image files found in {subfolder_path}")
                continue

            # 读取每一张图片并转换为RGB
            images = []
            filenames = []  # 用来存储每张图片的名称
            for image_file in tqdm(image_files, desc=f"Processing {subfolder}", unit="file"):
                image_path = os.path.join(subfolder_path, image_file)
                try:
                    img = Image.open(image_path).convert('RGB')  # 确保是RGB模式
                    img_array = np.array(img)  # 转换为numpy数组
                    images.append(img_array)
                    filenames.append(image_file)  # 保存图片的名称
                except Exception as e:
                    logging.error(f"Error reading {image_path}: {e}")

            if images:
                images = np.array(images)

                # 将图片数据和文件名一起写入HDF5文件
                with h5py.File(output_h5_path, 'w') as f:
                    f.create_dataset('images', data=images, compression='gzip', compression_opts=9)
                    f.create_dataset('filenames', data=np.array(filenames, dtype='S100'))  # 保存文件名
                logging.info(f"Created {output_h5_path}")
            else:
                logging.warning(f"No valid images to save in {subfolder_path}")

    logging.info("Finished processing all subfolders.")


# 主函数，直接接受参数传入父文件夹路径
def main(input_folder):
    if os.path.isdir(input_folder):
        process_folder(input_folder)
    else:
        logging.error(f"错误: '{input_folder}' 不是有效的文件夹路径")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("使用方法: python script.py <父文件夹路径>")
    else:
        input_folder = sys.argv[1]
        main(input_folder)
