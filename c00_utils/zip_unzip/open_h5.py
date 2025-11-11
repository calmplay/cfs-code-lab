import os
import h5py
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm
import sys

# 设置日志格式
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def extract_images_from_h5(h5_file, output_folder):
    """从HDF5文件中提取图像并保存为JPEG，确保图像名称一致"""
    try:
        logging.info(f"正在打开HDF5文件: {h5_file}")
        with h5py.File(h5_file, 'r') as f:
            # 调试：打印HDF5文件中的所有键名
            logging.info(f"HDF5文件内容: {list(f.keys())}")

            # 检查是否存在'images'数据集
            if 'images' not in f:
                logging.error(f"错误: HDF5文件中未找到'images'数据集")
                logging.info(f"可用的数据集: {list(f.keys())}")
                return

            images = f['images']
            filenames = f['filenames'][:]  # 获取文件名
            logging.info(f"数据集形状: {images.shape}, 数据类型: {images.dtype}")

            # 创建输出文件夹，以HDF5文件名命名（不含扩展名）
            folder_name = os.path.splitext(os.path.basename(h5_file))[0]
            folder_output_path = os.path.join(output_folder, folder_name)
            os.makedirs(folder_output_path, exist_ok=True)
            logging.info(f"创建输出文件夹: {folder_output_path}")

            # 使用tqdm显示拆包进度条
            for idx, (img_array, filename) in tqdm(enumerate(zip(images, filenames)), total=len(images),
                                                   desc=f"Extracting {folder_name}"):
                # 确保数组数据类型正确
                if img_array.dtype != np.uint8:
                    logging.warning(f"图像{idx}数据类型为{img_array.dtype}，正在转换为uint8")
                    # 根据数据范围进行适当的缩放和转换
                    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                        if img_array.max() <= 1.0:  # 假设是0-1范围的浮点数
                            img_array = (img_array * 255).astype(np.uint8)
                        else:  # 假设是0-255范围的浮点数
                            img_array = img_array.astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)

                # 检查数组维度
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    # RGB图像
                    img = Image.fromarray(img_array, 'RGB')
                elif len(img_array.shape) == 2:
                    # 灰度图像
                    img = Image.fromarray(img_array, 'L')
                else:
                    logging.warning(f"图像{idx}的维度异常: {img_array.shape}，尝试直接转换")
                    img = Image.fromarray(img_array)

                # 使用从HDF5中读取的文件名
                img_filename = filenames[idx].decode('utf-8')  # 从字节解码为字符串
                img_path = os.path.join(folder_output_path, img_filename)
                img.save(img_path, format='JPEG', quality=100)
                logging.debug(f"保存图像: {img_path}")

            logging.info(f"成功从{h5_file}提取{len(images)}张图像")

    except Exception as e:
        logging.error(f"处理文件{h5_file}时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())


def find_h5_files(input_folder):
    """递归查找输入文件夹中的所有HDF5文件"""
    h5_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.h5', '.hdf5')):
                h5_files.append(os.path.join(root, file))
    return h5_files


def process_folder(input_folder):
    """处理父文件夹中所有的HDF5文件"""
    output_folder = f"{input_folder}_extracted"
    os.makedirs(output_folder, exist_ok=True)
    logging.info(f"输出文件夹: {output_folder}")

    # 查找所有的HDF5文件
    h5_files = find_h5_files(input_folder)

    if not h5_files:
        logging.error(f"在文件夹 {input_folder} 中未找到任何HDF5文件")
        logging.info("支持的HDF5文件扩展名: .h5, .hdf5")
        return

    logging.info(f"找到 {len(h5_files)} 个HDF5文件")

    for h5_file in h5_files:
        logging.info(f"处理HDF5文件: {h5_file}")
        extract_images_from_h5(h5_file, output_folder)

    logging.info(f"图像提取完成！文件保存在: {output_folder}")


# 主函数，直接接受参数传入父文件夹路径
def main(input_folder):
    if os.path.isdir(input_folder):
        process_folder(input_folder)
    else:
        logging.error(f"错误: '{input_folder}' 不是有效的文件夹路径")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("使用方法: python script.py <包含HDF5文件的文件夹路径>")
        logging.info("示例: python script.py /path/to/your/folder")
        sys.exit(1)
    else:
        input_folder = sys.argv[1]
        main(input_folder)
