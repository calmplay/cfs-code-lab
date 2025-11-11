# -*- coding: utf-8 -*-
# @Time    : 2025/6/11 21:51
# @Author  : CFuShn
# @Comments: 
# @Software: PyCharm

import base64
import io
import os
import re
import shutil
import subprocess
import tarfile
import zipfile

import cv2
import numpy as np
import pandas as pd
import py7zr
from PIL import Image

__all__ = ["extract"]  # 使得调用方 import * 只导入指定项


def extract(src_file, target_dir, clean_first=False, image_col_of_parquet='image',
        path_col_of_parquet='path'):
    if clean_first and os.path.exists(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)

    if os.path.exists(target_dir) and any(os.scandir(target_dir)):
        return

    os.makedirs(target_dir, exist_ok=True)

    base_dir = os.path.dirname(src_file)
    base_name = os.path.basename(src_file)

    if base_name.endswith(".parquet") or base_name.endswith(".parquet.gzip"):
        """
        parquet提取
        图片将保存至 <target_dir>/data
        csv将保存至 <target_dir>
        """
        print(f"正在提取: {src_file} 等文件...")
        parquet_dir = base_dir
        all_records = []
        seen_rel_paths = {}
        decode_func = None

        for filename in os.listdir(parquet_dir):
            if not filename.endswith(".parquet"):
                continue
            df = pd.read_parquet(os.path.join(parquet_dir, filename))

            if decode_func is None:
                # 在第一行上探测并定义解码逻辑
                sample_data = df.iloc[0][image_col_of_parquet]

                def detect_decoder(data):
                    if isinstance(data, np.ndarray):
                        return lambda d: d[..., ::-1]
                    elif isinstance(data, bytes):
                        return lambda d: cv2.imdecode(np.frombuffer(d, dtype=np.uint8),
                                                      cv2.IMREAD_COLOR)
                    elif isinstance(data, dict):
                        if "bytes" in data:
                            return lambda d: cv2.imdecode(np.frombuffer(d["bytes"], dtype=np.uint8),
                                                          cv2.IMREAD_COLOR)
                        elif "uri" in data:
                            return lambda d: cv2.imread(d["uri"])
                        elif "encoded" in data:
                            return lambda d: cv2.imdecode(
                                    np.frombuffer(base64.b64decode(d["encoded"]), dtype=np.uint8),
                                    cv2.IMREAD_COLOR)
                        elif "data" in data:
                            return lambda d: cv2.cvtColor(
                                    np.array(Image.open(io.BytesIO(d["data"])).convert("RGB")),
                                    cv2.COLOR_RGB2BGR)
                        else:
                            return None
                    return None

                decode_func = detect_decoder(sample_data)

            if decode_func is None:
                print("未能识别图像数据格式，跳过该文件。")
                continue

            for idx, row in df.iterrows():
                if path_col_of_parquet in row:
                    rel_path = row[path_col_of_parquet]
                elif (isinstance(row[image_col_of_parquet], dict)
                      and path_col_of_parquet in row[image_col_of_parquet]):
                    rel_path = row[image_col_of_parquet][path_col_of_parquet]
                else:
                    rel_path = f"{filename.replace('.parquet', '')}_{idx:06d}.png"

                rel_path = str(rel_path).lstrip("/")

                # 强制 .png 扩展名，去除原始扩展
                rel_path_no_ext = os.path.splitext(rel_path)[0]
                # 判断是否重复
                base_key = rel_path_no_ext
                count = seen_rel_paths.get(base_key, 0)
                if count > 0:
                    rel_path_no_ext = f"{base_key}_{count}"
                seen_rel_paths[base_key] = count + 1

                rel_path_png = rel_path_no_ext + ".jpg"
                os.makedirs(os.path.join(target_dir, "data"), exist_ok=True)
                full_path = os.path.join(target_dir, "data", rel_path_png)

                try:
                    img = decode_func(row[image_col_of_parquet])
                    if img is not None:
                        cv2.imwrite(full_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    else:
                        raise ValueError("解码函数返回了 None")
                except Exception as e:
                    print(f"图像解码失败: {rel_path} {e}")
                    continue

                meta_row = {}
                for col in df.columns:
                    if col == image_col_of_parquet:
                        continue
                    val = row[col]
                    if isinstance(val, dict):
                        for k, v in val.items():
                            meta_row[f"{col}.{k}"] = v
                    else:
                        meta_row[col] = val
                meta_row['path'] = rel_path_png
                all_records.append(meta_row)

        if all_records:
            pd.DataFrame(all_records).to_csv(os.path.join(target_dir, "meta.csv"), index=False)
            print(f"提取完成,共导出 {len(all_records)} 张图像及其元信息至: {target_dir}")
        else:
            print("未找到有效图像记录。")

    else:
        # 压缩文件提取
        print(f"正在解压: {src_file}")
        if base_name.endswith(".zip"):
            with zipfile.ZipFile(src_file, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        elif base_name.endswith(('.tar', '.tar.gz', '.tgz')):
            with tarfile.open(src_file, 'r:*') as tar_ref:
                tar_ref.extractall(target_dir)
        elif base_name.endswith(".7z"):
            with py7zr.SevenZipFile(src_file, mode='r') as z:
                z.extractall(path=target_dir)
        elif re.match(r".*\.7z\.\d{3}$", base_name):
            print("检测到多卷 .7z 文件，尝试调用系统命令解压:")
            try:
                command = ["7z", "x", src_file, f"-o{target_dir}"]
                print({' '.join(command)})
                result = subprocess.run(command, capture_output=True, text=True)
            except FileNotFoundError:
                raise RuntimeError("未找到 `7z` 解压命令，请先安装并确保其在系统 PATH 中。\n")
            if result.returncode != 0:
                raise RuntimeError(
                        f"7z 解压执行失败（{result.returncode}）:\n{result.stderr.strip()}")
        else:
            raise ValueError(f"不支持的文件格式: {src_file}")
        print(f"解压完成: {target_dir}")
