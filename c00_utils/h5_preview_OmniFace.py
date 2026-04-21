# -*- coding: utf-8 -*-
# @Time    : 2025/9/16 15:40
# @Author  : CFuShn
# @Comments: 
# @Software: PyCharm

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 OmniFace h5 文件：
- images 按 JPEG 字节流存储 (dtype=object)
- 随机抽取 N 张，解码后可视化 preview
- 显示对应的 id / age
"""

import argparse
import io
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random


def main():
    parser = argparse.ArgumentParser(description="验证 OmniFace H5 文件 (JPEG 字节流)")
    parser.add_argument("--input", default="/home/data/OmniFace_202602042244.h5", help="h5 文件路径")
    parser.add_argument("--num", "-n", type=int, default=10, help="抽取数量 (默认10)")
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f:
        images = f["images"]   # (N,), 每个元素是 JPEG 字节流
        ids = f["id"]          # (N,), bytes
        ages = f["age"]        # (N,), float32

        N = images.shape[0]
        idxs = random.sample(range(N), args.num)

        print(f"数据集大小: {N}, 随机抽取 {args.num} 个样本")

        plt.figure(figsize=(15, 6))
        for i, idx in enumerate(idxs):
            jpeg_bytes = images[idx]
            img = np.array(Image.open(io.BytesIO(jpeg_bytes)))  # (H, W, 3) RGB

            uid = ids[idx]
            if isinstance(uid, bytes):
                uid = uid.decode("utf-8", errors="replace")
            age = ages[idx]

            print(f"样本 {i}: index={idx}, shape={img.shape}, id={uid}, age={age}")

            plt.subplot(2, (args.num + 1) // 2, i + 1)
            plt.imshow(img)
            plt.title(f"\n{uid}\nage={age:.1f}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

    """
    python h5_preview_OmniFace.py --input /home/data/OmniFace_202602042244.h5 --num 10
    """