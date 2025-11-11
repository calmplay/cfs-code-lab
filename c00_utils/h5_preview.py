# -*- coding: utf-8 -*-
# @Time    : 2025/9/16 15:40
# @Author  : CFuShn
# @Comments: 
# @Software: PyCharm

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 h5 文件：
- 随机抽取 10 张 images
- 打印 shape 和对应的 label
- 可视化 preview
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random


def main():
    parser = argparse.ArgumentParser(description="验证 H5 文件的 images/labels")
    parser.add_argument("--input",default="/home/cy/datasets/CCGM/SteeringAngle_256x256.h5", help="h5 文件路径")
    parser.add_argument("--num", "-n", type=int, default=10, help="抽取数量 (默认10)")
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f:
        images = f["images"]   # (N,3,H,W), RGB-CHW
        labels = f["labels"]   # (N,)

        N = images.shape[0]
        idxs = random.sample(range(N), args.num)

        print(f"数据集大小: {N}, 随机抽取 {args.num} 个样本")

        plt.figure(figsize=(15, 6))
        for i, idx in enumerate(idxs):
            img_chw = images[idx]  # (3,H,W)
            label = labels[idx]

            print(f"样本 {i}: index={idx}, shape={img_chw.shape}, label={label}")

            # 转成 HWC 显示
            img_hwc = np.transpose(img_chw, (1, 2, 0))

            plt.subplot(2, (args.num + 1)//2, i+1)
            plt.imshow(img_hwc)
            plt.title(f"idx {idx}\nlabel {label:.2f}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

    """
    python h5_preview.py --input /path/to/output_sr.h5 --num 10
    """