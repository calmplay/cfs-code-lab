# -*- coding: utf-8 -*-
# @Time    : 2026/4/21 11:47
# @Author  : CFuShn
# @Comments: HuggingFace 数据集预览工具 (对应 h5_preview.py)
# @Software: PyCharm

"""
HuggingFace 数据集预览工具
=============================
功能对标 h5_preview.py, 但输入是 HF 数据集文件夹 (含多个 parquet 分片).

用法:
  cd /home/cy/nuist-lab/cfs-code-lab/c00_utils
  python hf_preview.py /home/data/HF/OmniFace --num 10
  python hf_preview.py /home/data/HF/OmniShape --num 10
  python hf_preview.py /home/data/HF/OmniFace --split val --num 5
  python hf_preview.py /home/data/HF/OmniFace --label-fields id,age --num 10
"""

import argparse
import os
import random
import re
import sys
from collections import defaultdict
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

# 关闭 toolbar 和刻度
plt.rcParams["toolbar"] = "None"
from datasets import load_dataset, Image as HFImage, Sequence, Value, ClassLabel


def main():
    parser = argparse.ArgumentParser(description="HuggingFace 数据集预览工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument("dataset_dir", type=str, nargs="?", default="/home/data/HF/OmniFace",
                        help="数据集文件夹路径")
    parser.add_argument("--label-fields", type=str, default="id,age",
                        help="手动指定显示的标签字段 (逗号分隔, 默认自动选择)", )
    parser.add_argument("--num", "-n", type=int, default=10, help="抽取数量 (默认 10)")
    parser.add_argument("--split", type=str, default="train", help="指定 split (默认 train)")

    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    label_fields = None
    if args.label_fields:
        label_fields = [f.strip() for f in args.label_fields.split(",")]

    preview_dataset(
        dataset_dir=dataset_dir,
        split_name=args.split,
        num=args.num,
        label_fields=label_fields,
    )


# ============================================================================
# Phase 1: 文件夹扫描与数据集加载
# ============================================================================

def scan_splits(dataset_dir: str) -> dict:
    """扫描 data/ 目录, 按 split 分组 parquet 文件."""
    data_dir = os.path.join(dataset_dir, "data")
    if not os.path.isdir(data_dir):
        data_dir = dataset_dir

    parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet")])
    if not parquet_files:
        return {}

    split_pattern = re.compile(r"^(train|validation|val|test|dev|training|eval)-\d+")
    splits = defaultdict(list)

    for fname in parquet_files:
        fpath = os.path.join(data_dir, fname)
        m = split_pattern.match(fname)
        if m:
            split_name = m.group(1)
            split_name = {"validation": "val", "training": "train", "eval": "test"}.get(
                split_name, split_name
            )
        else:
            split_name = fname.replace(".parquet", "")
        splits[split_name].append(fpath)

    return dict(splits)


def load_split(dataset_dir: str, split_name: str) -> object:
    """加载指定 split 的数据集."""
    splits = scan_splits(dataset_dir)

    if not splits:
        print(f"[ERROR] 未找到 parquet 文件: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    if split_name not in splits:
        print(f"[ERROR] split '{split_name}' 不存在. 可用: {list(splits.keys())}", file=sys.stderr)
        sys.exit(1)

    files = splits[split_name]
    ds = load_dataset("parquet", data_files=files, split="train")
    return ds


# ============================================================================
# Phase 2: 智能标签选择
# ============================================================================

def auto_select_label_fields(ds) -> List[str]:
    """
    自动选择适合显示在 title 中的标签字段.
    策略: 优先 string, 其次 int/float 标量, 跳过 Image 和 Sequence.
    最多选 2 个.
    """
    features = ds.features
    string_fields = []
    scalar_fields = []

    for fname, feat in features.items():
        if isinstance(feat, HFImage):
            continue
        if isinstance(feat, Sequence):
            continue
        if isinstance(feat, ClassLabel):
            scalar_fields.append(fname)
        elif isinstance(feat, Value):
            if feat.dtype == "string":
                string_fields.append(fname)
            elif feat.dtype in ("int32", "int64", "float32", "float64"):
                scalar_fields.append(fname)

    # 优先 string, 其次 scalar, 最多 2 个
    selected = string_fields[:2]
    if len(selected) < 2:
        remaining = [f for f in scalar_fields if f not in selected]
        selected.extend(remaining[:2 - len(selected)])

    return selected


# ============================================================================
# Phase 3: 可视化
# ============================================================================

def find_image_field(ds) -> Optional[str]:
    """找到 Image 类型的字段名."""
    for fname, feat in ds.features.items():
        if isinstance(feat, HFImage):
            return fname
    return None


def format_label_value(val, feat) -> str:
    """格式化单个标签值用于显示."""
    if isinstance(feat, ClassLabel):
        try:
            return str(feat.int2str(int(val)))
        except Exception:
            return str(val)
    elif isinstance(feat, Value) and feat.dtype == "string":
        s = str(val)
        return s if len(s) <= 30 else s[:30] + "..."
    else:
        return str(val)


def preview_dataset(
        dataset_dir: str,
        split_name: str = "train",
        num: int = 10,
        label_fields: Optional[List[str]] = None,
):
    """加载并预览数据集."""
    print(f"加载数据集: {dataset_dir} [{split_name}]", file=sys.stderr)
    ds = load_split(dataset_dir, split_name)

    N = len(ds)
    print(f"数据集大小: {N:,}, 随机抽取 {num} 个样本", file=sys.stderr)

    # 找到 image 字段
    img_field = find_image_field(ds)
    if img_field is None:
        print("[ERROR] 未找到 Image 类型的字段", file=sys.stderr)
        sys.exit(1)

    # 选择标签字段
    if label_fields is None:
        label_fields = auto_select_label_fields(ds)

    print(f"图片字段: {img_field}, 标签字段: {label_fields}", file=sys.stderr)

    # 随机采样
    idxs = random.sample(range(N), min(num, N))

    # 可视化
    cols = min(num, 5)
    rows = (num + cols - 1) // cols
    plt.figure(figsize=(3 * cols, 3.5 * rows))

    for i, idx in enumerate(idxs):
        sample = ds[idx]
        img = sample[img_field]

        # 打印信息
        info_parts = [f"样本 {i}: index={idx}"]
        if hasattr(img, "size"):
            info_parts.append(f"size={img.size[0]}x{img.size[1]}")
        for lf in label_fields:
            if lf in sample:
                val = sample[lf]
                feat = ds.features[lf]
                info_parts.append(f"{lf}={format_label_value(val, feat)}")
        print(" | ".join(info_parts))

        # 显示图片 (转 numpy array, 避免 PyCharm SciView 渲染空白)
        plt.subplot(rows, cols, i + 1)
        if hasattr(img, "size"):
            img = np.array(img)
        plt.imshow(img)

        # title: 标签字段值
        title_parts = []
        for lf in label_fields:
            if lf in sample:
                val = sample[lf]
                feat = ds.features[lf]
                title_parts.append(format_label_value(val, feat))
        plt.title("\n".join(title_parts) if title_parts else "", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
