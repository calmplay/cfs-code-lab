# -*- coding: utf-8 -*-
# @Time    : 2026/4/21 11:47
# @Author  : CFuShn
# @Comments: HuggingFace 数据集结构分析工具 (对应 h5_inspect.py)
# @Software: PyCharm

"""
HuggingFace 数据集结构分析工具
================================
功能对标 h5_inspect.py, 但输入是 HF 数据集文件夹 (含多个 parquet 分片).

用法:
cd /home/cy/nuist-lab/cfs-code-lab/c00_utils

# OmniFace
python hf_inspect.py /home/data/HF/OmniFace64-V1_20260430
python hf_inspect.py /home/data/HF/OmniFace512-V1_20260430

# OmniShape
python hf_inspect.py /home/data/HF/OmniShape64-V1_20260430
python hf_inspect.py /home/data/HF/OmniShape128-V1_20260430
python hf_inspect.py /home/data/HF/OmniShape128-V1_test

# JSON 输出
python hf_inspect.py /home/data/HF/OmniFace64-V1_20260430 -j > structure.json

# 采样数量
python hf_inspect.py /home/data/HF/OmniShape128-V1_20260430 --sample 5

# ImageNet
python hf_inspect.py /home/cy/datasets/imagenet-1k
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

from datasets import load_dataset, Image as HFImage, Sequence, Value, ClassLabel


# ============================================================================
# Phase 1: 文件夹扫描
# ============================================================================

def scan_dataset_dir(dataset_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    扫描数据集文件夹, 按 split 分组 parquet 文件.

    Returns:
        {
            "train": {"files": [...], "num_shards": 139, "total_size": 68200000000},
            "val":   {"files": [...], "num_shards": 9,   "total_size": 3800000000},
        }
    """
    data_dir = os.path.join(dataset_dir, "data")
    if not os.path.isdir(data_dir):
        # 也可能 parquet 直接在根目录
        data_dir = dataset_dir

    parquet_files = sorted([
        f for f in os.listdir(data_dir) if f.endswith(".parquet")
    ])

    if not parquet_files:
        return {}

    # 按 split 分组: train-00000-of-00139.parquet -> split="train"
    split_pattern = re.compile(r"^(all|train|validation|val|test|dev|training|eval)-\d+")
    splits = defaultdict(lambda: {"files": [], "num_shards": 0, "total_size": 0})

    for fname in parquet_files:
        fpath = os.path.join(data_dir, fname)
        fsize = os.path.getsize(fpath)

        m = split_pattern.match(fname)
        if m:
            split_name = m.group(1)
            # 标准化 split 名称
            split_name = {"validation": "val", "training": "train", "eval": "test"}.get(
                split_name, split_name
            )
        else:
            # 不带 split 前缀的文件, 尝试用文件名作为 split
            split_name = fname.replace(".parquet", "")

        splits[split_name]["files"].append(fpath)
        splits[split_name]["total_size"] += fsize

    for split_name in splits:
        splits[split_name]["num_shards"] = len(splits[split_name]["files"])

    return dict(splits)


def load_dataset_infos(dataset_dir: str) -> Optional[dict]:
    """加载 dataset_infos.json (如果存在)."""
    fpath = os.path.join(dataset_dir, "dataset_infos.json")
    if os.path.isfile(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def check_meta_parquet(dataset_dir: str) -> Optional[Dict[str, Any]]:
    """检查 meta/ 目录下的 parquet 文件."""
    meta_dir = os.path.join(dataset_dir, "meta")
    if not os.path.isdir(meta_dir):
        return None
    
    meta_files = sorted([
        f for f in os.listdir(meta_dir) 
        if f.endswith(".parquet") and f.startswith("meta_")
    ])
    if not meta_files:
        return None
    
    total_size = sum(
        os.path.getsize(os.path.join(meta_dir, f)) 
        for f in meta_files
    )
    
    return {
        "dir": meta_dir,
        "files": meta_files,
        "num_files": len(meta_files),
        "total_size": total_size,
    }


# ============================================================================
# Phase 2: 数据集加载与字段分析
# ============================================================================

def _format_size(size_bytes: int) -> str:
    """格式化文件大小."""
    if size_bytes >= 1e12:
        return f"{size_bytes / 1e12:.2f} TB"
    elif size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.2f} GB"
    elif size_bytes >= 1e6:
        return f"{size_bytes / 1e6:.2f} MB"
    else:
        return f"{size_bytes / 1e3:.2f} KB"


def _feature_type_str(feature) -> str:
    """将 HF Feature 对象转为可读的类型字符串."""
    if isinstance(feature, HFImage):
        return "Image"
    elif isinstance(feature, ClassLabel):
        return f"ClassLabel(num_classes={feature.num_classes})"
    elif isinstance(feature, Sequence):
        inner = _feature_type_str(feature.feature)
        length = getattr(feature, "length", -1)
        if length > 0:
            return f"Sequence[{inner}, length={length}]"
        else:
            return f"Sequence[{inner}]"
    elif isinstance(feature, Value):
        return str(feature.dtype)
    else:
        return str(type(feature).__name__)


def analyze_split(
        files: List[str],
        sample_n: int = 3,
) -> Dict[str, Any]:
    """
    加载一个 split 并分析所有字段.

    Returns:
        {
            "num_samples": int,
            "features": {field_name: {info}},
        }
    """
    ds = load_dataset("parquet", data_files=files, split="train")
    num_samples = len(ds)
    features = ds.features

    fields_info = {}

    for field_name, feature in features.items():
        info: Dict[str, Any] = {
            "dtype": _feature_type_str(feature),
        }

        # 采样值
        if sample_n > 0 and num_samples > 0:
            try:
                sample_vals = ds[field_name][:sample_n]
                info["sample"] = _format_sample(sample_vals, feature)
            except Exception as e:
                info["sample_error"] = str(e)

        # Image 字段: 额外获取尺寸信息
        if isinstance(feature, HFImage):
            try:
                img = ds[field_name][0]
                if hasattr(img, "size"):
                    info["image_size"] = f"{img.size[0]}x{img.size[1]}"
                if hasattr(img, "mode"):
                    info["image_mode"] = img.mode
            except Exception:
                pass

        fields_info[field_name] = info

    return {
        "num_samples": num_samples,
        "features": fields_info,
    }


def _format_sample(values: Any, feature) -> Any:
    """格式化采样值, 使其可读."""
    if isinstance(feature, HFImage):
        # Image 不适合在文本中显示完整数据
        if hasattr(values[0], "size"):
            return [f"PIL Image {v.size[0]}x{v.size[1]} ({v.mode})" for v in values]
        return ["<PIL Image>"] * len(values)
    elif isinstance(feature, Sequence):
        # Sequence: 显示为 list，浮点数保留2位小数
        result = []
        for v in values:
            if isinstance(v, (list, tuple)):
                formatted = []
                for item in v:
                    if isinstance(item, float) or (
                            hasattr(item, 'item') and isinstance(item.item(), float)):
                        val = item.item() if hasattr(item, 'item') else item
                        formatted.append(round(val, 2))
                    else:
                        formatted.append(item)
                if len(formatted) > 5:
                    result.append(formatted[:5] + ["..."])
                else:
                    result.append(formatted)
            else:
                result.append(v)
        return result
    elif isinstance(feature, Value) and feature.dtype == "string":
        # 字符串: 截断过长的
        result = []
        for v in values:
            s = str(v)
            if len(s) > 50:
                s = s[:50] + "..."
            result.append(s)
        return result
    elif isinstance(feature, ClassLabel):
        # ClassLabel: 显示名称
        try:
            return [feature.int2str(int(v)) if v is not None else None for v in values]
        except Exception:
            return list(values)
    else:
        # 数值: 格式化显示，浮点数保留2位小数
        vals = list(values)
        result = []
        for v in vals:
            if hasattr(v, "item"):
                v = v.item()
            # 浮点数保留2位小数
            if isinstance(v, float):
                result.append(round(v, 2))
            else:
                result.append(v)
        return result


# ============================================================================
# Phase 3: 输出
# ============================================================================

def print_tree_result(
        dataset_dir: str,
        splits_info: Dict[str, Dict],
        split_analyses: Dict[str, Dict],
        dataset_infos: Optional[dict],
        meta_info: Optional[Dict[str, Any]],
):
    """树形文本输出."""
    print(f"# Dataset: {dataset_dir}")

    # Splits 概览
    split_summaries = []
    for split_name in sorted(splits_info.keys()):
        info = splits_info[split_name]
        size_str = _format_size(info["total_size"])
        split_summaries.append(f"{split_name} ({info['num_shards']} shards, {size_str})")
    print(f"# Splits: {', '.join(split_summaries)}")

    if dataset_infos:
        print(f"# dataset_infos.json: found")

    if meta_info:
        meta_size = _format_size(meta_info["total_size"])
        print(f"# meta/: {meta_info['num_files']} files, {meta_size}")

    print()

    # 每个 split 的字段详情
    for split_name in sorted(split_analyses.keys()):
        analysis = split_analyses[split_name]
        num_samples = analysis["num_samples"]
        fields = analysis["features"]

        print(f"// {split_name} ({num_samples:,} samples)")

        field_names = list(fields.keys())
        for i, fname in enumerate(field_names):
            finfo = fields[fname]
            is_last = (i == len(field_names) - 1)
            prefix = "└─ " if is_last else "├─ "

            # 构建描述行
            parts = [f"{fname} ({finfo['dtype']})"]

            # Image 额外信息
            if "image_size" in finfo:
                parts.append(f"size={finfo['image_size']}")
            if "image_mode" in finfo:
                parts.append(f"mode={finfo['image_mode']}")

            # 采样值
            if "sample" in finfo:
                sample = finfo["sample"]
                sample_str = str(sample)
                if len(sample_str) > 100:
                    sample_str = sample_str[:100] + "..."
                parts.append(f"sample={sample_str}")

            print(f"{prefix}{parts[0]} :: " + "; ".join(parts[1:]))

        print()

    # 总大小
    total_size = sum(info["total_size"] for info in splits_info.values())
    print(f"# Total size: {_format_size(total_size)}")


def build_json_result(
        dataset_dir: str,
        splits_info: Dict[str, Dict],
        split_analyses: Dict[str, Dict],
        dataset_infos: Optional[dict],
        meta_info: Optional[Dict[str, Any]],
) -> dict:
    """构建 JSON 输出的结构化数据."""
    result = {
        "dataset_dir": dataset_dir,
        "splits": {},
        "total_size": sum(info["total_size"] for info in splits_info.values()),
    }

    for split_name in sorted(splits_info.keys()):
        info = splits_info[split_name]
        result["splits"][split_name] = {
            "num_shards": info["num_shards"],
            "total_size": info["total_size"],
            "files": [os.path.basename(f) for f in info["files"]],
        }
        if split_name in split_analyses:
            result["splits"][split_name]["num_samples"] = split_analyses[split_name]["num_samples"]
            result["splits"][split_name]["features"] = split_analyses[split_name]["features"]

    if dataset_infos:
        result["dataset_infos"] = dataset_infos

    if meta_info:
        result["meta"] = meta_info

    return result


# ============================================================================
# Phase 4: CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace 数据集结构分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset_dir", help="数据集文件夹路径")
    parser.add_argument("-j", "--json", action="store_true", help="JSON 输出")
    parser.add_argument("--sample", type=int, default=3, help="采样数量 (默认 3)")
    parser.add_argument("--split", type=str, default=None, help="只查看指定 split")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)

    # Phase 1: 扫描
    print(f"扫描数据集目录: {dataset_dir}", file=sys.stderr)
    splits_info = scan_dataset_dir(dataset_dir)

    if not splits_info:
        print(f"[ERROR] 未找到 parquet 文件: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    dataset_infos = load_dataset_infos(dataset_dir)
    meta_info = check_meta_parquet(dataset_dir)

    # 过滤 split
    if args.split:
        if args.split not in splits_info:
            print(f"[ERROR] split '{args.split}' 不存在. 可用: {list(splits_info.keys())}",
                  file=sys.stderr)
            sys.exit(1)
        splits_to_analyze = {args.split: splits_info[args.split]}
    else:
        splits_to_analyze = splits_info

    # Phase 2: 分析每个 split
    split_analyses = {}
    for split_name, info in splits_to_analyze.items():
        print(f"  分析 {split_name} ({info['num_shards']} shards)...", file=sys.stderr)
        analysis = analyze_split(info["files"], sample_n=args.sample)
        split_analyses[split_name] = analysis

    # Phase 3: 输出
    if args.json:
        result = build_json_result(dataset_dir, splits_info, split_analyses, dataset_infos,
                                   meta_info)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print_tree_result(dataset_dir, splits_to_analyze, split_analyses, dataset_infos,
                          meta_info)


if __name__ == "__main__":
    main()
