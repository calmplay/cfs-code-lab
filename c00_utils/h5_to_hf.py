# -*- coding: utf-8 -*-
# @Time    : 2026/4/20 17:48
# @Author  : CFuShn
# @Comments: H5 -> HuggingFace datasets (parquet shards) 转换工具
# @Software: PyCharm

"""
H5 -> HuggingFace Datasets (Parquet 分片) 转换工具
===================================================

cd /home/cy/nuist-lab/cfs-code-lab/c00_utils

# 转换 OmniFace
python h5_to_hf.py -d omniface \
    -i /home/data/OmniFace_202602042244.h5 \
    -o /home/data/HF/OmniFace

# 转换 OmniShape
python h5_to_hf.py -d omnishape \
    -i /home/data/OmniShape1k_18000a_128x128_20251204.h5 \
    -o /home/data/HF/OmniShape

# 验证
python h5_to_hf.py -d omniface \
    -i /home/data/OmniFace_202602042244.h5 \
    -o /home/data/HF/OmniFace --verify

支持数据集:
  - OmniFace_202602042244.h5   (JPEG 字节流, 1,121,349 张)
  - OmniShape1k_18000a_128x128_20251204.h5  (uint8 CHW, 18,000,000 张)

用法:
  # 转换 OmniFace
  python h5_to_hf.py --dataset omniface \
      --input /home/data/OmniFace_202602042244.h5 \
      --output /home/data/HF/OmniFace

  # 转换 OmniShape
  python h5_to_hf.py --dataset omnishape \
      --input /home/data/OmniShape1k_18000a_128x128_20251204.h5 \
      --output /home/data/HF/OmniShape

  # 转换后验证
  python h5_to_hf.py --dataset omniface \
      --input /home/data/OmniFace_202602042244.h5 \
      --output /home/data/HF/OmniFace \
      --verify

  # 自定义参数
  python h5_to_hf.py --dataset omnishape \
      --input /path/to/file.h5 \
      --output /path/to/output \
      --shard-size 500 \
      --jpeg-quality 95 \
      --seed 42
"""

import argparse
import io
import json
import os
import sys
import time
from multiprocessing import Pool
from typing import Any, Dict, Generator, List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets import Dataset, Features, Image as HFImage, Value, Sequence


# ============================================================================
# Phase 1: 通用工具函数
# ============================================================================

def _encode_single_jpeg(args):
    """worker 函数: (hwc_array, quality) -> jpeg_bytes"""
    img_hwc, quality = args
    pil_img = Image.fromarray(img_hwc)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def encode_jpegs_parallel(
    imgs_chw: np.ndarray,
    quality: int = 95,
    workers: int = 16,
) -> List[bytes]:
    """
    批量将 CHW uint8 数组编码为 JPEG bytes (多进程并行).

    Args:
        imgs_chw: (B, 3, H, W) uint8 数组
        quality: JPEG 质量
        workers: 并行进程数

    Returns:
        JPEG bytes 列表
    """
    # 批量 transpose: (B,3,H,W) -> (B,H,W,3)
    imgs_hwc = np.transpose(imgs_chw, (0, 2, 3, 1)).copy()  # contiguous
    B = imgs_hwc.shape[0]

    if workers <= 1 or B <= 64:
        # 小批量或单进程: 直接串行
        return [_encode_single_jpeg((imgs_hwc[i], quality)) for i in range(B)]

    # 多进程并行
    tasks = [(imgs_hwc[i], quality) for i in range(B)]
    with Pool(processes=workers) as pool:
        return pool.map(_encode_single_jpeg, tasks, chunksize=max(1, B // (workers * 4)))


def bytes_to_str(val: Any) -> str:
    """numpy bytes / python bytes -> str"""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.ndarray) and val.dtype.kind in ("S", "O"):
        return str(val)
    return str(val)


def build_features(fields_config: Dict[str, dict]) -> Features:
    """
    根据字段配置构建 HF Features 对象.

    fields_config 示例:
        {
            "image":     {"type": "image"},
            "age":       {"type": "float32"},
            "male":      {"type": "int64"},
            "id":        {"type": "string"},
            "gaze_dir":  {"type": "sequence_float", "length": 2},
        }
    """
    feat_dict = {}
    for name, cfg in fields_config.items():
        t = cfg["type"]
        if t == "image":
            feat_dict[name] = HFImage()
        elif t == "string":
            feat_dict[name] = Value("string")
        elif t == "int8":
            feat_dict[name] = Value("int32")   # HF 对 int8 支持有限, 升格
        elif t in ("int32", "int64"):
            feat_dict[name] = Value(t)
        elif t == "float32":
            feat_dict[name] = Value("float32")
        elif t == "sequence_float":
            length = cfg.get("length", -1)
            if length > 0:
                feat_dict[name] = Sequence(Value("float32"), length=length)
            else:
                feat_dict[name] = Sequence(Value("float32"))
        elif t == "sequence_int":
            length = cfg.get("length", -1)
            if length > 0:
                feat_dict[name] = Sequence(Value("int32"), length=length)
            else:
                feat_dict[name] = Sequence(Value("int32"))
        else:
            raise ValueError(f"不支持的字段类型: {t} (字段: {name})")
    return Features(feat_dict)


def write_shards(
    batch_generator: Generator[Dict[str, list], None, None],
    output_dir: str,
    split_name: str,
    features: Features,
    max_shard_size_mb: int = 500,
    total_samples: int = 0,
) -> List[str]:
    """
    通用分片写入函数.

    Args:
        batch_generator: 每次产出一个 batch 的 dict (字段名 -> list of values)
        output_dir:     输出目录 (data/ 子目录)
        split_name:     "train" / "val"
        features:       HF Features 对象
        max_shard_size_mb: 每个分片最大大小 (MB)
        total_samples:  总样本数 (用于进度显示)

    Returns:
        写入的 parquet 文件路径列表
    """
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 第一遍: 收集所有 batch, 按 shard 大小切分
    # 策略: 累积 batch, 当估算大小超过阈值时写入一个 shard
    shard_paths = []
    current_batch = None
    current_size_bytes = 0
    shard_idx = 0
    processed = 0
    t0 = time.time()

    pbar = tqdm(
        total=total_samples,
        desc=f"  [{split_name}]",
        unit="img",
        unit_scale=True,
        ncols=120,
        file=sys.stdout,
    )

    def _estimate_batch_size(batch: Dict[str, list]) -> int:
        """粗略估算一个 batch 占用的字节数"""
        size = 0
        n = len(next(iter(batch.values())))
        for k, v in batch.items():
            if isinstance(v[0], bytes):
                size += sum(len(x) for x in v)
            elif isinstance(v[0], np.ndarray):
                size += sum(x.nbytes for x in v)
            elif isinstance(v[0], (list, tuple)):
                size += sum(
                    sum(np.array(item).nbytes if not isinstance(item, (int, float)) else 8 for item in row)
                    for row in v
                )
            elif isinstance(v[0], str):
                size += sum(len(s.encode("utf-8")) for s in v)
            else:
                # 标量
                size += n * 8
        return size

    def _flush_batch(batch: Dict[str, list]):
        nonlocal shard_idx, shard_paths
        n_shards = 0  # 占位, 后续 rename 时填入正确总数
        fname = f"{split_name}-{shard_idx:05d}-of-{n_shards:05d}.parquet"
        fpath = os.path.join(data_dir, fname)
        ds = Dataset.from_dict(batch, features=features)
        ds.to_parquet(fpath)
        shard_paths.append(fpath)
        shard_idx += 1

    for batch in batch_generator:
        batch_size = len(next(iter(batch.values())))
        batch_bytes = _estimate_batch_size(batch)

        # 如果当前累积为空, 直接赋值
        if current_batch is None:
            current_batch = batch
            current_size_bytes = batch_bytes
        else:
            # 追加到当前 batch
            for k in current_batch:
                current_batch[k].extend(batch[k])
            current_size_bytes += batch_bytes

        processed += batch_size
        pbar.update(batch_size)

        # 超过阈值, 写入 shard
        if current_size_bytes >= max_shard_size_mb * 1024 * 1024:
            _flush_batch(current_batch)
            current_batch = None
            current_size_bytes = 0

            elapsed = time.time() - t0
            speed = processed / elapsed if elapsed > 0 else 0
            eta = (total_samples - processed) / speed if speed > 0 and total_samples > 0 else 0
            pbar.set_postfix(
                shard=shard_idx,
                speed=f"{speed:.0f}/s",
                eta=f"{eta/60:.1f}min",
            )

    # 写入剩余数据
    if current_batch is not None and len(next(iter(current_batch.values()))) > 0:
        _flush_batch(current_batch)

    pbar.close()

    # 重命名所有分片, 填入正确的总数
    total_shards = len(shard_paths)
    for i, fpath in enumerate(shard_paths):
        new_fname = f"{split_name}-{i:05d}-of-{total_shards:05d}.parquet"
        new_fpath = os.path.join(data_dir, new_fname)
        if fpath != new_fpath:
            os.rename(fpath, new_fpath)
            shard_paths[i] = new_fpath

    elapsed = time.time() - t0
    print(
        f"  [{split_name}] 完成! {processed:,} 样本 -> {total_shards} 个分片, "
        f"耗时 {elapsed/60:.1f} min",
        flush=True,
    )

    return shard_paths


def save_dataset_infos(
    output_dir: str,
    dataset_name: str,
    features: Features,
    splits_info: Dict[str, dict],
):
    """
    生成 dataset_infos.json (HF datasets 兼容格式).
    """
    infos = {
        dataset_name: {
            "features": features.to_dict(),
            "splits": splits_info,
        }
    }
    fpath = os.path.join(output_dir, "dataset_infos.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=2, default=str)
    print(f"  已保存 {fpath}")


# ============================================================================
# Phase 2: OmniFace 转换
# ============================================================================

# OmniFace 字段配置
OMNIFACE_FIELDS = {
    "image":              {"type": "image"},
    "id":                 {"type": "string"},
    "origin":             {"type": "string"},
    "prompt":             {"type": "string"},
    "age":                {"type": "float32"},
    "arousal":            {"type": "float32"},
    "valence":            {"type": "float32"},
    "is_sr":              {"type": "int8"},
    "expression":         {"type": "int64"},
    "race":               {"type": "int64"},
    "male":               {"type": "int64"},
    "arched_eyebrows":    {"type": "int64"},
    "attractive":         {"type": "int64"},
    "bags_under_eyes":    {"type": "int64"},
    "bald":               {"type": "int64"},
    "bangs":              {"type": "int64"},
    "big_lips":           {"type": "int64"},
    "big_nose":           {"type": "int64"},
    "black_hair":         {"type": "int64"},
    "blond_hair":         {"type": "int64"},
    "blurry":             {"type": "int64"},
    "brown_hair":         {"type": "int64"},
    "bushy_eyebrows":     {"type": "int64"},
    "chubby":             {"type": "int64"},
    "double_chin":        {"type": "int64"},
    "eyeglasses":         {"type": "int64"},
    "five_o_clock_shadow":{"type": "int64"},
    "goatee":             {"type": "int64"},
    "gray_hair":          {"type": "int64"},
    "heavy_makeup":       {"type": "int64"},
    "high_cheekbones":    {"type": "int64"},
    "mouth_slightly_open":{"type": "int64"},
    "mustache":           {"type": "int64"},
    "narrow_eyes":        {"type": "int64"},
    "no_beard":           {"type": "int64"},
    "oval_face":          {"type": "int64"},
    "pale_skin":          {"type": "int64"},
    "pointy_nose":        {"type": "int64"},
    "receding_hairline":  {"type": "int64"},
    "rosy_cheeks":        {"type": "int64"},
    "sideburns":          {"type": "int64"},
    "smiling":            {"type": "int64"},
    "straight_hair":      {"type": "int64"},
    "wavy_hair":          {"type": "int64"},
    "wearing_earrings":   {"type": "int64"},
    "wearing_hat":        {"type": "int64"},
    "wearing_lipstick":   {"type": "int64"},
    "wearing_necklace":   {"type": "int64"},
    "wearing_necktie":    {"type": "int64"},
    "young":              {"type": "int64"},
    "gaze_dir":           {"type": "sequence_float", "length": 2},
    "head_pose":          {"type": "sequence_float", "length": 3},
}

# H5 中需要跳过的字段 (非样本级数据)
OMNIFACE_SKIP_FIELDS = {"train_indices", "val_indices"}

# bytes 类型字段 (需要解码)
OMNIFACE_BYTES_FIELDS = {"id", "origin", "prompt"}


def convert_omniface(
    h5_path: str,
    output_dir: str,
    max_shard_size_mb: int = 500,
    workers: int = 16,
):
    """转换 OmniFace H5 -> HF parquet shards."""
    print(f"\n{'='*60}")
    print(f"开始转换 OmniFace: {h5_path}")
    print(f"{'='*60}\n")

    features = build_features(OMNIFACE_FIELDS)

    # 收集所有需要读取的标量/向量字段名 (排除 image 和 skip)
    label_fields = [k for k in OMNIFACE_FIELDS if k != "image"]

    with h5py.File(h5_path, "r") as f:
        N = f["images"].shape[0]
        print(f"  数据集大小: {N:,}")

        # 读取 train/val 划分
        train_idx = f["train_indices"][:]
        val_idx = f["val_indices"][:]
        print(f"  train: {len(train_idx):,}, val: {len(val_idx):,}")

        splits = {
            "train": train_idx,
            "val": val_idx,
        }

        splits_info = {}

        for split_name, indices in splits.items():
            print(f"\n  --- 处理 {split_name} split ({len(indices):,} 样本) ---")

            def batch_gen():
                batch_size = 8192  # 增大 batch, 减少 h5 访问次数
                for start in range(0, len(indices), batch_size):
                    end = min(start + batch_size, len(indices))
                    idx_batch = indices[start:end]

                    batch = {}

                    # images: 直接取 JPEG bytes
                    batch["image"] = [f["images"][i] for i in idx_batch]

                    # 其他字段
                    for field in label_fields:
                        if field in OMNIFACE_SKIP_FIELDS:
                            continue
                        vals = f[field][idx_batch]
                        if field in OMNIFACE_BYTES_FIELDS:
                            batch[field] = [bytes_to_str(v) for v in vals]
                        elif OMNIFACE_FIELDS[field]["type"].startswith("sequence_"):
                            # 向量字段: 转为 list
                            batch[field] = [v.tolist() for v in vals]
                        else:
                            # 标量字段
                            batch[field] = vals.tolist()

                    yield batch

            shard_paths = write_shards(
                batch_gen(),
                output_dir,
                split_name,
                features,
                max_shard_size_mb=max_shard_size_mb,
                total_samples=len(indices),
            )

            splits_info[split_name] = {
                "num_examples": len(indices),
                "num_shards": len(shard_paths),
                "shard_names": [os.path.basename(p) for p in shard_paths],
            }

    # 保存 dataset_infos.json
    save_dataset_infos(output_dir, "OmniFace", features, splits_info)
    print(f"\n  OmniFace 转换完成! 输出目录: {output_dir}")


# ============================================================================
# Phase 3: OmniShape 转换
# ============================================================================

OMNISHAPE_FIELDS = {
    "image":                    {"type": "image"},
    "model_id":                 {"type": "string"},
    "class":                    {"type": "int32"},
    "class100":                 {"type": "int32"},
    "anisotropy":               {"type": "float32"},
    "volume":                   {"type": "float32"},
    "volume_ratio":             {"type": "float32"},
    "hull_volume":              {"type": "float32"},
    "surface_area_ratio":       {"type": "float32"},
    "vert_count":               {"type": "int32"},
    "mat_count":                {"type": "int32"},
    "mat_slots":                {"type": "int32"},
    "mat_complexity":           {"type": "float32"},
    "2d_coverage":              {"type": "float32"},
    "2d_hw_ratio":              {"type": "float32"},
    "2d_rgb_complexity":        {"type": "float32"},
    "2d_silhouette_complexity": {"type": "float32"},
    "view_label":               {"type": "sequence_float", "length": 3},
    "xyz_size":                 {"type": "sequence_float", "length": 3},
}

OMNISHAPE_BYTES_FIELDS = {"model_id"}

# meta/ 子组字段配置
OMNISHAPE_META_FIELDS = {
    "model_id":                {"type": "string"},
    "model_name":              {"type": "string"},
    "class_id":                {"type": "string"},
    "class_name":              {"type": "string"},
    "class100_id":             {"type": "string"},
    "class100_name":           {"type": "string"},
    "model_anisotropy":        {"type": "float32"},
    "model_hull_volume":       {"type": "float32"},
    "model_mat_complexity":    {"type": "float32"},
    "model_mat_count":         {"type": "int32"},
    "model_mat_slots":         {"type": "int32"},
    "model_surface_area_ratio":{"type": "float32"},
    "model_vert_count":        {"type": "int32"},
    "model_volume":            {"type": "float32"},
    "model_volume_ratio":      {"type": "float32"},
    "model_xyz_size":          {"type": "sequence_float", "length": 3},
}


def _split_omnishape_indices(
    f: h5py.File,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    按 model_id 不重叠划分 train/val.
    同一 model_id 的所有视角必须在同一 split.
    """
    rng = np.random.RandomState(seed)
    all_model_ids = f["model_id"][:]

    # 获取唯一 model_id 及其首次出现位置 (用于排序)
    unique_ids, first_indices = np.unique(all_model_ids, return_index=True)
    # 按 model_id 字典序排序
    sort_order = np.argsort(first_indices)
    unique_ids = unique_ids[sort_order]

    # 打乱
    rng.shuffle(unique_ids)

    # 划分
    n_val = max(1, int(len(unique_ids) * val_ratio))
    val_model_ids = set(unique_ids[:n_val])
    train_model_ids = set(unique_ids[n_val:])

    # 构建索引 (向量化, 避免 1800 万次 set 查找)
    # 用字典映射 model_id -> 0/1 (train/val)
    model_to_split = {}
    for mid in val_model_ids:
        model_to_split[mid] = 1
    for mid in train_model_ids:
        model_to_split[mid] = 0

    split_flags = np.array([model_to_split[mid] for mid in all_model_ids], dtype=np.int8)
    train_indices = np.where(split_flags == 0)[0].astype(np.int32)
    val_indices = np.where(split_flags == 1)[0].astype(np.int32)

    return train_indices, val_indices


def convert_omnishape(
    h5_path: str,
    output_dir: str,
    max_shard_size_mb: int = 500,
    jpeg_quality: int = 95,
    val_ratio: float = 0.05,
    seed: int = 42,
    workers: int = 16,
):
    """转换 OmniShape H5 -> HF parquet shards."""
    print(f"\n{'='*60}")
    print(f"开始转换 OmniShape: {h5_path}")
    print(f"{'='*60}\n")

    features = build_features(OMNISHAPE_FIELDS)
    label_fields = [k for k in OMNISHAPE_FIELDS if k != "image"]

    with h5py.File(h5_path, "r") as f:
        N = f["images"].shape[0]
        print(f"  数据集大小: {N:,}")

        # ---- 3.2 train/val 划分 ----
        print(f"  按 model_id 划分 train/val (val_ratio={val_ratio}, seed={seed})...")
        train_idx, val_idx = _split_omnishape_indices(f, val_ratio, seed)
        print(f"  train: {len(train_idx):,}, val: {len(val_idx):,}")

        # 保存划分索引
        split_info = {
            "seed": seed,
            "val_ratio": val_ratio,
            "train_count": int(len(train_idx)),
            "val_count": int(len(val_idx)),
        }
        split_fpath = os.path.join(output_dir, "split_indices.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(split_fpath, "w") as sf:
            json.dump(split_info, sf, indent=2)
        print(f"  划分信息已保存: {split_fpath}")

        splits = {
            "train": train_idx,
            "val": val_idx,
        }

        splits_info = {}

        for split_name, indices in splits.items():
            print(f"\n  --- 处理 {split_name} split ({len(indices):,} 样本) ---")

            def batch_gen():
                batch_size = 10920  # 8x h5 chunk (1365*8), 充分利用大内存
                for start in range(0, len(indices), batch_size):
                    end = min(start + batch_size, len(indices))
                    idx_batch = indices[start:end]

                    batch = {}

                    # images: uint8 CHW -> JPEG bytes (多进程并行编码)
                    imgs_chw = f["images"][idx_batch]  # (B, 3, H, W)
                    batch["image"] = encode_jpegs_parallel(
                        imgs_chw, quality=jpeg_quality, workers=workers
                    )

                    # 其他字段
                    for field in label_fields:
                        vals = f[field][idx_batch]
                        if field in OMNISHAPE_BYTES_FIELDS:
                            batch[field] = [bytes_to_str(v) for v in vals]
                        elif OMNISHAPE_FIELDS[field]["type"].startswith("sequence_"):
                            batch[field] = [v.tolist() for v in vals]
                        else:
                            batch[field] = vals.tolist()

                    yield batch

            shard_paths = write_shards(
                batch_gen(),
                output_dir,
                split_name,
                features,
                max_shard_size_mb=max_shard_size_mb,
                total_samples=len(indices),
            )

            splits_info[split_name] = {
                "num_examples": len(indices),
                "num_shards": len(shard_paths),
                "shard_names": [os.path.basename(p) for p in shard_paths],
            }

        # ---- 3.5 保存 meta/ 子组 ----
        print(f"\n  --- 保存 meta/ 子组 ---")
        meta_features = build_features(OMNISHAPE_META_FIELDS)
        meta_batch = {}
        meta_bytes_fields = {"model_id", "class_id", "class100_id"}
        meta_object_fields = {"model_name", "class_name", "class100_name"}

        for field, cfg in OMNISHAPE_META_FIELDS.items():
            vals = f[f"meta/{field}"][:]
            if field in meta_bytes_fields:
                meta_batch[field] = [bytes_to_str(v) for v in vals]
            elif field in meta_object_fields:
                # h5py object dtype 读取后已经是 bytes 或 str
                meta_batch[field] = [bytes_to_str(v) if isinstance(v, bytes) else str(v) for v in vals]
            elif cfg["type"].startswith("sequence_"):
                meta_batch[field] = [v.tolist() for v in vals]
            else:
                meta_batch[field] = vals.tolist()

        meta_ds = Dataset.from_dict(meta_batch, features=meta_features)
        meta_fpath = os.path.join(output_dir, "meta.parquet")
        meta_ds.to_parquet(meta_fpath)
        print(f"  meta 已保存: {meta_fpath} ({len(meta_batch[next(iter(meta_batch))]):,} 条)")

    # 保存 dataset_infos.json
    save_dataset_infos(output_dir, "OmniShape", features, splits_info)
    print(f"\n  OmniShape 转换完成! 输出目录: {output_dir}")


# ============================================================================
# Phase 4: 验证
# ============================================================================

def verify_conversion(
    h5_path: str,
    output_dir: str,
    dataset_name: str,
):
    """验证转换后的 parquet 数据是否正确."""
    from datasets import load_dataset

    print(f"\n{'='*60}")
    print(f"验证 {dataset_name}: {output_dir}")
    print(f"{'='*60}\n")

    # 加载 parquet
    data_dir = os.path.join(output_dir, "data")
    parquet_files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")
    ])

    if not parquet_files:
        print("  [ERROR] 未找到 parquet 文件!")
        return

    # 按 split 分组
    from collections import defaultdict
    split_files = defaultdict(list)
    for f in parquet_files:
        fname = os.path.basename(f)
        # 提取 split 名称: train-00000-of-00139.parquet -> train
        split_name = fname.split("-")[0]
        split_files[split_name].append(f)

    for split_name, files in split_files.items():
        print(f"  --- {split_name} ({len(files)} 个分片) ---")
        ds = load_dataset("parquet", data_files=files, split="train")
        print(f"    样本数: {len(ds):,}")
        print(f"    字段: {list(ds.features.keys())}")
        print(f"    Features: {ds.features}")

        # 检查图片
        sample = ds[0]
        img = sample["image"]
        print(f"    图片类型: {type(img)}")
        if hasattr(img, "size"):
            print(f"    图片尺寸: {img.size}")
        print(f"    前 10 个字段值:")
        for j, k in enumerate(sample.keys()):
            if k == "image":
                continue
            v = sample[k]
            val_repr = repr(v)
            if len(val_repr) > 80:
                val_repr = val_repr[:80] + "..."
            print(f"      {k}: {val_repr}")
            if j >= 10:
                break

    # 与 h5 对比样本数
    print(f"\n  --- 与 H5 源文件对比 ---")
    with h5py.File(h5_path, "r") as f:
        h5_n = f["images"].shape[0]
        print(f"    H5 源样本数: {h5_n:,}")

    total_parquet = sum(len(load_dataset("parquet", data_files=files, split="train"))
                        for files in split_files.values())
    print(f"    Parquet 总样本数: {total_parquet:,}")
    if h5_n == total_parquet:
        print(f"    ✓ 样本数一致!")
    else:
        print(f"    ✗ 样本数不一致! 差异: {h5_n - total_parquet:,}")

    print(f"\n  验证完成!")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="H5 -> HuggingFace Datasets (Parquet) 转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=["omniface", "omnishape"],
        required=True,
        help="数据集名称",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="H5 文件路径",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=500,
        help="每个 parquet 分片的最大大小 (MB), 默认 500",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="OmniShape 图片 JPEG 编码质量 (仅 OmniShape 使用), 默认 95",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="OmniShape val 集比例 (仅 OmniShape 使用), 默认 0.05",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (仅 OmniShape 划分使用), 默认 42",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="验证模式: 读取转换后的 parquet 并与 h5 对比",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=16,
        help="并行进程数 (用于 JPEG 编码), 默认 16",
    )

    args = parser.parse_args()

    if args.verify:
        dataset_name = "OmniFace" if args.dataset == "omniface" else "OmniShape"
        verify_conversion(args.input, args.output, dataset_name)
        return

    if args.dataset == "omniface":
        convert_omniface(
            h5_path=args.input,
            output_dir=args.output,
            max_shard_size_mb=args.shard_size,
            workers=args.workers,
        )
    elif args.dataset == "omnishape":
        convert_omnishape(
            h5_path=args.input,
            output_dir=args.output,
            max_shard_size_mb=args.shard_size,
            jpeg_quality=args.jpeg_quality,
            val_ratio=args.val_ratio,
            seed=args.seed,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
