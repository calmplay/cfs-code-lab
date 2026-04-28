# -*- coding: utf-8 -*-
# @Time    : 2026/4/21
# @Author  : OpenAI
# @Comments: H5 -> HuggingFace Datasets (Parquet shards) 转换工具（优化版）

"""
H5 -> HuggingFace Datasets (Parquet 分片) 转换工具
===================================================

目标:
1. 图片列统一写成 HuggingFace Image 标准底层结构:
   {"bytes": <image_file_bytes>, "path": None}
2. OmniFace:
   - 支持原始 images 为 JPEG 字节流 (shape=(N,))
   - 支持原始 images 为 uint8 CHW 数组 (shape=(N, 3, H, W))
3. OmniShape:
   - uint8 CHW -> JPEG(quality=95 默认)
   - 进程池持久化, 只创建一次
   - 使用“按样本数近似控 shard”替代逐元素估大小
4. 其他样本列 / meta 列完整保留

示例:
  cd /home/cy/nuist-lab/cfs-code-lab/c00_utils

  # OmniFace
  python h5_to_hf_optimized.py -d omniface \
      -i /home/data/OmniFace_202602042244.h5 \
      -o /home/data/HF/OmniFace_o

  python h5_to_hf_optimized.py -d omniface \
      -i /home/data/OmniFace_64x64_20260421.h5 \
      -o /home/data/HF/OmniFace64_o

  # OmniShape
  python h5_to_hf_optimized.py -d omnishape \
      -i /home/data/OmniShape1k_18000a_128x128_20251204.h5 \
      -o /home/data/HF/OmniShape_o \
      -w 32

  # OmniShape较慢, 可以先快速验证
  python h5_to_hf_optimized.py -d omnishape \
      -i /home/data/OmniShape1k_18000a_128x128_20251204.h5 \
      -o /home/data/HF/OmniShape_test \
      -w 32 \
      --max-samples 10000

  # 验证
  python h5_to_hf_optimized.py -d omniface -i /home/data/OmniFace_64x64_20260421.h5 -o /home/data/HF/OmniFace64 --verify
  python h5_to_hf_optimized.py -d omnishape -i /home/data/OmniShape1k_18000a_128x128_20251204.h5 -o /home/data/HF/OmniShape --verify
"""

import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Sequence as TypingSequence, Tuple

import h5py
import numpy as np
from PIL import Image
from datasets import Dataset, Features, Image as HFImage, Sequence, Value
from tqdm import tqdm


# ============================================================================
# 通用工具
# ============================================================================

def default_workers() -> int:
    cpu = os.cpu_count() or 16
    return max(1, min(cpu - 2, 32))


def _encode_single_jpeg(args: Tuple[np.ndarray, int]) -> bytes:
    """worker: (HWC uint8, quality) -> jpeg bytes"""
    img_hwc, quality = args
    pil_img = Image.fromarray(img_hwc)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class JPEGEncoder:
    """可复用的 JPEG 编码器. workers<=1 时退化为串行."""

    def __init__(self, workers: int, quality: int):
        self.workers = max(1, int(workers))
        self.quality = int(quality)
        self.pool: Optional[Pool] = None
        if self.workers > 1:
            self.pool = Pool(processes=self.workers)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def encode_chw_batch(self, imgs_chw: np.ndarray) -> List[bytes]:
        """
        imgs_chw: (B, 3, H, W) uint8
        return: list[bytes]
        """
        if imgs_chw.ndim != 4 or imgs_chw.shape[1] != 3:
            raise ValueError(f"期望 CHW RGB 批量图像, 实际 shape={imgs_chw.shape}")

        imgs_hwc = np.transpose(imgs_chw, (0, 2, 3, 1)).copy()
        bsz = imgs_hwc.shape[0]

        if self.pool is None or bsz <= 64:
            return [_encode_single_jpeg((imgs_hwc[i], self.quality)) for i in range(bsz)]

        chunksize = max(1, bsz // (self.workers * 4))
        tasks = ((imgs_hwc[i], self.quality) for i in range(bsz))
        return list(self.pool.imap(_encode_single_jpeg, tasks, chunksize=chunksize))


class ManagedEncoder:
    """方便 with 使用."""

    def __init__(self, workers: int, quality: int):
        self.encoder = JPEGEncoder(workers=workers, quality=quality)

    def __enter__(self) -> JPEGEncoder:
        return self.encoder

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.encoder.close()
        return False


class ApproxShardPlanner:
    """
    基于一次小样本校准的“按样本数控 shard”规划器.
    不在每个 batch 上逐元素估大小.
    """

    def __init__(self, target_mb: int, safety_ratio: float = 0.92):
        self.target_bytes = int(target_mb * 1024 * 1024 * safety_ratio)
        self.examples_per_shard: Optional[int] = None

    def fit(self, calibration_batch: Dict[str, list], min_examples: int = 1024) -> int:
        n = len(next(iter(calibration_batch.values())))
        if n <= 0:
            raise ValueError("calibration_batch 为空")

        total = 0
        for key, values in calibration_batch.items():
            if not values:
                continue
            v0 = values[0]
            if key == "image" and isinstance(v0, dict) and "bytes" in v0:
                total += sum(len(v["bytes"]) for v in values)
            elif isinstance(v0, str):
                total += sum(len(v.encode("utf-8")) for v in values)
            elif isinstance(v0, (int, float, np.integer, np.floating)):
                total += 8 * len(values)
            elif isinstance(v0, list):
                # 小样本上做一次校准即可
                total += sum(np.asarray(v).nbytes for v in values)
            else:
                total += 16 * len(values)

        avg = max(1.0, total / n)
        est = max(min_examples, int(self.target_bytes / avg))
        self.examples_per_shard = est
        return est


class ShardWriter:
    """按样本数近似控 shard, 避免每个 batch 逐元素估大小."""

    def __init__(
            self,
            output_dir: str,
            split_name: str,
            features: Features,
            total_samples: int,
            examples_per_shard: int,
    ):
        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.split_name = split_name
        self.features = features
        self.total_samples = total_samples
        self.examples_per_shard = max(1, int(examples_per_shard))

        self.current_batch: Optional[Dict[str, list]] = None
        self.current_rows = 0
        self.shard_paths: List[str] = []
        self.shard_idx = 0
        self.processed = 0
        self.t0 = time.time()

        self.pbar = tqdm(
            total=total_samples,
            desc=f"  [{split_name}]",
            unit="img",
            unit_scale=True,
            ncols=120,
            file=sys.stdout,
        )

    def _ensure_batch(self, batch: Dict[str, list]):
        if self.current_batch is None:
            self.current_batch = {k: [] for k in batch}

    def add_batch(self, batch: Dict[str, list]):
        n = len(next(iter(batch.values())))
        self._ensure_batch(batch)
        assert self.current_batch is not None

        for k in self.current_batch:
            self.current_batch[k].extend(batch[k])
        self.current_rows += n
        self.processed += n
        self.pbar.update(n)

        if self.current_rows >= self.examples_per_shard:
            self.flush()

    def flush(self):
        if self.current_batch is None or self.current_rows == 0:
            return

        tmp_name = f"{self.split_name}-{self.shard_idx:05d}-of-00000.parquet"
        fpath = os.path.join(self.data_dir, tmp_name)
        ds = Dataset.from_dict(self.current_batch, features=self.features)
        ds.to_parquet(fpath)
        self.shard_paths.append(fpath)
        self.shard_idx += 1

        elapsed = time.time() - self.t0
        speed = self.processed / elapsed if elapsed > 0 else 0.0
        eta = (self.total_samples - self.processed) / speed if speed > 0 else 0.0
        self.pbar.set_postfix(
            shard=self.shard_idx,
            rows=self.current_rows,
            speed=f"{speed:.0f}/s",
            eta=f"{eta / 60:.1f}min",
        )

        self.current_batch = None
        self.current_rows = 0

    def close(self) -> List[str]:
        self.flush()
        self.pbar.close()

        total_shards = len(self.shard_paths)
        for i, old_path in enumerate(self.shard_paths):
            new_name = f"{self.split_name}-{i:05d}-of-{total_shards:05d}.parquet"
            new_path = os.path.join(self.data_dir, new_name)
            if old_path != new_path:
                os.rename(old_path, new_path)
                self.shard_paths[i] = new_path

        elapsed = time.time() - self.t0
        print(
            f"  [{self.split_name}] 完成! {self.processed:,} 样本 -> {total_shards} 个分片, "
            f"每片约 {self.examples_per_shard:,} 样本, 耗时 {elapsed / 60:.1f} min",
            flush=True,
        )
        return self.shard_paths


def bytes_to_str(val: Any) -> str:
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")
    return str(val)


def build_features(fields_config: Dict[str, dict]) -> Features:
    feat_dict = {}
    for name, cfg in fields_config.items():
        t = cfg["type"]
        if t == "image":
            feat_dict[name] = HFImage()
        elif t == "string":
            feat_dict[name] = Value("string")
        elif t == "int8":
            feat_dict[name] = Value("int32")
        elif t in ("int32", "int64"):
            feat_dict[name] = Value(t)
        elif t == "float32":
            feat_dict[name] = Value("float32")
        elif t == "sequence_float":
            length = cfg.get("length", -1)
            feat_dict[name] = Sequence(Value("float32"), length=length if length > 0 else -1)
        elif t == "sequence_int":
            length = cfg.get("length", -1)
            feat_dict[name] = Sequence(Value("int32"), length=length if length > 0 else -1)
        else:
            raise ValueError(f"不支持的字段类型: {t} (字段: {name})")
    return Features(feat_dict)


def save_dataset_infos(
        output_dir: str,
        dataset_name: str,
        features: Features,
        splits_info: Dict[str, dict],
):
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


def image_dicts_from_jpeg_bytes_list(jpeg_bytes_list: TypingSequence[bytes]) -> List[
    Dict[str, Optional[bytes]]]:
    return [{"bytes": bytes(b), "path": None} for b in jpeg_bytes_list]


def infer_image_storage_mode(h5_images_ds: h5py.Dataset) -> str:
    """
    返回:
      - 'jpeg_bytes' : shape=(N,), object/bytes, 每个元素是一张压缩图片字节流
      - 'chw_uint8'  : shape=(N,3,H,W), uint8 RGB
    """
    if h5_images_ds.ndim == 1:
        return "jpeg_bytes"
    if h5_images_ds.ndim == 4 and h5_images_ds.shape[1] == 3 and h5_images_ds.dtype == np.uint8:
        return "chw_uint8"
    raise ValueError(
        f"无法识别的 images 存储形式: dtype={h5_images_ds.dtype}, shape={h5_images_ds.shape}"
    )


# ============================================================================
# OmniFace
# ============================================================================

OMNIFACE_FIELDS = {
    "image": {"type": "image"},
    "id": {"type": "string"},
    "origin": {"type": "string"},
    "prompt": {"type": "string"},
    "age": {"type": "float32"},
    "arousal": {"type": "float32"},
    "valence": {"type": "float32"},
    "is_sr": {"type": "int8"},
    "expression": {"type": "int64"},
    "race": {"type": "int64"},
    "male": {"type": "int64"},
    "arched_eyebrows": {"type": "int64"},
    "attractive": {"type": "int64"},
    "bags_under_eyes": {"type": "int64"},
    "bald": {"type": "int64"},
    "bangs": {"type": "int64"},
    "big_lips": {"type": "int64"},
    "big_nose": {"type": "int64"},
    "black_hair": {"type": "int64"},
    "blond_hair": {"type": "int64"},
    "blurry": {"type": "int64"},
    "brown_hair": {"type": "int64"},
    "bushy_eyebrows": {"type": "int64"},
    "chubby": {"type": "int64"},
    "double_chin": {"type": "int64"},
    "eyeglasses": {"type": "int64"},
    "five_o_clock_shadow": {"type": "int64"},
    "goatee": {"type": "int64"},
    "gray_hair": {"type": "int64"},
    "heavy_makeup": {"type": "int64"},
    "high_cheekbones": {"type": "int64"},
    "mouth_slightly_open": {"type": "int64"},
    "mustache": {"type": "int64"},
    "narrow_eyes": {"type": "int64"},
    "no_beard": {"type": "int64"},
    "oval_face": {"type": "int64"},
    "pale_skin": {"type": "int64"},
    "pointy_nose": {"type": "int64"},
    "receding_hairline": {"type": "int64"},
    "rosy_cheeks": {"type": "int64"},
    "sideburns": {"type": "int64"},
    "smiling": {"type": "int64"},
    "straight_hair": {"type": "int64"},
    "wavy_hair": {"type": "int64"},
    "wearing_earrings": {"type": "int64"},
    "wearing_hat": {"type": "int64"},
    "wearing_lipstick": {"type": "int64"},
    "wearing_necklace": {"type": "int64"},
    "wearing_necktie": {"type": "int64"},
    "young": {"type": "int64"},
    "gaze_dir": {"type": "sequence_float", "length": 2},
    "head_pose": {"type": "sequence_float", "length": 3},
}

OMNIFACE_SKIP_FIELDS = {"train_indices", "val_indices"}
OMNIFACE_BYTES_FIELDS = {"id", "origin", "prompt"}


def build_omniface_batch(
        f: h5py.File,
        idx_batch: np.ndarray,
        label_fields: List[str],
        image_mode: str,
        encoder: Optional[JPEGEncoder],
) -> Dict[str, list]:
    batch: Dict[str, list] = {}

    if image_mode == "jpeg_bytes":
        jpeg_bytes = [bytes(f["images"][i]) for i in idx_batch]
    elif image_mode == "chw_uint8":
        if encoder is None:
            raise ValueError("OmniFace CHW 图像模式下 encoder 不能为空")
        imgs_chw = f["images"][idx_batch]
        jpeg_bytes = encoder.encode_chw_batch(imgs_chw)
    else:
        raise ValueError(f"未知 image_mode: {image_mode}")

    batch["image"] = image_dicts_from_jpeg_bytes_list(jpeg_bytes)

    for field in label_fields:
        if field in OMNIFACE_SKIP_FIELDS:
            continue
        vals = f[field][idx_batch]
        if field in OMNIFACE_BYTES_FIELDS:
            batch[field] = [bytes_to_str(v) for v in vals]
        elif OMNIFACE_FIELDS[field]["type"].startswith("sequence_"):
            batch[field] = [v.tolist() for v in vals]
        else:
            batch[field] = vals.tolist()

    return batch


def convert_omniface(
        h5_path: str,
        output_dir: str,
        max_shard_size_mb: int = 500,
        jpeg_quality: int = 95,
        workers: int = 1,
):
    print(f"\n{'=' * 60}")
    print(f"开始转换 OmniFace: {h5_path}")
    print(f"{'=' * 60}\n")

    os.makedirs(output_dir, exist_ok=True)
    features = build_features(OMNIFACE_FIELDS)
    label_fields = [k for k in OMNIFACE_FIELDS if k != "image"]

    with h5py.File(h5_path, "r") as f:
        images_ds = f["images"]
        image_mode = infer_image_storage_mode(images_ds)
        print(f"  images 存储模式: {image_mode}, shape={images_ds.shape}, dtype={images_ds.dtype}")

        n_total = images_ds.shape[0]
        print(f"  数据集大小: {n_total:,}")

        train_idx = f["train_indices"][:]
        val_idx = f["val_indices"][:]
        print(f"  train: {len(train_idx):,}, val: {len(val_idx):,}")

        splits = {"train": train_idx, "val": val_idx}
        splits_info = {}

        with ManagedEncoder(workers=workers if image_mode == "chw_uint8" else 1,
                            quality=jpeg_quality) as encoder:
            for split_name, indices in splits.items():
                print(f"\n  --- 处理 {split_name} split ({len(indices):,} 样本) ---")

                # 与 H5 chunk 对齐
                h5_chunk0 = images_ds.chunks[0] if images_ds.chunks else 1024
                batch_size = int(h5_chunk0 * 8)

                calib_n = min(batch_size, len(indices))
                calib_batch = build_omniface_batch(
                    f=f,
                    idx_batch=indices[:calib_n],
                    label_fields=label_fields,
                    image_mode=image_mode,
                    encoder=encoder if image_mode == "chw_uint8" else None,
                )
                planner = ApproxShardPlanner(target_mb=max_shard_size_mb)
                examples_per_shard = planner.fit(calib_batch, min_examples=max(1024, batch_size))
                print(f"  估计每 shard 约 {examples_per_shard:,} 样本")

                writer = ShardWriter(
                    output_dir=output_dir,
                    split_name=split_name,
                    features=features,
                    total_samples=len(indices),
                    examples_per_shard=examples_per_shard,
                )

                writer.add_batch(calib_batch)

                for start in range(calib_n, len(indices), batch_size):
                    end = min(start + batch_size, len(indices))
                    idx_batch = indices[start:end]
                    batch = build_omniface_batch(
                        f=f,
                        idx_batch=idx_batch,
                        label_fields=label_fields,
                        image_mode=image_mode,
                        encoder=encoder if image_mode == "chw_uint8" else None,
                    )
                    writer.add_batch(batch)

                shard_paths = writer.close()
                splits_info[split_name] = {
                    "num_examples": len(indices),
                    "num_shards": len(shard_paths),
                    "shard_names": [os.path.basename(p) for p in shard_paths],
                }

    save_dataset_infos(output_dir, "OmniFace", features, splits_info)
    print(f"\n  OmniFace 转换完成! 输出目录: {output_dir}")


# ============================================================================
# OmniShape
# ============================================================================

OMNISHAPE_FIELDS = {
    "image": {"type": "image"},
    "model_id": {"type": "string"},
    "class": {"type": "int32"},
    "class100": {"type": "int32"},
    "anisotropy": {"type": "float32"},
    "volume": {"type": "float32"},
    "volume_ratio": {"type": "float32"},
    "hull_volume": {"type": "float32"},
    "surface_area_ratio": {"type": "float32"},
    "vert_count": {"type": "int32"},
    "mat_count": {"type": "int32"},
    "mat_slots": {"type": "int32"},
    "mat_complexity": {"type": "float32"},
    "2d_coverage": {"type": "float32"},
    "2d_hw_ratio": {"type": "float32"},
    "2d_rgb_complexity": {"type": "float32"},
    "2d_silhouette_complexity": {"type": "float32"},
    "view_label": {"type": "sequence_float", "length": 3},
    "xyz_size": {"type": "sequence_float", "length": 3},
}

OMNISHAPE_BYTES_FIELDS = {"model_id"}

OMNISHAPE_META_FIELDS = {
    "model_id": {"type": "string"},
    "model_name": {"type": "string"},
    "class_id": {"type": "string"},
    "class_name": {"type": "string"},
    "class100_id": {"type": "string"},
    "class100_name": {"type": "string"},
    "model_anisotropy": {"type": "float32"},
    "model_hull_volume": {"type": "float32"},
    "model_mat_complexity": {"type": "float32"},
    "model_mat_count": {"type": "int32"},
    "model_mat_slots": {"type": "int32"},
    "model_surface_area_ratio": {"type": "float32"},
    "model_vert_count": {"type": "int32"},
    "model_volume": {"type": "float32"},
    "model_volume_ratio": {"type": "float32"},
    "model_xyz_size": {"type": "sequence_float", "length": 3},
}


def _split_omnishape_indices(
        f: h5py.File,
        val_ratio: float = 0.05,
        seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """按 model_id 不重叠划分 train/val."""
    rng = np.random.RandomState(seed)
    all_model_ids = f["model_id"][:]

    unique_ids, first_indices = np.unique(all_model_ids, return_index=True)
    unique_ids = unique_ids[np.argsort(first_indices)]
    rng.shuffle(unique_ids)

    n_val = max(1, int(len(unique_ids) * val_ratio))
    val_model_ids = unique_ids[:n_val]

    # C 侧向量化 membership, 比 Python 字典/循环更快
    is_val = np.isin(all_model_ids, val_model_ids)
    val_indices = np.where(is_val)[0].astype(np.int32)
    train_indices = np.where(~is_val)[0].astype(np.int32)
    return train_indices, val_indices


def build_omnishape_batch(
        f: h5py.File,
        idx_batch: np.ndarray,
        label_fields: List[str],
        encoder: JPEGEncoder,
) -> Dict[str, list]:
    batch: Dict[str, list] = {}

    imgs_chw = f["images"][idx_batch]
    jpeg_bytes = encoder.encode_chw_batch(imgs_chw)
    batch["image"] = image_dicts_from_jpeg_bytes_list(jpeg_bytes)

    for field in label_fields:
        vals = f[field][idx_batch]
        if field in OMNISHAPE_BYTES_FIELDS:
            batch[field] = [bytes_to_str(v) for v in vals]
        elif OMNISHAPE_FIELDS[field]["type"].startswith("sequence_"):
            batch[field] = [v.tolist() for v in vals]
        else:
            batch[field] = vals.tolist()

    return batch


def save_omnishape_meta(f: h5py.File, output_dir: str):
    print(f"\n  --- 保存 meta/ 子组 ---")
    meta_features = build_features(OMNISHAPE_META_FIELDS)
    meta_batch: Dict[str, list] = {}
    meta_bytes_fields = {"model_id", "class_id", "class100_id"}
    meta_object_fields = {"model_name", "class_name", "class100_name"}

    # 读取所有字段并检查长度一致性
    field_lengths = {}
    for field, cfg in OMNISHAPE_META_FIELDS.items():
        vals = f[f"meta/{field}"][:]
        field_lengths[field] = len(vals)
        
    # 检查所有字段长度是否一致
    lengths = list(field_lengths.values())
    if len(set(lengths)) > 1:
        print(f"  [警告] meta 字段长度不一致: {field_lengths}")
        # 使用最小长度作为基准
        min_length = min(lengths)
        print(f"  [警告] 使用最小长度 {min_length} 作为基准")
    else:
        min_length = lengths[0]

    # 重新读取并截断到最小长度
    for field, cfg in OMNISHAPE_META_FIELDS.items():
        vals = f[f"meta/{field}"][:min_length]
        if field in meta_bytes_fields:
            meta_batch[field] = [bytes_to_str(v) for v in vals]
        elif field in meta_object_fields:
            meta_batch[field] = [bytes_to_str(v) if isinstance(v, (bytes, np.bytes_)) else str(v)
                                 for v in vals]
        elif cfg["type"].startswith("sequence_"):
            meta_batch[field] = [v.tolist() for v in vals]
        else:
            meta_batch[field] = vals.tolist()

    meta_ds = Dataset.from_dict(meta_batch, features=meta_features)
    meta_fpath = os.path.join(output_dir, "meta.parquet")
    meta_ds.to_parquet(meta_fpath)
    print(f"  meta 已保存: {meta_fpath} ({len(next(iter(meta_batch.values()))):,} 条)")


def convert_omnishape(
        h5_path: str,
        output_dir: str,
        max_shard_size_mb: int = 500,
        jpeg_quality: int = 95,
        val_ratio: float = 0.05,
        seed: int = 42,
        workers: int = 16,
        max_samples: int = None,
):
    print(f"\n{'=' * 60}")
    print(f"开始转换 OmniShape: {h5_path}")
    print(f"{'=' * 60}\n")

    os.makedirs(output_dir, exist_ok=True)
    features = build_features(OMNISHAPE_FIELDS)
    label_fields = [k for k in OMNISHAPE_FIELDS if k != "image"]

    with h5py.File(h5_path, "r") as f:
        images_ds = f["images"]
        if images_ds.ndim != 4 or images_ds.shape[1] != 3 or images_ds.dtype != np.uint8:
            raise ValueError(
                f"OmniShape images 期望为 uint8 CHW, 实际 shape={images_ds.shape}, dtype={images_ds.dtype}")

        n_total = images_ds.shape[0]
        print(f"  数据集大小: {n_total:,}")
        print(
            f"  images shape={images_ds.shape}, chunks={images_ds.chunks}, dtype={images_ds.dtype}")
        print(f"  JPEG quality={jpeg_quality}, workers={workers}")

        print(f"  按 model_id 划分 train/val (val_ratio={val_ratio}, seed={seed})...")
        train_idx, val_idx = _split_omnishape_indices(f, val_ratio, seed)
        
        # 限制最大样本数
        if max_samples is not None:
            print(f"  [限制样本数模式] 只处理前 {max_samples:,} 个样本")
            train_idx = train_idx[:max_samples]
            val_idx = val_idx[:max_samples]
            print(f"  限制后 - train: {len(train_idx):,}, val: {len(val_idx):,}")
        else:
            print(f"  train: {len(train_idx):,}, val: {len(val_idx):,}")

        split_info = {
            "seed": seed,
            "val_ratio": val_ratio,
            "train_count": int(len(train_idx)),
            "val_count": int(len(val_idx)),
            "max_samples": max_samples,
        }
        split_fpath = os.path.join(output_dir, "split_indices.json")
        with open(split_fpath, "w", encoding="utf-8") as sf:
            json.dump(split_info, sf, indent=2)
        print(f"  划分信息已保存: {split_fpath}")

        splits = {"train": train_idx, "val": val_idx}
        splits_info = {}

        h5_chunk0 = images_ds.chunks[0] if images_ds.chunks else 1365
        batch_size = int(h5_chunk0 * 8)

        with ManagedEncoder(workers=workers, quality=jpeg_quality) as encoder:
            for split_name, indices in splits.items():
                print(f"\n  --- 处理 {split_name} split ({len(indices):,} 样本) ---")

                calib_n = min(batch_size, len(indices))
                calib_batch = build_omnishape_batch(
                    f=f,
                    idx_batch=indices[:calib_n],
                    label_fields=label_fields,
                    encoder=encoder,
                )
                planner = ApproxShardPlanner(target_mb=max_shard_size_mb)
                examples_per_shard = planner.fit(calib_batch, min_examples=max(batch_size, 2048))
                print(f"  估计每 shard 约 {examples_per_shard:,} 样本")

                writer = ShardWriter(
                    output_dir=output_dir,
                    split_name=split_name,
                    features=features,
                    total_samples=len(indices),
                    examples_per_shard=examples_per_shard,
                )

                writer.add_batch(calib_batch)

                for start in range(calib_n, len(indices), batch_size):
                    end = min(start + batch_size, len(indices))
                    idx_batch = indices[start:end]
                    batch = build_omnishape_batch(
                        f=f,
                        idx_batch=idx_batch,
                        label_fields=label_fields,
                        encoder=encoder,
                    )
                    writer.add_batch(batch)

                shard_paths = writer.close()
                splits_info[split_name] = {
                    "num_examples": len(indices),
                    "num_shards": len(shard_paths),
                    "shard_names": [os.path.basename(p) for p in shard_paths],
                }

        save_omnishape_meta(f, output_dir)

    save_dataset_infos(output_dir, "OmniShape", features, splits_info)
    print(f"\n  OmniShape 转换完成! 输出目录: {output_dir}")


# ============================================================================
# 验证
# ============================================================================

def verify_conversion(
        h5_path: str,
        output_dir: str,
        dataset_name: str,
):
    from datasets import load_dataset

    print(f"\n{'=' * 60}")
    print(f"验证 {dataset_name}: {output_dir}")
    print(f"{'=' * 60}\n")

    data_dir = os.path.join(output_dir, "data")
    parquet_files = sorted(
        [os.path.join(data_dir, x) for x in os.listdir(data_dir) if x.endswith(".parquet")]
    )
    if not parquet_files:
        print("  [ERROR] 未找到 parquet 文件!")
        return

    split_files = defaultdict(list)
    for fpath in parquet_files:
        split_name = os.path.basename(fpath).split("-")[0]
        split_files[split_name].append(fpath)

    total_parquet = 0
    for split_name, files in split_files.items():
        print(f"  --- {split_name} ({len(files)} 个分片) ---")
        ds = load_dataset("parquet", data_files=files, split="train")
        total_parquet += len(ds)
        print(f"    样本数: {len(ds):,}")
        print(f"    字段: {list(ds.features.keys())}")
        print(f"    Features: {ds.features}")

        sample = ds[0]
        img = sample["image"]
        print(f"    图片类型: {type(img)}")
        if hasattr(img, "size"):
            print(f"    图片尺寸: {img.size}")
        for j, k in enumerate(sample.keys()):
            if k == "image":
                continue
            v = sample[k]
            vr = repr(v)
            if len(vr) > 80:
                vr = vr[:80] + "..."
            print(f"      {k}: {vr}")
            if j >= 10:
                break

    print(f"\n  --- 与 H5 源文件对比 ---")
    with h5py.File(h5_path, "r") as f:
        h5_n = int(f["images"].shape[0])
        print(f"    H5 源样本数: {h5_n:,}")
    print(f"    Parquet 总样本数: {total_parquet:,}")
    if h5_n == total_parquet:
        print("    ✓ 样本数一致!")
    else:
        print(f"    ✗ 样本数不一致! 差异: {h5_n - total_parquet:,}")

    meta_path = os.path.join(output_dir, "meta.parquet")
    if os.path.exists(meta_path):
        print(f"    发现 meta: {meta_path}")

    print("\n  验证完成!")


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
    parser.add_argument("--input", "-i", required=True, help="H5 文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument(
        "--shard-size",
        type=int,
        default=500,
        help="目标 shard 大小 (MB, 近似控制), 默认 500",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG 编码质量 (OmniShape 必用, OmniFace CHW 模式下也会使用), 默认 95",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="OmniShape val 集比例, 默认 0.05",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (OmniShape 划分使用), 默认 42",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="验证模式: 读取转换后的 parquet 并与 h5 对比",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=default_workers(),
        help=f"JPEG 编码并行进程数, 默认 {default_workers()}",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="限制处理的最大样本数，例如100000",
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
            jpeg_quality=args.jpeg_quality,
            workers=args.workers,
        )
    else:
        convert_omnishape(
            h5_path=args.input,
            output_dir=args.output,
            max_shard_size_mb=args.shard_size,
            jpeg_quality=args.jpeg_quality,
            val_ratio=args.val_ratio,
            seed=args.seed,
            workers=args.workers,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
