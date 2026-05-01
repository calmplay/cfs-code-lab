# -*- coding: utf-8 -*-
# @Time    : 2026/4/30
# @Author  : OpenAI
# @Comments: H5 -> HuggingFace Datasets (Parquet shards) 转换工具（重构版，只输出 all split）

"""
H5 -> HuggingFace Datasets (Parquet 分片) 转换工具
===================================================

核心特性:
1. 不划分 train/val/test，只输出 all split
2. 严格顺序读取 H5，Parquet 样本顺序与 H5 原始顺序一致
3. 图片列统一写成 HuggingFace Image 标准底层结构:
   {"bytes": <image_file_bytes>, "path": None}
4. OmniFace:
   - 支持原始 images 为 JPEG 字节流 (shape=(N,))
   - 支持原始 images 为 uint8 CHW 数组 (shape=(N, 3, H, W))
5. OmniShape:
   - uint8 CHW -> JPEG(quality=95 默认)
   - 进程池持久化, 只创建一次
   - meta 数据自动遍历 H5 /meta group 并完整保存

示例:
cd /home/cy/nuist-lab/cfs-code-lab/c00_utils

# OmniFace 转换
python h5_to_hf.py -d omniface \
  -i /home/data/OmniFace64-V1_20260421.h5 \
  -o /home/data/HF/OmniFace64-V1_20260430 \
  -w 12

python h5_to_hf.py -d omniface \
  -i /mhd/home/data/OmniFace_202602042244.h5 \
  -o /home/data/HF/OmniFace512-V1_20260430 \
  -w 12

# OmniShape 转换
python h5_to_hf.py -d omnishape \
  -i /home/data/OmniShape64-V1_20260421.h5 \
  -o /home/data/HF/OmniShape64-V1_20260430 \
  -w 24

python h5_to_hf.py -d omnishape \
  -i /home/data/OmniShape1k_18000a_128x128_20251204.h5 \
  -o /home/data/HF/OmniShape128-V1_20260430 \
  -w 24

# 快速验证（只处理前 10000 个样本）
python h5_to_hf.py -d omnishape \
  -i /home/data/OmniShape1k_18000a_128x128_20251204.h5 \
  -o /home/data/HF/OmniShape128-V1_test \
  -w 16 \
  --max-samples 12800

# 验证
python h5_to_hf.py -d omniface -i /home/data/OmniFace64-V1_20260421.h5 -o /home/data/HF/OmniFace64 --verify
python h5_to_hf.py -d omnishape -i /home/data/OmniShape1k_18000a_128x128_20251204.h5 -o /home/data/HF/OmniShape --verify
"""

import argparse
import io
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Sequence as TypingSequence, Tuple, Union

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
    基于一次小样本校准的"按样本数数控 shard"规划器.
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


def image_dicts_from_jpeg_bytes_list(jpeg_bytes_list: TypingSequence[bytes]) -> List[
    Dict[str, Optional[bytes]]]:
    return [{"bytes": bytes(b), "path": None} for b in jpeg_bytes_list]


def infer_image_storage_mode(h5_images_ds: h5py.Dataset) -> str:
    """
    返回:
      - 'jpeg_bytes': shape=(N,), object/bytes, 每个元素是一张压缩图片字节流
      - 'chw_uint8': shape=(N,3,H,W), uint8 RGB
    """
    if h5_images_ds.ndim == 1:
        return "jpeg_bytes"
    if h5_images_ds.ndim == 4 and h5_images_ds.shape[1] == 3 and h5_images_ds.dtype == np.uint8:
        return "chw_uint8"
    raise ValueError(
        f"无法识别的 images 存储形式: dtype={h5_images_ds.dtype}, shape={h5_images_ds.shape}"
    )


def prepare_output_dir(output_dir: str, overwrite: bool = False):
    """准备输出目录，如果已存在则报错或清空"""
    if os.path.exists(output_dir):
        if not overwrite:
            data_dir = os.path.join(output_dir, "data")
            if os.path.exists(data_dir):
                parquet_files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
                if parquet_files:
                    raise FileExistsError(
                        f"输出目录已存在且包含 parquet 文件: {output_dir}\n"
                        f"请使用 --overwrite 参数覆盖，或指定新的输出目录。"
                    )
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)


def save_dataset_infos(
        output_dir: str,
        dataset_name: str,
        features: Features,
        num_examples: int,
        shard_paths: List[str],
):
    """保存 dataset_infos.json，只包含 all split"""
    infos = {
        dataset_name: {
            "features": features.to_dict(),
            "splits": {
                "all": {
                    "num_examples": num_examples,
                    "num_shards": len(shard_paths),
                    "shard_names": [os.path.basename(p) for p in shard_paths],
                }
            },
        }
    }
    fpath = os.path.join(output_dir, "dataset_infos.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=2, default=str)
    print(f"  已保存 {fpath}")


def save_indices_json(
        output_dir: str,
        dataset_name: str,
        num_examples: int,
        max_samples: Optional[int] = None,
):
    """保存 indices.json，记录样本顺序信息"""
    indices_info = {
        "dataset_name": dataset_name,
        "order": "source_index_ascending",
        "num_examples": num_examples,
        "max_samples": max_samples,
        "source_index_start": 0,
        "source_index_end_exclusive": num_examples,
    }
    fpath = os.path.join(output_dir, "indices.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(indices_info, f, indent=2)
    print(f"  索引信息已保存: {fpath}")


# ============================================================================
# OmniFace
# ============================================================================

OMNIFACE_FIELDS = {
    "image": {"type": "image"},
    "source_index": {"type": "int64"},
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

OMNIFACE_SKIP_FIELDS = {"train_indices", "val_indices", "test_indices"}
OMNIFACE_BYTES_FIELDS = {"id", "origin", "prompt"}


def check_omniface_fields(f: h5py.File):
    """检查 OmniFace H5 文件中是否存在所有必需字段"""
    missing_fields = []

    if "images" not in f:
        missing_fields.append("images")

    for field in OMNIFACE_FIELDS.keys():
        # image 是输出列，由 H5 的 images 转换得到；source_index 是转换时生成的
        if field in {"image", "source_index"}:
            continue
        if field in OMNIFACE_SKIP_FIELDS:
            continue
        if field not in f:
            missing_fields.append(field)

    if missing_fields:
        raise ValueError(f"OmniFace H5 文件缺少以下字段: {missing_fields}")


def build_omniface_batch_from_slice(
        f: h5py.File,
        start: int,
        end: int,
        label_fields: List[str],
        image_mode: str,
        encoder: Optional[JPEGEncoder],
) -> Dict[str, list]:
    """
    从连续 slice 构造 OmniFace batch。
    使用连续 slice 读取 H5，保证 gzip chunk 顺序读取性能。
    """
    batch: Dict[str, list] = {}

    if image_mode == "jpeg_bytes":
        img_vals = f["images"][start:end]
        jpeg_bytes = [bytes(v) for v in img_vals]
    elif image_mode == "chw_uint8":
        if encoder is None:
            raise ValueError("OmniFace CHW 图像模式下 encoder 不能为空")
        imgs_chw = f["images"][start:end]
        jpeg_bytes = encoder.encode_chw_batch(imgs_chw)
    else:
        raise ValueError(f"未知 image_mode: {image_mode}")

    batch["image"] = image_dicts_from_jpeg_bytes_list(jpeg_bytes)

    batch["source_index"] = list(range(start, end))

    for field in label_fields:
        if field in OMNIFACE_SKIP_FIELDS:
            continue
        if field == "source_index":
            continue
        if field not in f:
            continue
        vals = f[field][start:end]
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
        max_samples: Optional[int] = None,
        overwrite: bool = False,
):
    print(f"\n{'=' * 60}")
    print(f"开始转换 OmniFace: {h5_path}")
    print(f"{'=' * 60}\n")

    prepare_output_dir(output_dir, overwrite=overwrite)
    features = build_features(OMNIFACE_FIELDS)
    label_fields = [k for k in OMNIFACE_FIELDS if k != "image"]

    with h5py.File(h5_path, "r") as f:
        check_omniface_fields(f)

        images_ds = f["images"]
        image_mode = infer_image_storage_mode(images_ds)
        print(f"  images 存储模式: {image_mode}, shape={images_ds.shape}, dtype={images_ds.dtype}")

        n_total = images_ds.shape[0]
        print(f"  数据集大小: {n_total:,}")

        n_process = n_total if max_samples is None else min(max_samples, n_total)
        if max_samples is not None:
            print(f"  [限制样本数模式] 只处理前 {n_process:,} 个样本")

        print(f"  使用 H5 原始顺序重打包，不排序、不 shuffle")

        h5_chunk0 = images_ds.chunks[0] if images_ds.chunks else 1024
        batch_size = int(h5_chunk0 * 8)
        print(f"  batch_size={batch_size} (H5 chunk={h5_chunk0})")

        with ManagedEncoder(workers=workers if image_mode == "chw_uint8" else 1,
                            quality=jpeg_quality) as encoder:
            print(f"\n  --- 处理 all split ({n_process:,} 样本) ---")

            calib_n = min(batch_size, n_process)
            calib_batch = build_omniface_batch_from_slice(
                f=f,
                start=0,
                end=calib_n,
                label_fields=label_fields,
                image_mode=image_mode,
                encoder=encoder if image_mode == "chw_uint8" else None,
            )
            planner = ApproxShardPlanner(target_mb=max_shard_size_mb)
            examples_per_shard = planner.fit(calib_batch, min_examples=max(1024, batch_size))
            print(f"  估计每 shard 约 {examples_per_shard:,} 样本")

            writer = ShardWriter(
                output_dir=output_dir,
                split_name="all",
                features=features,
                total_samples=n_process,
                examples_per_shard=examples_per_shard,
            )

            writer.add_batch(calib_batch)

            for start in range(calib_n, n_process, batch_size):
                end = min(start + batch_size, n_process)
                batch = build_omniface_batch_from_slice(
                    f=f,
                    start=start,
                    end=end,
                    label_fields=label_fields,
                    image_mode=image_mode,
                    encoder=encoder if image_mode == "chw_uint8" else None,
                )
                writer.add_batch(batch)

            shard_paths = writer.close()

    save_indices_json(output_dir, "OmniFace", n_process, max_samples)
    save_dataset_infos(output_dir, "OmniFace", features, n_process, shard_paths)
    print(f"\n  OmniFace 转换完成! 输出目录: {output_dir}")


# ============================================================================
# OmniShape
# ============================================================================

OMNISHAPE_FIELDS = {
    "image": {"type": "image"},
    "source_index": {"type": "int64"},
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


def check_omnishape_fields(f: h5py.File):
    """检查 OmniShape H5 文件中是否存在所有必需字段"""
    missing_fields = []

    if "images" not in f:
        missing_fields.append("images")

    for field in OMNISHAPE_FIELDS.keys():
        # image 是输出列，由 H5 的 images 转换得到；source_index 是转换时生成的
        if field in {"image", "source_index"}:
            continue
        if field not in f:
            missing_fields.append(field)

    if missing_fields:
        raise ValueError(f"OmniShape H5 文件缺少以下字段: {missing_fields}")


def build_omnishape_batch_from_slice(
        f: h5py.File,
        start: int,
        end: int,
        label_fields: List[str],
        encoder: JPEGEncoder,
) -> Dict[str, list]:
    """
    从连续 slice 构造 OmniShape batch。
    使用连续 slice 读取 H5，保证 gzip chunk 顺序读取性能。
    """
    batch: Dict[str, list] = {}

    imgs_chw = f["images"][start:end]
    jpeg_bytes = encoder.encode_chw_batch(imgs_chw)
    batch["image"] = image_dicts_from_jpeg_bytes_list(jpeg_bytes)

    batch["source_index"] = list(range(start, end))

    for field in label_fields:
        if field == "source_index":
            continue
        if field not in f:
            continue
        vals = f[field][start:end]
        if field in OMNISHAPE_BYTES_FIELDS:
            batch[field] = [bytes_to_str(v) for v in vals]
        elif OMNISHAPE_FIELDS[field]["type"].startswith("sequence_"):
            batch[field] = [v.tolist() for v in vals]
        else:
            batch[field] = vals.tolist()

    return batch


def save_omnishape_meta(f: h5py.File, output_dir: str):
    """
    自动遍历 H5 中 /meta group 的所有字段并保存。
    如果字段长度一致，保存为 meta.parquet。
    否则，按长度分组保存为多个 parquet 文件。
    """
    print(f"\n  --- 保存 meta 子组 ---")

    if "meta" not in f:
        print("  [WARNING] H5 文件中不存在 /meta group，跳过 meta 保存")
        return

    meta_group = f["meta"]
    meta_fields = list(meta_group.keys())

    if not meta_fields:
        print("  [WARNING] /meta group 为空，跳过 meta 保存")
        return

    print(f"  发现 meta 字段: {meta_fields}")

    field_info = {}
    for field in meta_fields:
        ds = meta_group[field]
        shape = ds.shape
        dtype = str(ds.dtype)
        field_info[field] = {
            "shape": list(shape),
            "dtype": dtype,
            "length": shape[0] if len(shape) > 0 else 0,
        }

    length_groups = {}
    for field, info in field_info.items():
        length = info["length"]
        if length not in length_groups:
            length_groups[length] = []
        length_groups[length].append(field)

    print(f"  字段长度分组: {dict((k, len(v)) for k, v in length_groups.items())}")

    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    saved_files = []

    for length, fields in length_groups.items():
        if length == 0:
            print(f"  [WARNING] 跳过长度为 0 的字段: {fields}")
            continue

        group_name = f"meta_l{length}"
        batch = {}

        for field in fields:
            ds = meta_group[field]
            vals = ds[:]

            if ds.dtype.kind == 'S' or ds.dtype.kind == 'O':
                if len(vals.shape) == 1:
                    batch[field] = [bytes_to_str(v) for v in vals]
                else:
                    batch[field] = [bytes_to_str(v) if isinstance(v, (bytes, np.bytes_)) else str(v)
                                    for v in vals]
            elif len(vals.shape) > 1:
                batch[field] = [v.tolist() for v in vals]
            else:
                batch[field] = vals.tolist()

        meta_ds = Dataset.from_dict(batch)
        meta_fpath = os.path.join(meta_dir, f"{group_name}.parquet")
        meta_ds.to_parquet(meta_fpath)
        saved_files.append(meta_fpath)
        print(f"  已保存: {meta_fpath} ({length:,} 条, {len(fields)} 个字段)")

    meta_schema = {
        "fields": field_info,
        "length_groups": {str(k): v for k, v in length_groups.items()},
    }
    schema_fpath = os.path.join(meta_dir, "meta_schema.json")
    with open(schema_fpath, "w", encoding="utf-8") as sf:
        json.dump(meta_schema, sf, indent=2)
    print(f"  已保存 meta_schema.json")


def load_omnishape_meta(output_dir: str, as_pandas: bool = False) -> Union[
    Dataset, Dict[str, "pd.DataFrame"]]:
    """
    读取 OmniShape meta 数据。

    Args:
        output_dir: 输出目录
        as_pandas: 是否返回 pandas DataFrame，默认返回 datasets.Dataset

    Returns:
        如果 as_pandas=False，返回 dict[str, Dataset]
        如果 as_pandas=True，返回 dict[str, pd.DataFrame]
    """
    meta_dir = os.path.join(output_dir, "meta")

    if not os.path.exists(meta_dir):
        raise FileNotFoundError(f"meta 目录不存在: {meta_dir}")

    from datasets import load_dataset

    result = {}
    for fname in os.listdir(meta_dir):
        if fname.endswith(".parquet") and fname.startswith("meta_"):
            fpath = os.path.join(meta_dir, fname)
            ds = load_dataset("parquet", data_files=fpath, split="train")
            key = fname.replace(".parquet", "")
            if as_pandas:
                result[key] = ds.to_pandas()
            else:
                result[key] = ds

    return result


def convert_omnishape(
        h5_path: str,
        output_dir: str,
        max_shard_size_mb: int = 500,
        jpeg_quality: int = 95,
        workers: int = 16,
        max_samples: Optional[int] = None,
        overwrite: bool = False,
):
    print(f"\n{'=' * 60}")
    print(f"开始转换 OmniShape: {h5_path}")
    print(f"{'=' * 60}\n")

    prepare_output_dir(output_dir, overwrite=overwrite)
    features = build_features(OMNISHAPE_FIELDS)
    label_fields = [k for k in OMNISHAPE_FIELDS if k != "image"]

    with h5py.File(h5_path, "r") as f:
        check_omnishape_fields(f)

        images_ds = f["images"]
        if images_ds.ndim != 4 or images_ds.shape[1] != 3 or images_ds.dtype != np.uint8:
            raise ValueError(
                f"OmniShape images 期望 uint8 CHW, 实际 shape={images_ds.shape}, dtype={images_ds.dtype}"
            )

        n_total = images_ds.shape[0]
        print(f"  数据集大小: {n_total:,}")
        print(
            f"  images shape={images_ds.shape}, chunks={images_ds.chunks}, dtype={images_ds.dtype}")
        print(f"  JPEG quality={jpeg_quality}, workers={workers}")

        n_process = n_total if max_samples is None else min(max_samples, n_total)
        if max_samples is not None:
            print(f"  [限制样本数模式] 只处理前 {n_process:,} 个样本")

        print(f"  使用 H5 source_index 原始顺序重打包，不排序、不 shuffle")

        h5_chunk0 = images_ds.chunks[0] if images_ds.chunks else 1365
        batch_size = int(h5_chunk0 * 8)
        print(f"  batch_size={batch_size} (H5 chunk={h5_chunk0})")

        with ManagedEncoder(workers=workers, quality=jpeg_quality) as encoder:
            print(f"\n  --- 处理 all split ({n_process:,} 样本) ---")

            calib_n = min(batch_size, n_process)
            calib_batch = build_omnishape_batch_from_slice(
                f=f,
                start=0,
                end=calib_n,
                label_fields=label_fields,
                encoder=encoder,
            )
            planner = ApproxShardPlanner(target_mb=max_shard_size_mb)
            examples_per_shard = planner.fit(calib_batch, min_examples=max(batch_size, 2048))
            print(f"  估计每 shard 约 {examples_per_shard:,} 样本")

            writer = ShardWriter(
                output_dir=output_dir,
                split_name="all",
                features=features,
                total_samples=n_process,
                examples_per_shard=examples_per_shard,
            )

            writer.add_batch(calib_batch)

            for start in range(calib_n, n_process, batch_size):
                end = min(start + batch_size, n_process)
                batch = build_omnishape_batch_from_slice(
                    f=f,
                    start=start,
                    end=end,
                    label_fields=label_fields,
                    encoder=encoder,
                )
                writer.add_batch(batch)

            shard_paths = writer.close()

        save_omnishape_meta(f, output_dir)

    save_indices_json(output_dir, "OmniShape", n_process, max_samples)
    save_dataset_infos(output_dir, "OmniShape", features, n_process, shard_paths)
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

    if not os.path.exists(data_dir):
        print(f"  [ERROR] data 目录不存在: {data_dir}")
        return

    old_parquet_files = [
        x for x in os.listdir(data_dir)
        if (x.startswith("train-") or x.startswith("val-") or x.startswith("test-")) and x.endswith(
            ".parquet")
    ]
    if old_parquet_files:
        print(f"  [ERROR] 发现旧的 train/val/test parquet 文件，请重新转换！")
        print(f"  旧文件: {old_parquet_files[:5]}")
        return

    parquet_files = sorted([
        os.path.join(data_dir, x)
        for x in os.listdir(data_dir)
        if x.startswith("all-") and x.endswith(".parquet")
    ])
    if not parquet_files:
        print("  [ERROR] 未找到 all-*.parquet 文件!")
        return

    indices_path = os.path.join(output_dir, "indices.json")
    indices_info = None
    expected_n = None

    if os.path.exists(indices_path):
        with open(indices_path, "r", encoding="utf-8") as f:
            indices_info = json.load(f)
            expected_n = indices_info["num_examples"]
            print(
                f"  读取到 indices.json: order={indices_info['order']}, num_examples={expected_n:,}")
    else:
        print(f"  [WARNING] 未找到 indices.json")

    with h5py.File(h5_path, "r") as f:
        h5_n = int(f["images"].shape[0])
        print(f"  H5 源样本数: {h5_n:,}")

    if expected_n is None:
        expected_n = h5_n

    print(f"\n  --- 读取 parquet 文件 ---")
    print(f"  读取 {len(parquet_files)} 个 parquet 分片...")
    ds = load_dataset("parquet", data_files=parquet_files, split="train")
    total_parquet = len(ds)
    print(f"  Parquet 总样本数: {total_parquet:,}")

    if total_parquet != expected_n:
        print(f"  [ERROR] 样本数不匹配: expected={expected_n}, actual={total_parquet}")
        return
    print(f"  ✓ 样本数匹配!")

    print(f"\n  --- 校验样本顺序 ---")
    num_check_samples = min(5, expected_n)

    with h5py.File(h5_path, "r") as f:
        if dataset_name == "OmniFace":
            expected_ids = [bytes_to_str(f["id"][i]) for i in range(num_check_samples)]
            actual_ids = [ds[i]["id"] for i in range(num_check_samples)]
            if expected_ids == actual_ids:
                print(f"  ✓ OmniFace 前 {num_check_samples} 个 id 匹配!")
            else:
                print(f"  [ERROR] OmniFace id 不匹配!")
                print(f"    expected: {expected_ids}")
                print(f"    actual: {actual_ids}")
                return

            if "source_index" in ds.features:
                expected_source_indices = list(range(num_check_samples))
                actual_source_indices = [ds[i]["source_index"] for i in range(num_check_samples)]
                if expected_source_indices == actual_source_indices:
                    print(f"  ✓ source_index 字段正确!")
                else:
                    print(
                        f"  [WARNING] source_index 不匹配: expected={expected_source_indices}, actual={actual_source_indices}")

        elif dataset_name == "OmniShape":
            expected_model_ids = [bytes_to_str(f["model_id"][i]) for i in range(num_check_samples)]
            expected_classes = [int(f["class"][i]) for i in range(num_check_samples)]
            expected_class100s = [int(f["class100"][i]) for i in range(num_check_samples)]
            expected_view_labels = f["view_label"][:num_check_samples].tolist()

            actual_model_ids = [ds[i]["model_id"] for i in range(num_check_samples)]
            actual_classes = [ds[i]["class"] for i in range(num_check_samples)]
            actual_class100s = [ds[i]["class100"] for i in range(num_check_samples)]
            actual_view_labels = [ds[i]["view_label"] for i in range(num_check_samples)]

            if expected_model_ids != actual_model_ids:
                print(f"  [ERROR] OmniShape model_id 不匹配!")
                return
            if expected_classes != actual_classes:
                print(f"  [ERROR] OmniShape class 不匹配!")
                return
            if expected_class100s != actual_class100s:
                print(f"  [ERROR] OmniShape class100 不匹配!")
                return
            if not np.allclose(expected_view_labels, actual_view_labels, atol=1e-6):
                print(f"  [ERROR] OmniShape view_label 不匹配!")
                return

            print(f"  ✓ OmniShape 前 {num_check_samples} 个样本字段匹配!")

            if "source_index" in ds.features:
                expected_source_indices = list(range(num_check_samples))
                actual_source_indices = [ds[i]["source_index"] for i in range(num_check_samples)]
                if expected_source_indices == actual_source_indices:
                    print(f"  ✓ source_index 字段正确!")
                else:
                    print(
                        f"  [WARNING] source_index 不匹配: expected={expected_source_indices}, actual={actual_source_indices}")

    print(f"\n  --- 数据集信息 ---")
    print(f"  字段: {list(ds.features.keys())}")

    sample = ds[0]
    img = sample["image"]
    print(f"  图片类型: {type(img)}")
    if hasattr(img, "size"):
        print(f"  图片尺寸: {img.size}")

    for j, k in enumerate(sample.keys()):
        if k == "image":
            continue
        v = sample[k]
        vr = repr(v)
        if len(vr) > 80:
            vr = vr[:80] + "..."
        print(f"    {k}: {vr}")
        if j >= 10:
            break

    meta_dir = os.path.join(output_dir, "meta")
    if os.path.exists(meta_dir):
        meta_files = [f for f in os.listdir(meta_dir) if f.endswith(".parquet")]
        print(f"\n  发现 meta 目录: {meta_dir}")
        print(f"  meta 文件: {meta_files}")

    print(f"\n  ✓ 验证完成! 所有检查通过!")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="H5 -> HuggingFace Datasets (Parquet) 转换工具（只输出 all split）",
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
        help="限制处理的最大样本数（全局前 N 条）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出目录",
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
            max_samples=args.max_samples,
            overwrite=args.overwrite,
        )
    else:
        convert_omnishape(
            h5_path=args.input,
            output_dir=args.output,
            max_shard_size_mb=args.shard_size,
            jpeg_quality=args.jpeg_quality,
            workers=args.workers,
            max_samples=args.max_samples,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
