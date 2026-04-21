# -*- coding: utf-8 -*-
# @Time    : 2026/4/21 10:29
# @Author  : CFuShn
# @Comments: H5 图片下采样工具 (128x128 / 512x512 -> 64x64)
# @Software: PyCharm

"""
cd /home/cy/nuist-lab/cfs-code-lab/c00_utils

# OmniShape 128->64
python h5_to_64.py -d omnishape \
    -i /home/data/OmniShape1k_18000a_128x128_20251204.h5 \
    -o /home/data/OmniShape1k_18000a_64x64_20260421.h5

# OmniFace 512->64
python h5_to_64.py -d omniface \
    -i /home/data/OmniFace_202602042244.h5 \
    -o /home/data/OmniFace_64x64_20260421.h5

H5 图片下采样工具
==================

将 OmniShape (128x128) 和 OmniFace (512x512 JPEG) 各生成一个 64x64 版本,
统一为 uint8 RGB 数组 (N,3,64,64) + gzip 压缩存储.

用法:
  # OmniShape 128->64
  python h5_to_64.py -d omnishape \
      -i /home/data/OmniShape1k_18000a_128x128_20251204.h5 \
      -o /home/data/OmniShape1k_18000a_64x64_20260421.h5

  # OmniFace 512->64
  python h5_to_64.py -d omniface \
      -i /home/data/OmniFace_202602042244.h5 \
      -o /home/data/OmniFace_64x64_20260421.h5

  # 自定义 workers
  python h5_to_64.py -d omnishape -i ... -o ... -w 32
"""

import argparse
import io
import os
import sys
import time
from multiprocessing import Pool
from typing import List, Optional

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# ============================================================================
# Phase 1: 通用工具函数
# ============================================================================

def _resize_single_hwc(args):
    """worker: (hwc_array, target_size) -> resized_hwc_array"""
    img_hwc, target_size = args
    pil_img = Image.fromarray(img_hwc)
    pil_img = pil_img.resize((target_size, target_size), Image.LANCZOS)
    return np.array(pil_img, dtype=np.uint8)


def resize_chw_batch(
    imgs_chw: np.ndarray,
    target_size: int = 64,
    workers: int = 16,
) -> np.ndarray:
    """
    批量 resize: (B,3,H,W) uint8 -> (B,3,target_size,target_size) uint8.
    使用 LANCZOS 滤波器, 多进程并行.
    """
    B = imgs_chw.shape[0]
    # 批量 CHW -> HWC
    imgs_hwc = np.transpose(imgs_chw, (0, 2, 3, 1)).copy()

    if workers <= 1 or B <= 64:
        resized = np.stack([
            _resize_single_hwc((imgs_hwc[i], target_size)) for i in range(B)
        ])
    else:
        tasks = [(imgs_hwc[i], target_size) for i in range(B)]
        with Pool(processes=workers) as pool:
            resized_list = pool.map(
                _resize_single_hwc, tasks, chunksize=max(1, B // (workers * 4))
            )
        resized = np.stack(resized_list)

    # HWC -> CHW
    return np.transpose(resized, (0, 3, 1, 2)).copy()


def _decode_jpeg_to_hwc(args):
    """worker: (jpeg_bytes, target_size) -> resized_hwc_array"""
    jpeg_bytes, target_size = args
    pil_img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    pil_img = pil_img.resize((target_size, target_size), Image.LANCZOS)
    return np.array(pil_img, dtype=np.uint8)


def decode_jpegs_to_chw(
    jpeg_list: list,
    target_size: int = 64,
    workers: int = 16,
) -> np.ndarray:
    """
    批量 JPEG 解码 + resize: bytes list -> (B,3,target_size,target_size) uint8.
    使用 LANCZOS 滤波器, 多进程并行.
    """
    B = len(jpeg_list)

    if workers <= 1 or B <= 64:
        hwc_list = [
            _decode_jpeg_to_hwc((jpeg_list[i], target_size)) for i in range(B)
        ]
    else:
        tasks = [(jpeg_list[i], target_size) for i in range(B)]
        with Pool(processes=workers) as pool:
            hwc_list = pool.map(
                _decode_jpeg_to_hwc, tasks, chunksize=max(1, B // (workers * 4))
            )

    # list of (H,W,3) -> (B,3,H,W)
    hwc_batch = np.stack(hwc_list)
    return np.transpose(hwc_batch, (0, 3, 1, 2)).copy()


def copy_h5_dataset(
    src: h5py.File,
    dst: h5py.File,
    name: str,
    indices: Optional[np.ndarray] = None,
):
    """
    复制一个 h5 dataset, 保持 dtype/shape/compression/chunks 一致.

    Args:
        src: 源 h5 文件句柄
        dst: 目标 h5 文件句柄
        name: dataset 名称 (如 "age" 或 "meta/model_id")
        indices: 可选, 按索引选取子集复制
    """
    src_ds = src[name]

    if indices is not None:
        data = src_ds[indices]
        shape = data.shape if isinstance(data, np.ndarray) else (len(indices),)
    else:
        data = src_ds[:]
        shape = src_ds.shape

    # 构建创建参数
    kwargs = {
        "dtype": src_ds.dtype,
        "compression": src_ds.compression,
        "shuffle": bool(src_ds.shuffle) if src_ds.shuffle else False,
        "fletcher32": bool(src_ds.fletcher32) if src_ds.fletcher32 else False,
    }
    if src_ds.compression_opts is not None:
        kwargs["compression_opts"] = src_ds.compression_opts
    if src_ds.chunks is not None:
        # chunk 的第一维按实际 shape 调整
        old_chunks = list(src_ds.chunks)
        if indices is not None and len(shape) == len(old_chunks):
            old_chunks[0] = min(old_chunks[0], shape[0])
        kwargs["chunks"] = tuple(old_chunks)
    if src_ds.fillvalue is not None and src_ds.fillvalue != 0:
        kwargs["fillvalue"] = src_ds.fillvalue

    dst.create_dataset(name, shape=shape, **kwargs)
    dst[name][:] = data


def copy_h5_group(
    src: h5py.File,
    dst: h5py.File,
    group_name: str,
):
    """复制一个 h5 group 及其所有 dataset."""
    src_grp = src[group_name]
    dst.create_group(group_name)

    for key in src_grp.keys():
        full_name = f"{group_name}/{key}"
        copy_h5_dataset(src, dst, full_name)


# ============================================================================
# Phase 2: OmniShape 128 -> 64
# ============================================================================

# OmniShape 源文件中需要复制的非 images 字段
OMNISHAPE_LABEL_FIELDS = [
    "2d_coverage", "2d_hw_ratio", "2d_rgb_complexity", "2d_silhouette_complexity",
    "anisotropy", "class", "class100", "hull_volume",
    "mat_complexity", "mat_count", "mat_slots",
    "model_id", "surface_area_ratio", "vert_count",
    "view_label", "volume", "volume_ratio", "xyz_size",
]


def convert_omnishape(
    h5_path: str,
    output_path: str,
    target_size: int = 64,
    batch_size: int = 10920,
    workers: int = 16,
):
    """OmniShape 128x128 -> 64x64."""
    print(f"\n{'='*60}")
    print(f"OmniShape 下采样: {h5_path}")
    print(f"  -> {output_path} ({target_size}x{target_size})")
    print(f"{'='*60}\n")

    with h5py.File(h5_path, "r") as src:
        N = src["images"].shape[0]
        src_chunks = src["images"].chunks  # (1365, 3, 128, 128)
        print(f"  源数据: {N:,} 张, {src['images'].shape[2]}x{src['images'].shape[3]}")
        print(f"  源 chunk: {src_chunks}")

        # 创建输出 h5
        with h5py.File(output_path, "w") as dst:
            # 创建 images dataset
            new_chunks = (src_chunks[0], 3, target_size, target_size)
            dst.create_dataset(
                "images",
                shape=(N, 3, target_size, target_size),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=4,
                chunks=new_chunks,
                fillvalue=0,
            )
            print(f"  输出 images: uint8 ({N},3,{target_size},{target_size}), chunk={new_chunks}")

            # 分批 resize 写入
            pbar = tqdm(total=N, desc="  [resize]", unit="img", unit_scale=True, ncols=120)
            t0 = time.time()

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                imgs_chw = src["images"][start:end]  # (B, 3, 128, 128)
                resized = resize_chw_batch(imgs_chw, target_size, workers)
                dst["images"][start:end] = resized
                pbar.update(end - start)

            pbar.close()
            elapsed = time.time() - t0
            print(f"  resize 完成, 耗时 {elapsed/60:.1f} min")

            # 复制标签字段
            print(f"  复制 {len(OMNISHAPE_LABEL_FIELDS)} 个标签字段...")
            for field in OMNISHAPE_LABEL_FIELDS:
                copy_h5_dataset(src, dst, field)
                pbar_fields = tqdm(
                    [field], desc="  [fields]", unit="field", ncols=120, leave=False,
                )
                pbar_fields.close()

            # 复制 meta/ 子组
            print(f"  复制 meta/ 子组...")
            copy_h5_group(src, dst, "meta")

    # 验证输出
    with h5py.File(output_path, "r") as f:
        out_size = f.id.get_filesize()
        print(f"\n  输出文件: {output_path}")
        print(f"  文件大小: {out_size / 1e9:.2f} GB")
        print(f"  images shape: {f['images'].shape}, dtype: {f['images'].dtype}")
        print(f"  字段列表: {sorted(f.keys())}")
        if "meta" in f:
            print(f"  meta/ 子字段: {sorted(f['meta'].keys())}")


# ============================================================================
# Phase 3: OmniFace 512 -> 64
# ============================================================================

# OmniFace 源文件中需要复制的非 images 字段
OMNIFACE_LABEL_FIELDS = [
    "age", "arousal", "valence",
    "arched_eyebrows", "attractive", "bags_under_eyes", "bald", "bangs",
    "big_lips", "big_nose", "black_hair", "blond_hair", "blurry",
    "brown_hair", "bushy_eyebrows", "chubby", "double_chin",
    "expression", "eyeglasses", "five_o_clock_shadow", "goatee",
    "gray_hair", "heavy_makeup", "high_cheekbones",
    "male", "mouth_slightly_open", "mustache", "narrow_eyes",
    "no_beard", "oval_face", "pale_skin", "pointy_nose",
    "race", "receding_hairline", "rosy_cheeks", "sideburns",
    "smiling", "straight_hair", "wavy_hair",
    "wearing_earrings", "wearing_hat", "wearing_lipstick",
    "wearing_necklace", "wearing_necktie", "young",
    "gaze_dir", "head_pose",
    "id", "origin", "prompt", "is_sr",
]


def convert_omniface(
    h5_path: str,
    output_path: str,
    target_size: int = 64,
    batch_size: int = 8192,
    workers: int = 16,
):
    """OmniFace 512x512 JPEG -> 64x64 uint8."""
    print(f"\n{'='*60}")
    print(f"OmniFace 下采样: {h5_path}")
    print(f"  -> {output_path} ({target_size}x{target_size})")
    print(f"{'='*60}\n")

    with h5py.File(h5_path, "r") as src:
        N = src["images"].shape[0]
        print(f"  源数据: {N:,} 张 (JPEG 字节流)")

        # 创建输出 h5
        with h5py.File(output_path, "w") as dst:
            # 创建 images dataset: uint8 (N,3,64,64)
            dst.create_dataset(
                "images",
                shape=(N, 3, target_size, target_size),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=4,
                chunks=(1024, 3, target_size, target_size),
                fillvalue=0,
            )
            print(f"  输出 images: uint8 ({N},3,{target_size},{target_size}), chunk=(1024,3,{target_size},{target_size})")

            # 分批 JPEG 解码 + resize 写入
            pbar = tqdm(total=N, desc="  [decode+resize]", unit="img", unit_scale=True, ncols=120)
            t0 = time.time()

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                jpeg_batch = [src["images"][i] for i in range(start, end)]
                imgs_chw = decode_jpegs_to_chw(jpeg_batch, target_size, workers)
                dst["images"][start:end] = imgs_chw
                pbar.update(end - start)

            pbar.close()
            elapsed = time.time() - t0
            print(f"  decode+resize 完成, 耗时 {elapsed/60:.1f} min")

            # 复制标签字段
            print(f"  复制 {len(OMNIFACE_LABEL_FIELDS)} 个标签字段...")
            for field in tqdm(OMNIFACE_LABEL_FIELDS, desc="  [fields]", unit="field", ncols=120):
                copy_h5_dataset(src, dst, field)

            # 复制 train_indices / val_indices
            print(f"  复制 train_indices / val_indices...")
            copy_h5_dataset(src, dst, "train_indices")
            copy_h5_dataset(src, dst, "val_indices")

    # 验证输出
    with h5py.File(output_path, "r") as f:
        out_size = f.id.get_filesize()
        print(f"\n  输出文件: {output_path}")
        print(f"  文件大小: {out_size / 1e9:.2f} GB")
        print(f"  images shape: {f['images'].shape}, dtype: {f['images'].dtype}")
        print(f"  字段列表: {sorted(f.keys())}")


# ============================================================================
# Phase 4: CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="H5 图片下采样工具 (128/512 -> 64)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=["omnishape", "omniface"],
        required=True,
        help="数据集名称",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="源 H5 文件路径",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出 H5 文件路径",
    )
    parser.add_argument(
        "--target-size", "-s",
        type=int,
        default=64,
        help="目标分辨率 (默认 64)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=0,
        help="每批处理数量 (0=自动: OmniShape 10920, OmniFace 8192)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=16,
        help="并行进程数 (默认 16)",
    )

    args = parser.parse_args()

    if args.dataset == "omnishape":
        batch_size = args.batch_size if args.batch_size > 0 else 10920
        convert_omnishape(
            h5_path=args.input,
            output_path=args.output,
            target_size=args.target_size,
            batch_size=batch_size,
            workers=args.workers,
        )
    elif args.dataset == "omniface":
        batch_size = args.batch_size if args.batch_size > 0 else 8192
        convert_omniface(
            h5_path=args.input,
            output_path=args.output,
            target_size=args.target_size,
            batch_size=batch_size,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
