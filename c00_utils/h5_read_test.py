# -*- coding: utf-8 -*-
# @Time    : 2026/4/27
# @Author  : OpenAI
# @Comments: 测试 H5 格式数据 batch 读取性能

"""
测试 H5 格式数据 batch 读取性能
==================================

功能:
1. 使用标准的 H5OmniDataset 类
2. 模拟训练时的 batch 持续循环读取
3. 计算读取速度、IO 速率等性能指标
4. 输出清晰的日志信息

使用示例:
```bash
# 测试 OmniFace (64x64)
python h5_read_test.py \
    --input /home/data/OmniFace_64x64_20260421.h5 \
    --datasource OmniFace \
    --size 64 \
    --batch_size 128 \
    --num_batches 100

# 测试 OmniShape (128x128)
python h5_read_test.py \
    --input /home/data/OmniShape1k_18000a_128x128_20251204.h5 \
    --datasource OmniShape \
    --size 128 \
    --batch_size 128 \
    --num_batches 100
```
"""

import argparse
import time
import sys
import gc

import torch
from tqdm import tqdm

from h5_omni_dataset import H5OmniDataset


def test_h5_read(
    h5_path: str,
    datasource: str = "OmniShape",
    size: int = None,
    split: str = "train",
    batch_size: int = 128,
    num_batches: int = 100,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True
):
    print(f"\n{'=' * 70}")
    print(f"测试 H5 格式数据 batch 读取性能")
    print(f"{'=' * 70}")
    print(f"  输入文件: {h5_path}")
    print(f"  数据集类型: {datasource}")
    print(f"  图像尺寸: {size if size else '使用文件中的尺寸'}")
    print(f"  数据集分割: {split}")
    print(f"  Batch 大小: {batch_size}")
    print(f"  测试批次数: {num_batches}")
    print(f"  是否打乱: {shuffle}")
    print(f"  工作进程数: {num_workers}")
    print(f"  是否固定内存: {pin_memory}")
    print()

    # 创建数据集
    print(f"  正在创建数据集...")
    start_time = time.time()
    dataset = H5OmniDataset(
        path=h5_path,
        size=size,
        datasource=datasource,
        split=split
    )
    dataset_create_time = time.time() - start_time
    print(f"  数据集创建完成，耗时: {dataset_create_time:.4f}秒")
    print(f"  数据集总样本数: {len(dataset):,}")
    print()

    # 创建 DataLoader
    print(f"  正在创建 DataLoader...")
    start_time = time.time()
    dataloader = dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    dataloader_create_time = time.time() - start_time
    print(f"  DataLoader 创建完成，耗时: {dataloader_create_time:.4f}秒")
    print()

    # 测试读取性能
    print("  开始测试 batch 读取...")
    print("  " + "-" * 60)

    total_samples = 0
    total_time = 0
    total_bytes = 0
    first_batch_time = None

    with tqdm(range(num_batches), file=sys.stdout) as pbar:
        for batch_idx in pbar:
            try:
                start_time = time.time()

                # 使用循环迭代
                batch = next(iter(dataloader))

                end_time = time.time()
                batch_time = end_time - start_time

                # 记录第一个 batch 的时间
                if first_batch_time is None:
                    first_batch_time = batch_time

                # 统计信息
                batch_size_actual = len(batch['image'])
                total_samples += batch_size_actual
                total_time += batch_time

                # 计算读取的数据量（字节）（近似）
                # 图像大小 + 其他标签大小
                img_bytes = batch['image'].nelement() * batch['image'].element_size()
                other_bytes = 0
                for k, v in batch.items():
                    if k != 'image':
                        if isinstance(v, torch.Tensor):
                            other_bytes += v.nelement() * v.element_size()
                        else:
                            other_bytes += sys.getsizeof(v)

                batch_bytes = img_bytes + other_bytes
                total_bytes += batch_bytes

                # 更新进度条
                avg_speed = total_samples / total_time if total_time > 0 else 0
                avg_io = total_bytes / (1024 * 1024 * total_time) if total_time > 0 else 0
                pbar.set_postfix({
                    'batch_speed': f'{batch_size_actual / batch_time:.1f}img/s',
                    'avg_speed': f'{avg_speed:.1f}img/s',
                    'avg_io': f'{avg_io:.2f}MB/s'
                })

                # 每 10 个 batch 输出详细信息
                if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                    batch_speed = batch_size_actual / batch_time if batch_time > 0 else 0
                    batch_io = batch_bytes / (1024 * 1024 * batch_time) if batch_time > 0 else 0
                    print(f"  Batch {batch_idx + 1:4d}: 样本={batch_size_actual:4d}, 耗时={batch_time:.4f}s, 速度={batch_speed:.1f}img/s, IO={batch_io:.2f}MB/s")

                # 定期清理内存
                if batch_idx % 50 == 0:
                    del batch
                    gc.collect()

            except Exception as e:
                print(f"  错误: Batch {batch_idx} 失败: {e}")
                continue

    print("  " + "-" * 60)
    print("  测试完成!")
    print()

    # 计算平均性能
    avg_speed = total_samples / total_time if total_time > 0 else 0
    avg_io = total_bytes / (1024 * 1024 * total_time) if total_time > 0 else 0

    print("  " + "=" * 50)
    print("  总统计信息:")
    print("  " + "=" * 50)
    print(f"    总样本数: {total_samples:,}")
    print(f"    总耗时: {total_time:.4f}秒")
    print(f"    第一个 batch 耗时: {first_batch_time:.4f}秒 (包含初始化)")
    print(f"    平均速度: {avg_speed:.2f} img/s")
    print(f"    平均 IO 速率: {avg_io:.2f} MB/s")
    print(f"    总数据量: {total_bytes / (1024 * 1024):.2f} MB")
    print()

    # 评估性能
    print("  性能评估:")
    if avg_speed > 1000:
        print("  ✅ 优秀：读取速度很快，适合高负载训练")
    elif avg_speed > 500:
        print("  ✅ 良好：读取速度不错，可以满足大多数训练需求")
    elif avg_speed > 200:
        print("  ⚠️ 一般：读取速度尚可，可能成为训练瓶颈")
    else:
        print("  ❌ 较慢：读取速度较慢，建议优化（增加 num_workers，使用 SSD 等）")
    print()

    # 清理资源
    dataset.close()
    print("  资源已清理")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="测试 H5 格式数据 batch 读取性能",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", "-i", required=True, help="H5 文件路径"
    )
    parser.add_argument(
        "--datasource", type=str, default="OmniShape",
        help="数据集类型：OmniFace 或 OmniShape",
        choices=["OmniFace", "OmniShape"]
    )
    parser.add_argument(
        "--size", type=int, default=None,
        help="图像尺寸（默认使用文件中的尺寸）"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="数据集分割：train 或 val"
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=128,
        help="Batch 大小（默认 128）"
    )
    parser.add_argument(
        "--num_batches", "-n", type=int, default=100,
        help="测试批次数（默认 100）"
    )
    parser.add_argument(
        "--no_shuffle", action="store_true",
        help="不打乱数据（默认会打乱）"
    )
    parser.add_argument(
        "--num_workers", "-w", type=int, default=8,
        help="工作进程数（默认 8）"
    )
    parser.add_argument(
        "--no_pin_memory", action="store_true",
        help="不固定内存（默认固定）"
    )

    args = parser.parse_args()

    test_h5_read(
        h5_path=args.input,
        datasource=args.datasource,
        size=args.size,
        split=args.split,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        shuffle=not args.no_shuffle,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory
    )


if __name__ == "__main__":
    main()
