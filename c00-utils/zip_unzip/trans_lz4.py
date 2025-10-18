#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import stat
import time
import lz4.frame
import logging
from tqdm import tqdm

# ===== 打包格式（顺序写入 LZ4 流）=====
# [magic: 7 bytes = b"LZ4DIR1"]
# 重复以下记录直到结束标记：
#   [path_len: uint32 little]
#   [path_bytes: utf-8, length = path_len]
#   [file_size: uint64 little]
#   [mtime_sec: int64 little]   # 修改时间（秒）
#   [mode: uint32 little]       # 权限位（如 0o644）
#   [file_bytes: length = file_size]
# 结束标记：
#   [path_len=0 (uint32 little)]

MAGIC = b"LZ4DIR1"
CHUNK = 4 * 1024 * 1024  # 4MB

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _write_u32(f, x): f.write(int(x).to_bytes(4, 'little', signed=False))
def _write_u64(f, x): f.write(int(x).to_bytes(8, 'little', signed=False))
def _write_i64(f, x): f.write(int(x).to_bytes(8, 'little', signed=True))

def compress_one_subfolder(subfolder_path: str, output_lz4_path: str, compression_level: int = 0):
    os.makedirs(os.path.dirname(output_lz4_path), exist_ok=True)

    # 收集文件清单
    file_list = []
    for root, _, files in os.walk(subfolder_path):
        for name in files:
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, subfolder_path)
            file_list.append((abs_path, rel_path))
    if not file_list:
        logging.warning(f"空文件夹，跳过：{subfolder_path}")
        return

    logging.info(f"开始压缩：{subfolder_path} -> {output_lz4_path}（{len(file_list)} 个文件）")

    # 打开 LZ4 输出流
    with lz4.frame.open(output_lz4_path, mode='wb', compression_level=compression_level) as out_f:
        # 写魔数
        out_f.write(MAGIC)

        # 逐文件写入记录
        for abs_path, rel_path in file_list:
            st = os.stat(abs_path)
            size = st.st_size
            mtime = int(st.st_mtime)
            mode = stat.S_IMODE(st.st_mode)

            rel_bytes = rel_path.encode('utf-8')
            _write_u32(out_f, len(rel_bytes))
            out_f.write(rel_bytes)
            _write_u64(out_f, size)
            _write_i64(out_f, mtime)
            _write_u32(out_f, mode)

            # 流式写入文件内容 + 每文件进度条
            with open(abs_path, 'rb') as in_f, tqdm(
                total=size, unit='B', unit_scale=True, unit_divisor=1024,
                desc=f"Compressing {rel_path}", leave=False
            ) as pbar:
                while True:
                    buf = in_f.read(CHUNK)
                    if not buf:
                        break
                    out_f.write(buf)
                    pbar.update(len(buf))

        # 写结束标记
        _write_u32(out_f, 0)

    logging.info(f"完成：{output_lz4_path}")

def process_parent(input_parent: str, compression_level: int = 0):
    output_parent = f"{input_parent}_trans"
    os.makedirs(output_parent, exist_ok=True)

    for name in os.listdir(input_parent):
        sub_path = os.path.join(input_parent, name)
        if os.path.isdir(sub_path):
            out_path = os.path.join(output_parent, f"{name}.lz4")
            compress_one_subfolder(sub_path, out_path, compression_level)

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("用法: python trans_lz4.py <父文件夹路径> [压缩等级(0~16, 可选，默认0)]")
        sys.exit(1)
    parent = sys.argv[1]
    level = int(sys.argv[2]) if len(sys.argv) == 3 else 0
    if not os.path.isdir(parent):
        print(f"错误：无效目录 {parent}")
        sys.exit(2)
    process_parent(parent, compression_level=level)