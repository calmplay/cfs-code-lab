#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import stat
import time
import lz4.frame
import logging
from tqdm import tqdm

MAGIC = b"LZ4DIR1"
CHUNK = 4 * 1024 * 1024  # 4MB

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _read_exact(f, n):
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise EOFError("意外到达文件结尾")
        buf += chunk
    return buf

def _read_u32(f): return int.from_bytes(_read_exact(f, 4), 'little', signed=False)
def _read_u64(f): return int.from_bytes(_read_exact(f, 8), 'little', signed=False)
def _read_i64(f): return int.from_bytes(_read_exact(f, 8), 'little', signed=True)

def extract_one_lz4(lz4_path: str, output_root: str):
    sub_name = os.path.splitext(os.path.basename(lz4_path))[0]
    out_dir = os.path.join(output_root, sub_name)
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"开始解压：{lz4_path} -> {out_dir}")

    with lz4.frame.open(lz4_path, mode='rb') as in_f:
        # 检查魔数
        magic = in_f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"{lz4_path} 不是有效的 LZ4DIR 包（魔数不匹配）")

        while True:
            path_len = _read_u32(in_f)
            if path_len == 0:
                break  # 结束标记
            rel_path = _read_exact(in_f, path_len).decode('utf-8')
            size = _read_u64(in_f)
            mtime = _read_i64(in_f)
            mode = _read_u32(in_f)

            out_file = os.path.join(out_dir, rel_path)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)

            with open(out_file, 'wb') as f_out, tqdm(
                total=size, unit='B', unit_scale=True, unit_divisor=1024,
                desc=f"Extracting {rel_path}", leave=False
            ) as pbar:
                remaining = size
                while remaining > 0:
                    to_read = CHUNK if remaining >= CHUNK else remaining
                    buf = _read_exact(in_f, to_read)
                    f_out.write(buf)
                    remaining -= len(buf)
                    pbar.update(len(buf))

            # 还原权限与时间戳
            try:
                os.chmod(out_file, mode)
            except Exception:
                pass
            try:
                os.utime(out_file, (mtime, mtime))
            except Exception:
                pass

    logging.info(f"完成解压：{lz4_path}")

def process_input(input_path: str):
    output_root = f"{input_path}_extracted"
    os.makedirs(output_root, exist_ok=True)

    # 支持传入父目录（包含多个 .lz4）或单个 .lz4 文件
    if os.path.isdir(input_path):
        targets = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(".lz4")]
        if not targets:
            logging.error("未在目录中找到 .lz4 文件")
            sys.exit(3)
    else:
        targets = [input_path] if input_path.lower().endswith(".lz4") else []
        if not targets:
            logging.error("输入既不是目录也不是 .lz4 文件")
            sys.exit(4)

    for lz4_file in targets:
        extract_one_lz4(lz4_file, output_root)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python open_lz4.py <包含 .lz4 的目录 或 单个 .lz4 文件>")
        sys.exit(1)
    process_input(sys.argv[1])