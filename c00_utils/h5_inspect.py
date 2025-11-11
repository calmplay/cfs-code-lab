# -*- coding: utf-8 -*-
# @Time    : 2025/9/16 13:54
# @Author  : CFuShn
# @Comments: 
# @Software: PyCharm

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDF5 结构分析脚本
- 列出 Groups / Datasets 的树结构
- 显示 dtype / shape / maxshape / compression / chunks / filters / fillvalue
- 显示属性（可选）
- 识别软链接 / 外部链接；统计硬链接计数
- 可选 JSON 输出
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import h5py

# ├─/└─ 树形前缀
def _tree_prefix(is_last_stack: List[bool]) -> str:
    parts = []
    for i, is_last in enumerate(is_last_stack[:-1]):
        parts.append("   " if is_last else "│  ")
    parts.append("└─ " if (is_last_stack and is_last_stack[-1]) else "├─ ")
    return "".join(parts)

def _obj_hard_link_count(obj) -> Optional[int]:
    try:
        info = h5py.h5o.get_info(obj.id)
        return int(info.nlink)
    except Exception:
        return None

def _dataset_filters(ds: h5py.Dataset) -> Dict[str, Any]:
    # 参考 h5py Dataset 属性
    f = {}
    try:
        f["compression"] = ds.compression
    except Exception:
        pass
    try:
        f["compression_opts"] = ds.compression_opts
    except Exception:
        pass
    try:
        f["shuffle"] = bool(ds.shuffle)
    except Exception:
        pass
    try:
        f["fletcher32"] = bool(ds.fletcher32)
    except Exception:
        pass
    try:
        f["scaleoffset"] = ds.scaleoffset
    except Exception:
        pass
    try:
        f["chunks"] = tuple(ds.chunks) if ds.chunks is not None else None
    except Exception:
        pass
    try:
        f["fillvalue"] = ds.fillvalue
    except Exception:
        pass
    return f

def _safe_attrs(h5obj, max_items: int = 10) -> Dict[str, Any]:
    out = {}
    try:
        keys = list(h5obj.attrs.keys())
    except Exception:
        return out
    for i, k in enumerate(keys):
        if i >= max_items:
            out["__truncated__"] = True
            break
        try:
            v = h5obj.attrs[k]
            # 转成可 JSON 化
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating, np.bool_)):
                out[k] = v.item()
            else:
                # bytes 转 str
                if isinstance(v, (bytes, bytearray)):
                    try:
                        out[k] = v.decode("utf-8", errors="replace")
                    except Exception:
                        out[k] = str(v)
                else:
                    out[k] = v
        except Exception as e:
            out[k] = f"<attr read error: {e}>"
    return out

def _ds_preview(ds: h5py.Dataset, sample: int) -> Optional[List[Any]]:
    if sample <= 0:
        return None
    try:
        # 小心超大数据集：只切前几个元素的扁平视图
        it = np.nditer(np.zeros((1,)))  # 触发 import numpy，防止打包工具警告
        arr = ds[tuple(slice(0, min(n, 1)) for n in ds.shape)] if ds.shape else ds[()]
        flat = np.array(arr).ravel()
        out = []
        for i in range(min(sample, flat.size)):
            val = flat[i]
            if isinstance(val, (np.generic,)):
                val = val.item()
            out.append(val)
        return out
    except Exception:
        return None

def _link_info(grp: h5py.Group, name: str) -> Optional[Dict[str, Any]]:
    # 识别链接类型：HardLink / SoftLink / ExternalLink
    try:
        lnk = grp.get(name, getlink=True)
        if isinstance(lnk, h5py.HardLink):
            return {"type": "hard"}
        elif isinstance(lnk, h5py.SoftLink):
            return {"type": "soft", "path": lnk.path}
        elif isinstance(lnk, h5py.ExternalLink):
            return {"type": "external", "filename": lnk.filename, "path": lnk.path}
        else:
            return {"type": "unknown"}
    except Exception:
        return None

def analyze_h5(
    path: str,
    depth: Optional[int],
    show_attrs: bool,
    attr_max: int,
    sample: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"file": path, "tree": {}}
    with h5py.File(path, "r") as f:
        # 文件级信息
        meta: Dict[str, Any] = {}
        try:
            meta["driver"] = f.driver
        except Exception:
            pass
        try:
            meta["libver"] = f.libver
        except Exception:
            pass
        try:
            meta["userblock_size"] = f.userblock_size
        except Exception:
            pass
        try:
            meta["swmr_mode"] = bool(f.swmr_mode)
        except Exception:
            pass
        try:
            # 低层获取文件大小
            meta["filesize"] = int(f.id.get_filesize())
        except Exception:
            pass
        meta["attrs"] = _safe_attrs(f, attr_max) if show_attrs else {}
        result["file_info"] = meta

        def walk(group: h5py.Group, node: Dict[str, Any], cur_depth: int):
            if depth is not None and cur_depth > depth:
                node["__truncated__"] = True
                return
            node["type"] = "group"
            node["attrs"] = _safe_attrs(group, attr_max) if show_attrs else {}
            children: Dict[str, Any] = {}
            for key in sorted(group.keys()):
                info = _link_info(group, key) or {}
                try:
                    obj = group.get(key, getlink=False)
                except Exception as e:
                    children[key] = {"type": "unreadable", "error": str(e), "link": info}
                    continue

                # 数据集
                if isinstance(obj, h5py.Dataset):
                    ds: h5py.Dataset = obj
                    ds_info: Dict[str, Any] = {
                        "type": "dataset",
                        "dtype": str(ds.dtype),
                        "shape": tuple(ds.shape) if ds.shape is not None else None,
                        "maxshape": tuple(ds.maxshape) if ds.maxshape is not None else None,
                        "ndim": int(ds.ndim),
                        "size": int(ds.size),
                        "filters": _dataset_filters(ds),
                        "is_virtual": bool(getattr(ds, "is_virtual", False)),
                        "link": info,
                    }
                    nlink = _obj_hard_link_count(ds)
                    if nlink is not None:
                        ds_info["hard_links"] = nlink
                    if show_attrs:
                        ds_info["attrs"] = _safe_attrs(ds, attr_max)
                    if sample > 0:
                        ds_info["sample"] = _ds_preview(ds, sample)
                    children[key] = ds_info

                # 组
                elif isinstance(obj, h5py.Group):
                    sub_node: Dict[str, Any] = {"link": info}
                    nlink = _obj_hard_link_count(obj)
                    if nlink is not None:
                        sub_node["hard_links"] = nlink
                    children[key] = sub_node
                    walk(obj, children[key], cur_depth + 1)

                else:
                    # 未知对象（很少见）
                    children[key] = {"type": str(type(obj)), "link": info}

            node["children"] = children

        result["tree"] = {}
        walk(f["/"], result["tree"], 0)
    return result

def print_tree(
    node: Dict[str, Any],
    name: str,
    is_last_stack: List[bool],
    show_attrs: bool,
    attr_max: int,
):
    prefix = _tree_prefix(is_last_stack) if is_last_stack else ""
    typ = node.get("type", "group")
    link = node.get("link") or {}
    link_tag = ""
    if link:
        t = link.get("type")
        if t == "soft":
            link_tag = f" [soft→{link.get('path')}]"
        elif t == "external":
            link_tag = f" [external→{link.get('filename')}:{link.get('path')}]"

    if typ == "group":
        print(f"{prefix}{name}/ (group){link_tag}")
        if "__truncated__" in node:
            print(f"{'   ' * len(is_last_stack)}… (depth truncated)")
        if show_attrs and node.get("attrs"):
            _print_attrs(node["attrs"], len(is_last_stack))
        children = node.get("children", {})
        keys = list(children.keys())
        for i, k in enumerate(keys):
            child = children[k]
            print_tree(
                child,
                k,
                is_last_stack + [i == len(keys) - 1],
                show_attrs,
                attr_max,
            )
    elif typ == "dataset":
        info = []
        if "dtype" in node:
            info.append(node["dtype"])
        if "shape" in node and node["shape"] is not None:
            info.append(f"shape={node['shape']}")
        filt = node.get("filters", {})
        if any(v is not None and v is not False for v in filt.values()):
            info.append(
                "filters="
                + ",".join(
                    f"{k}={v}"
                    for k, v in filt.items()
                    if v is not None and v is not False
                )
            )
        if "is_virtual" in node and node["is_virtual"]:
            info.append("virtual=True")
        hl = node.get("hard_links")
        if hl is not None and hl > 1:
            info.append(f"hard_links={hl}")
        print(f"{prefix}{name} (dataset){link_tag} :: " + "; ".join(info))
        if show_attrs and node.get("attrs"):
            _print_attrs(node["attrs"], len(is_last_stack))
        if "sample" in node and node["sample"] is not None:
            indent = "   " * (len(is_last_stack) + 1)
            print(f"{indent}sample: {node['sample']}")
    else:
        print(f"{prefix}{name} ({typ}){link_tag}")

def _print_attrs(attrs: Dict[str, Any], depth: int):
    indent = "   " * (depth + 1)
    for i, (k, v) in enumerate(attrs.items()):
        if k == "__truncated__":
            print(f"{indent}… (attrs truncated)")
            continue
        sv = v
        if isinstance(v, (list, tuple)) and len(v) > 8:
            sv = v[:8] + ["…"]
        print(f"{indent}@{k} = {sv!r}")

def main():
    p = argparse.ArgumentParser(
        description="HDF5(.h5/.hdf5) 结构分析器"
    )
    p.add_argument("file", help="HDF5 文件路径")
    p.add_argument("-d", "--depth", type=int, default=None, help="最大递归深度（默认不限）")
    p.add_argument("-a", "--attrs", action="store_true", help="打印对象属性")
    p.add_argument("--attr-max", type=int, default=10, help="每个对象最多显示的属性键数")
    p.add_argument("-j", "--json", action="store_true", help="输出 JSON（而不是树形文本）")
    p.add_argument("--sample", type=int, default=0, help="数据集采样显示的元素个数（默认不采样）")
    args = p.parse_args()

    try:
        info = analyze_h5(args.file, args.depth, args.attrs, args.attr_max, args.sample)
    except (OSError, IOError) as e:
        print(f"[ERR] 打不开文件：{e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[ERR] 解析失败：{e}", file=sys.stderr)
        sys.exit(3)

    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2, default=str))
    else:
        fi = info.get("file_info", {})
        print(f"# File: {info.get('file')}")
        if "filesize" in fi:
            print(f"# Size: {fi['filesize']} bytes")
        if "driver" in fi:
            print(f"# Driver: {fi['driver']}")
        if "libver" in fi:
            print(f"# Libver: {fi['libver']}")
        if args.attrs and fi.get("attrs"):
            print("# File attrs:")
            _print_attrs(fi["attrs"], 0)
        print()
        # 打印树
        root = info.get("tree", {})
        print_tree(root, "/", [], args.attrs, args.attr_max)

if __name__ == "__main__":
    main()

    # 使用示例:
    # # 基本用法：打印树
    # python h5_inspect.py your_file.h5
    # python /home/cy/nuist-lab/cfs-code-lab/c00-utils/h5_inspect.py /home/cy/datasets/facial/MixedFace/MixedFace_202510201043.h5
    # python /home/cy/nuist-lab/cfs-code-lab/c00-utils/h5_inspect.py /home/cy/datasets/CCGM/UTKFace_64x64.h5
    # python /home/cy/nuist-lab/cfs-code-lab/c00-utils/h5_inspect.py /home/cy/datasets/CCGM/ShapeNet_function_v1_16_64x64_900.h5
    # python /home/cy/nuist-lab/cfs-code-lab/c00-utils/h5_inspect.py /home/cy/datasets/CCGM/raw100_new_128x128.h5
    #
    # # 限制深度为2层
    # python h5_inspect.py your_file.h5 -d 2
    #
    # # 打印属性（最多每个对象 10 个键）
    # python h5_inspect.py your_file.h5 -a --attr-max 10
    #
    # # 输出 JSON（结构化）
    # python h5_inspect.py your_file.h5 -j > structure.json
    #
    # # 显示数据集的少量采样（最多 8 个元素）
    # python h5_inspect.py your_file.h5 --sample 8