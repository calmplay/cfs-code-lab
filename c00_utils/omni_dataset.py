# -*- coding: utf-8 -*-
# @Time    : 2026/4/28 20:28
# @Author  : CFuShn
# @Comments: 统一的 OmniFace 和 OmniShape 数据集接口

"""
统一的 OmniFace 和 OmniShape 数据集接口
========================================

功能:
1. 统一支持 OmniFace 和 OmniShape 两种数据集
2. 自动检测数据格式（H5 或 HF/parquet）
3. 提供标准的 PyTorch Dataset 接口
4. 提供快速生成 DataLoader 的方法
5. 提供单独的元数据查询接口（仅 OmniShape）
6. 支持 train/val/test 三种 split

使用示例:
from omni_dataset import OmniDataset

# 自动检测格式创建数据集（必须指定 split）
train_dataset = OmniDataset(
    path="/home/data/HF/OmniFace512",
    datasource="OmniShape",
    split="train")

# 通过 dataset 实例获取 DataLoader
train_loader = train_dataset.get_dataloader(
    batch_size=128,
    num_workers=8,
    shuffle=True)

# 使用数据
for batch in train_loader:
    images = batch["image"]  # 图像
    model_id = batch["model_id"]  # 模型 ID
    # ... 训练逻辑
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from datasets import load_dataset
    from PIL import Image
    import io

    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False

# OmniFace 字段定义
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

# OmniShape 字段定义
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

# OmniShape 元数据字段定义
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


def OmniDataset(
        path: str,
        size: Optional[int] = None,
        datasource: str = "OmniShape",
        split: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        format: Optional[str] = None
):
    """
    统一的数据集创建接口，自动检测数据格式

    Args:
        path: 数据路径（H5 文件或 HF 数据集目录）
        size: 图像尺寸
        datasource: 数据集类型，"OmniFace" 或 "OmniShape"
        split: 数据集分割，**必须指定** "train"、"val" 或 "test"
        transform: 图像变换
        format: 数据格式，"h5" 或 "hf"，默认自动检测

    Returns:
        H5OmniDataset 或 HFOmniDataset 实例

    Raises:
        ValueError: 如果未指定 split 参数
    """
    if split is None:
        raise ValueError("必须指定 split 参数（'train', 'val' 或 'test'）")

    # 自动检测格式
    if format is None:
        format = _detect_format(path)

    if format == "h5":
        return _H5OmniDataset(
            path=path,
            size=size,
            datasource=datasource,
            split=split,
            transform=transform
        )
    elif format == "hf":
        return _HFOmniDataset(
            path=path,
            size=size,
            datasource=datasource,
            split=split,
            transform=transform
        )
    else:
        raise ValueError(f"不支持的数据格式: {format}")


def _detect_format(path: str) -> str:
    """
    自动检测数据格式

    Args:
        path: 数据路径

    Returns:
        "h5" 或 "hf"
    """
    if path.endswith(".h5") or path.endswith(".hdf5"):
        return "h5"

    # 检查是否是 HF 数据集目录
    if os.path.isdir(path):
        data_dir = os.path.join(path, "data")
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if f.endswith(".parquet"):
                    return "hf"

    # 默认返回 h5
    return "h5"


class _H5OmniDataset(Dataset):
    """H5 格式的 OmniFace 和 OmniShape 数据集类"""

    def __init__(
            self,
            path: str,
            size: Optional[int] = None,
            datasource: str = "OmniShape",
            split: Optional[str] = None,
            transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            path: H5 文件路径
            size: 图像尺寸
            datasource: 数据集类型，"OmniFace" 或 "OmniShape"
            split: 数据集分割，"train"、"val" 或 "test"
            transform: 图像变换
        """
        if not HAS_H5PY:
            raise ImportError("需要安装 h5py 库来使用 H5OmniDataset")

        self.path = path
        self.size = size
        self.datasource = datasource
        self.split = split
        self.transform = transform

        # 根据数据集类型选择字段
        if self.datasource == "OmniFace":
            self.fields_config = OMNIFACE_FIELDS
            self.bytes_fields = {"id", "origin", "prompt"}
            self.skip_fields = {"train_indices", "val_indices"}
        elif self.datasource == "OmniShape":
            self.fields_config = OMNISHAPE_FIELDS
            self.bytes_fields = {"model_id"}
            self.skip_fields = set()
        else:
            raise ValueError(f"不支持的数据集类型: {self.datasource}")

        # 打开 H5 文件
        self.h5_file = h5py.File(self.path, 'r')

        # 加载元数据（仅 OmniShape）
        self.meta_data = {}
        self.model_id_to_meta = {}
        if self.datasource == "OmniShape":
            self._load_meta_data()

        # 获取有效索引
        self.indices = self._get_valid_indices()
        self._len = len(self.indices)

        # 默认变换
        if self.transform is None:
            self.transform = self._get_default_transform(h5_format=True)

    def _load_meta_data(self):
        """加载 OmniShape 元数据"""
        if "meta" not in self.h5_file:
            return

        meta_group = self.h5_file["meta"]
        field_lengths = {}
        for field in OMNISHAPE_META_FIELDS.keys():
            if field in meta_group:
                vals = meta_group[field][:]
                field_lengths[field] = len(vals)

        min_length = min(field_lengths.values()) if field_lengths else 0

        if min_length > 0:
            for field in OMNISHAPE_META_FIELDS.keys():
                if field in meta_group:
                    vals = meta_group[field][:min_length]
                    if field in {"model_id", "class_id", "class100_id"}:
                        self.meta_data[field] = [self._bytes_to_str(v) for v in vals]
                    elif field in {"model_name", "class_name", "class100_name"}:
                        self.meta_data[field] = [
                            self._bytes_to_str(v) if isinstance(v, (bytes, np.bytes_)) else str(v)
                            for v in vals
                        ]
                    elif OMNISHAPE_META_FIELDS[field]["type"].startswith("sequence_"):
                        self.meta_data[field] = [v.tolist() for v in vals]
                    else:
                        self.meta_data[field] = vals.tolist()

            if "model_id" in self.meta_data:
                for i, model_id in enumerate(self.meta_data["model_id"]):
                    self.model_id_to_meta[model_id] = {
                        k: v[i] for k, v in self.meta_data.items() if k != "model_id"
                    }

    def _get_valid_indices(self):
        """获取有效索引"""
        if self.split is None:
            raise ValueError("必须指定 split 参数（'train', 'val' 或 'test'）")

        if self.datasource == "OmniFace":
            if self.split == "train":
                if "train_indices" not in self.h5_file:
                    raise ValueError(f"H5 文件中不存在 'train_indices'，无法获取 train split")
                return self.h5_file["train_indices"][:]
            elif self.split == "val":
                if "val_indices" not in self.h5_file:
                    raise ValueError(f"H5 文件中不存在 'val_indices'，无法获取 val split")
                val_all = self.h5_file["val_indices"][:]
                split_idx = len(val_all) // 3
                return val_all[:split_idx]
            elif self.split == "test":
                if "val_indices" not in self.h5_file:
                    raise ValueError(f"H5 文件中不存在 'val_indices'，无法获取 test split")
                val_all = self.h5_file["val_indices"][:]
                split_idx = len(val_all) // 3
                return val_all[split_idx:]
            else:
                raise ValueError(f"不支持的 split: {self.split}，必须是 'train', 'val' 或 'test'")

        elif self.datasource == "OmniShape":
            n_total = self.h5_file["images"].shape[0]
            return np.arange(n_total)

        raise ValueError(f"不支持的数据集类型: {self.datasource}")

    @staticmethod
    def _bytes_to_str(val):
        """将字节转换为字符串"""
        if isinstance(val, (bytes, np.bytes_)):
            return val.decode("utf-8", errors="replace")
        return str(val)

    def _get_default_transform(self, h5_format=True):
        """获取默认的图像变换"""
        transform_list = []
        if h5_format:
            transform_list.append(transforms.ToPILImage())
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        if self.size is not None:
            idx = 1 if h5_format else 0
            transform_list.insert(idx, transforms.Resize((self.size, self.size)))

        return transforms.Compose(transform_list)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        """获取单个样本"""
        actual_idx = self.indices[idx]

        # 读取图像（CHW RGB 数组）
        img_chw = self.h5_file["images"][actual_idx]
        img_hwc = np.transpose(img_chw, (1, 2, 0))

        if self.transform is not None:
            img_tensor = self.transform(img_hwc)
        else:
            img_tensor = torch.from_numpy(img_hwc).permute(2, 0, 1)

        item = {"image": img_tensor}

        for field in self.fields_config.keys():
            if field == "image" or field in self.skip_fields:
                continue

            if field in self.h5_file:
                val = self.h5_file[field][actual_idx]

                if field in self.bytes_fields:
                    item[field] = self._bytes_to_str(val)
                elif self.fields_config[field]["type"].startswith("sequence_"):
                    item[field] = val.tolist() if isinstance(val, np.ndarray) else val
                else:
                    item[field] = val.item() if isinstance(val, np.generic) else val

        return item

    # ============ 元数据查询接口 ============
    def get_meta_by_index(self, idx: int) -> Optional[Dict]:
        if self.datasource != "OmniShape" or not self.meta_data:
            return None
        if idx < 0 or idx >= len(self.meta_data.get("model_id", [])):
            return None
        return {k: v[idx] for k, v in self.meta_data.items()}

    def get_meta_by_model_id(self, model_id: str) -> Optional[Dict]:
        if self.datasource != "OmniShape" or not self.model_id_to_meta:
            return None
        return self.model_id_to_meta.get(model_id)

    def get_meta_by_class_id(self, class_id: str) -> List[Dict]:
        if self.datasource != "OmniShape" or not self.meta_data:
            return []
        results = []
        class_ids = self.meta_data.get("class_id", [])
        for i, cid in enumerate(class_ids):
            if cid == class_id:
                results.append({k: v[i] for k, v in self.meta_data.items()})
        return results

    def get_meta_by_class100_id(self, class100_id: str) -> List[Dict]:
        if self.datasource != "OmniShape" or not self.meta_data:
            return []
        results = []
        class100_ids = self.meta_data.get("class100_id", [])
        for i, cid in enumerate(class100_ids):
            if cid == class100_id:
                results.append({k: v[i] for k, v in self.meta_data.items()})
        return results

    def get_meta_keys(self) -> List[str]:
        return list(self.meta_data.keys()) if self.meta_data else []

    def get_num_meta_entries(self) -> int:
        return len(next(iter(self.meta_data.values()))) if self.meta_data else 0

    def get_dataloader(
            self,
            batch_size: int = 128,
            shuffle: bool = True,
            num_workers: int = 8,
            pin_memory: bool = True,
            drop_last: bool = False,
            **kwargs
    ):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )

    def close(self):
        """关闭 H5 文件"""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class _HFOmniDataset(Dataset):
    """HuggingFace (parquet) 格式的 OmniFace 和 OmniShape 数据集类"""

    def __init__(
            self,
            path: str,
            size: Optional[int] = None,
            datasource: str = "OmniShape",
            split: Optional[str] = None,
            transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            path: 数据集路径
            size: 图像尺寸
            datasource: 数据集类型，"OmniFace" 或 "OmniShape"
            split: 数据集分割，**必须指定** "train"、"val" 或 "test"
            transform: 图像变换
        """
        if not HAS_HF_DATASETS:
            raise ImportError("需要安装 datasets 库来使用 HFOmniDataset")

        self.path = path
        self.size = size
        self.datasource = datasource
        self.split = split
        self.transform = transform

        # 根据数据集类型选择字段
        if self.datasource == "OmniFace":
            self.fields_config = OMNIFACE_FIELDS
        elif self.datasource == "OmniShape":
            self.fields_config = OMNISHAPE_FIELDS
        else:
            raise ValueError(f"不支持的数据集类型: {self.datasource}")

        # 加载数据集
        self.dataset = self._load_dataset()
        self._len = len(self.dataset)

        # 加载元数据（仅 OmniShape）
        self.meta_data = {}
        self.model_id_to_meta = {}
        if self.datasource == "OmniShape":
            self._load_meta_data()

        # 默认变换
        if self.transform is None:
            self.transform = self._get_default_transform()

    def _load_dataset(self):
        """加载 parquet 数据集"""
        data_dir = os.path.join(self.path, "data")

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        if self.split is None:
            raise ValueError("必须指定 split 参数（'train', 'val' 或 'test'）")

        split_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith(f"{self.split}-") and f.endswith(".parquet")
        ]

        if not split_files:
            raise ValueError(
                f"未找到 split '{self.split}' 对应的 parquet 文件。"
                f"请检查数据集目录 {data_dir} 中是否存在以 '{self.split}-' 开头的文件。"
            )

        return load_dataset("parquet", data_files=split_files, split="train")

    def _load_meta_data(self):
        """加载 OmniShape 元数据"""
        meta_path = os.path.join(self.path, "meta.parquet")

        if not os.path.exists(meta_path):
            return

        meta_ds = load_dataset("parquet", data_files=meta_path, split="train")

        self.meta_data = {}
        for field in OMNISHAPE_META_FIELDS.keys():
            if field in meta_ds.features:
                self.meta_data[field] = meta_ds[field]

        if "model_id" in self.meta_data:
            for i, model_id in enumerate(self.meta_data["model_id"]):
                self.model_id_to_meta[model_id] = {
                    k: v[i] for k, v in self.meta_data.items() if k != "model_id"
                }

    @staticmethod
    def _bytes_to_str(val):
        """将字节转换为字符串"""
        if isinstance(val, (bytes, np.bytes_)):
            return val.decode("utf-8", errors="replace")
        return str(val)

    def _get_default_transform(self):
        """获取默认的图像变换"""
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

        if self.size is not None:
            transform_list.insert(0, transforms.Resize((self.size, self.size)))

        return transforms.Compose(transform_list)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.dataset[idx]

        # 处理图像
        img = sample["image"]
        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"]))

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        item = {"image": img_tensor}

        for field in self.fields_config.keys():
            if field == "image":
                continue

            if field in sample:
                val = sample[field]
                if self.fields_config[field]["type"].startswith("sequence_"):
                    item[field] = val.tolist() if isinstance(val, np.ndarray) else val
                else:
                    item[field] = val

        return item

    # ============ 元数据查询接口 ============
    def get_meta_by_index(self, idx: int) -> Optional[Dict]:
        if self.datasource != "OmniShape" or not self.meta_data:
            return None
        if idx < 0 or idx >= len(self.meta_data.get("model_id", [])):
            return None
        return {k: v[idx] for k, v in self.meta_data.items()}

    def get_meta_by_model_id(self, model_id: str) -> Optional[Dict]:
        if self.datasource != "OmniShape" or not self.model_id_to_meta:
            return None
        return self.model_id_to_meta.get(model_id)

    def get_meta_by_class_id(self, class_id: str) -> List[Dict]:
        if self.datasource != "OmniShape" or not self.meta_data:
            return []
        results = []
        class_ids = self.meta_data.get("class_id", [])
        for i, cid in enumerate(class_ids):
            if cid == class_id:
                results.append({k: v[i] for k, v in self.meta_data.items()})
        return results

    def get_meta_by_class100_id(self, class100_id: str) -> List[Dict]:
        if self.datasource != "OmniShape" or not self.meta_data:
            return []
        results = []
        class100_ids = self.meta_data.get("class100_id", [])
        for i, cid in enumerate(class100_ids):
            if cid == class100_id:
                results.append({k: v[i] for k, v in self.meta_data.items()})
        return results

    def get_meta_keys(self) -> List[str]:
        return list(self.meta_data.keys()) if self.meta_data else []

    def get_num_meta_entries(self) -> int:
        return len(next(iter(self.meta_data.values()))) if self.meta_data else 0

    def get_dataloader(
            self,
            batch_size: int = 128,
            shuffle: bool = True,
            num_workers: int = 8,
            pin_memory: bool = True,
            drop_last: bool = False,
            **kwargs
    ):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )


__all__ = [
    "OmniDataset",
    "OMNIFACE_FIELDS",
    "OMNISHAPE_FIELDS",
    "OMNISHAPE_META_FIELDS",
]
