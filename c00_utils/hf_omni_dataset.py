# -*- coding: utf-8 -*-
# @Time    : 2026/4/27 17:51
# @Author  : CFuShn
# @Comments: HuggingFace (parquet) 格式的 OmniFace 和 OmniShape 数据集类

"""
HuggingFace (parquet) 格式的 OmniFace 和 OmniShape 数据集类
============================================================

功能:
1. 统一支持 OmniFace 和 OmniShape 两种数据集
2. 对于 HF 格式，图像存储格式一致
3. 对于 OmniShape，支持将元数据信息解析到每个样本
4. 提供标准的 PyTorch Dataset 接口
5. 提供快速生成 DataLoader 的方法

使用示例:
```python
from hf_omni_dataset import HFOmniDataset

# 创建数据集
dataset = HFOmniDataset(
    path="/home/data/HF/OmniShape",
    size=128,
    datasource="OmniShape"
)

# 获取 DataLoader
dataloader = dataset.get_dataloader(
    batch_size=128,
    shuffle=True,
    num_workers=8
)
```
"""

import os
from typing import Dict, List, Optional, Union
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import io


class HFOmniDataset(Dataset):
    """
    HuggingFace (parquet) 格式的 OmniFace 和 OmniShape 数据集类
    """
    
    # 定义 OmniFace 字段
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
    
    # 定义 OmniShape 字段
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
    
    # OmniShape 元数据字段
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
    
    def __init__(
        self,
        path: str,
        size: Optional[int] = None,
        datasource: str = "OmniShape",
        split: Optional[str] = "train",
        transform: Optional[transforms.Compose] = None
    ):
        """
        初始化 HF 数据集
        
        Args:
            path: 数据集目录路径
            size: 图像尺寸（默认使用文件中的尺寸）
            datasource: 数据集类型，"OmniFace" 或 "OmniShape"
            split: 数据集分割，"train" 或 "val"，"train" 为默认
            transform: 图像变换
        """
        self.path = path
        self.size = size
        self.datasource = datasource
        self.split = split
        self.transform = transform
        
        # 根据数据集类型选择字段
        if self.datasource == "OmniFace":
            self.fields_config = self.OMNIFACE_FIELDS
        elif self.datasource == "OmniShape":
            self.fields_config = self.OMNISHAPE_FIELDS
        else:
            raise ValueError(f"不支持的数据集类型: {self.datasource}")
        
        # 加载数据集
        self.dataset = self._load_dataset()
        
        # 总样本数
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
        
        # 查找对应 split 的 parquet 文件
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        # 加载数据集
        if self.split is not None:
            # 查找对应 split 的文件
            split_files = [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.startswith(f"{self.split}-") and f.endswith(".parquet")
            ]
            
            if split_files:
                return load_dataset("parquet", data_files=split_files, split="train")
        
        # 加载所有数据
        all_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".parquet")
        ]
        
        if not all_files:
            raise FileNotFoundError(f"未找到 parquet 文件: {data_dir}")
        
        return load_dataset("parquet", data_files=all_files, split="train")
    
    def _load_meta_data(self):
        """加载 OmniShape 元数据"""
        meta_path = os.path.join(self.path, "meta.parquet")
        
        if not os.path.exists(meta_path):
            return
            
        # 加载元数据
        meta_ds = load_dataset("parquet", data_files=meta_path, split="train")
        
        # 读取所有元数据字段
        self.meta_data = {}
        for field in self.OMNISHAPE_META_FIELDS.keys():
            if field in meta_ds.features:
                self.meta_data[field] = meta_ds[field]
        
        # 建立 model_id 到元数据的映射
        if "model_id" in self.meta_data:
            for i, model_id in enumerate(self.meta_data["model_id"]):
                self.model_id_to_meta[model_id] = {
                    k: v[i] for k, v in self.meta_data.items() if k != "model_id"
                }
    
    def _bytes_to_str(self, val):
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
        
        # 如果指定了尺寸，添加 resize
        if self.size is not None:
            transform_list.insert(0, transforms.Resize((self.size, self.size)))
        
        return transforms.Compose(transform_list)
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含所有字段的字典
        """
        # 从 HF 数据集获取样本
        sample = self.dataset[idx]
        
        # 处理图像
        img = sample["image"]
        
        # 如果是字典格式（bytes形式），需要解码
        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"]))
        
        # 应用图像变换
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            # 默认转换为 tensor
            img_tensor = transforms.ToTensor()(img)
        
        # 构建结果字典
        item = {"image": img_tensor}
        
        # 复制其他字段
        for field in self.fields_config.keys():
            if field == "image":
                continue
                
            if field in sample:
                    val = sample[field]
                    
                    if self.fields_config[field]["type"].startswith("sequence_"):
                        if isinstance(val, np.ndarray):
                            item[field] = val.tolist()
                        else:
                            item[field] = val
                    else:
                        item[field] = val
        
        # 对于 OmniShape，添加元数据
        if self.datasource == "OmniShape":
            model_id = item.get("model_id")
            if model_id and model_id in self.model_id_to_meta:
                item.update(self.model_id_to_meta[model_id])
        
        return item
    
    def get_dataloader(
        self,
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        """
        快速生成 DataLoader
        
        Args:
            batch_size: 批次大小
            shuffle: 是否打乱
            num_workers: 工作进程数
            pin_memory: 是否固定内存
            drop_last: 是否丢弃最后一个不完整的批次
            **kwargs: 其他 DataLoader 参数
            
        Returns:
            PyTorch DataLoader
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )
