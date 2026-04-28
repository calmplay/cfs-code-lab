# -*- coding: utf-8 -*-
# @Time    : 2026/4/27 17:51
# @Author  : CFuShn
# @Comments: H5 格式的 OmniFace 和 OmniShape 数据集类

"""
H5 格式的 OmniFace 和 OmniShape 数据集类
==============================================

功能:
1. 统一支持 OmniFace 和 OmniShape 两种数据集
2. 对于 H5 格式，图像只处理 RGB 数组格式
3. 提供标准的 PyTorch Dataset 接口
4. 提供快速生成 DataLoader 的方法
5. 提供单独的元数据查询接口（仅 OmniShape）

使用示例:
```python
from h5_omni_dataset import H5OmniDataset

# 创建数据集
dataset = H5OmniDataset(
    path="/home/data/OmniFace_64x64_20260421.h5",
    size=64,
    datasource="OmniFace"
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

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class H5OmniDataset(Dataset):
    """
    H5 格式的 OmniFace 和 OmniShape 数据集类
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
        split: Optional[str] = None,
        transform: Optional[transforms.Compose] = None
    ):
        """
        初始化 H5 数据集
        
        Args:
            path: H5 文件路径
            size: 图像尺寸（默认使用文件中的尺寸）
            datasource: 数据集类型，"OmniFace" 或 "OmniShape"
            split: 数据集分割，"train" 或 "val"，None 表示全部
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
            self.bytes_fields = {"id", "origin", "prompt"}
            self.skip_fields = {"train_indices", "val_indices"}
        elif self.datasource == "OmniShape":
            self.fields_config = self.OMNISHAPE_FIELDS
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
        
        # 总样本数
        self._len = len(self.indices)
        
        # 默认变换
        if self.transform is None:
            self.transform = self._get_default_transform()
    
    def _load_meta_data(self):
        """加载 OmniShape 元数据"""
        if "meta" not in self.h5_file:
            return
            
        meta_group = self.h5_file["meta"]
        
        # 读取所有字段并检查长度一致性
        field_lengths = {}
        for field in self.OMNISHAPE_META_FIELDS.keys():
            if field in meta_group:
                vals = meta_group[field][:]
                field_lengths[field] = len(vals)
        
        # 使用最小长度
        min_length = min(field_lengths.values()) if field_lengths else 0
        
        if min_length > 0:
            # 读取所有元数据字段
            for field in self.OMNISHAPE_META_FIELDS.keys():
                if field in meta_group:
                    vals = meta_group[field][:min_length]
                    if field in {"model_id", "class_id", "class100_id"}:
                        self.meta_data[field] = [self._bytes_to_str(v) for v in vals]
                    elif field in {"model_name", "class_name", "class100_name"}:
                        self.meta_data[field] = [
                            self._bytes_to_str(v) if isinstance(v, (bytes, np.bytes_)) else str(v)
                            for v in vals
                        ]
                    elif self.OMNISHAPE_META_FIELDS[field]["type"].startswith("sequence_"):
                        self.meta_data[field] = [v.tolist() for v in vals]
                    else:
                        self.meta_data[field] = vals.tolist()
            
            # 建立 model_id 到元数据的映射
            if "model_id" in self.meta_data:
                for i, model_id in enumerate(self.meta_data["model_id"]):
                    self.model_id_to_meta[model_id] = {
                        k: v[i] for k, v in self.meta_data.items() if k != "model_id"
                    }
    
    def _get_valid_indices(self):
        """获取有效索引"""
        if self.datasource == "OmniFace":
            if self.split is not None:
                if self.split == "train" and "train_indices" in self.h5_file:
                    return self.h5_file["train_indices"][:]
                elif self.split == "val" and "val_indices" in self.h5_file:
                    return self.h5_file["val_indices"][:]
        
        # 默认返回全部索引
        n_total = self.h5_file["images"].shape[0]
        return np.arange(n_total)
    
    def _bytes_to_str(self, val):
        """将字节转换为字符串"""
        if isinstance(val, (bytes, np.bytes_)):
            return val.decode("utf-8", errors="replace")
        return str(val)
    
    def _get_default_transform(self):
        """获取默认的图像变换"""
        transform_list = [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        
        # 如果指定了尺寸，添加 resize
        if self.size is not None:
            transform_list.insert(1, transforms.Resize((self.size, self.size)))
        
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
        actual_idx = self.indices[idx]
        
        # 读取图像（仅支持 CHW RGB 数组）
        img_chw = self.h5_file["images"][actual_idx]
        
        # 转换为 HWC 格式用于变换
        img_hwc = np.transpose(img_chw, (1, 2, 0))
        
        # 应用图像变换
        if self.transform is not None:
            img_tensor = self.transform(img_hwc)
        else:
            img_tensor = torch.from_numpy(img_hwc).permute(2, 0, 1)
        
        # 构建结果字典
        item = {"image": img_tensor}
        
        # 读取其他字段
        for field in self.fields_config.keys():
            if field == "image" or field in self.skip_fields:
                continue
                
            if field in self.h5_file:
                val = self.h5_file[field][actual_idx]
                
                if field in self.bytes_fields:
                    item[field] = self._bytes_to_str(val)
                elif self.fields_config[field]["type"].startswith("sequence_"):
                    if isinstance(val, np.ndarray):
                        item[field] = val.tolist()
                    else:
                        item[field] = val
                else:
                    if isinstance(val, np.generic):
                        item[field] = val.item()
                    else:
                        item[field] = val
        
        return item
    
    # ============ 元数据查询接口（仅 OmniShape）============
    def get_meta_by_index(self, idx: int) -> Optional[Dict]:
        """
        根据下标获取模型元数据
        
        Args:
            idx: 元数据索引
            
        Returns:
            元数据字典，如果不存在返回 None
        """
        if self.datasource != "OmniShape" or not self.meta_data:
            return None
            
        if idx < 0 or idx >= len(self.meta_data.get("model_id", [])):
            return None
            
        return {
            k: v[idx] for k, v in self.meta_data.items()
        }
    
    def get_meta_by_model_id(self, model_id: str) -> Optional[Dict]:
        """
        根据 model_id 获取模型元数据
        
        Args:
            model_id: 模型 ID
            
        Returns:
            元数据字典，如果不存在返回 None
        """
        if self.datasource != "OmniShape" or not self.model_id_to_meta:
            return None
            
        return self.model_id_to_meta.get(model_id)
    
    def get_meta_by_class_id(self, class_id: str) -> List[Dict]:
        """
        根据 class_id 获取所有匹配的模型元数据
        
        Args:
            class_id: 类别 ID
            
        Returns:
            元数据字典列表
        """
        if self.datasource != "OmniShape" or not self.meta_data:
            return []
            
        results = []
        class_ids = self.meta_data.get("class_id", [])
        for i, cid in enumerate(class_ids):
            if cid == class_id:
                results.append({
                    k: v[i] for k, v in self.meta_data.items()
                })
        return results
    
    def get_meta_by_class100_id(self, class100_id: str) -> List[Dict]:
        """
        根据 class100_id 获取所有匹配的模型元数据
        
        Args:
            class100_id: 粗粒度类别 ID
            
        Returns:
            元数据字典列表
        """
        if self.datasource != "OmniShape" or not self.meta_data:
            return []
            
        results = []
        class100_ids = self.meta_data.get("class100_id", [])
        for i, cid in enumerate(class100_ids):
            if cid == class100_id:
                results.append({
                    k: v[i] for k, v in self.meta_data.items()
                })
        return results
    
    def get_meta_keys(self) -> List[str]:
        """
        获取所有元数据字段名
        
        Returns:
            字段名列表
        """
        if not self.meta_data:
            return []
        return list(self.meta_data.keys())
    
    def get_num_meta_entries(self) -> int:
        """
        获取元数据条目数量
        
        Returns:
            条目数量
        """
        if not self.meta_data:
            return 0
        return len(next(iter(self.meta_data.values())))
    
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
    
    def close(self):
        """关闭 H5 文件"""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
