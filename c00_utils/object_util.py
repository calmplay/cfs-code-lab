# -*- coding: utf-8 -*-
# @Time    : 2025/6/12 12:07
# @Author  : CFuShn
# @Comments: 
# @Software: PyCharm

import inspect
from typing import List, Type, TypeVar

T = TypeVar("T")

__all__ = ["dict_to_object","list_to_objects"]  # 使得调用方 import * 只导入指定项

def dict_to_object(cls: Type[T], data: dict, field_map: dict = None) -> T:
    """
    将单个 dict 转为对象实例（支持字段重命名）
    :param cls: 目标类
    :param data: 原始字典
    :param field_map: 可选映射，如 {"path": "dataset_path"}
    :return: 类实例
    """
    if hasattr(cls, "from_dict") and callable(getattr(cls, "from_dict")):
        return cls.from_dict(data)

    field_map = field_map or {}
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.keys())[1:]  # 跳过 self

    kwargs = {}
    for param in params:
        dict_key = next((k for k, v in field_map.items() if v == param), param)
        if dict_key in data:
            kwargs[param] = data[dict_key]

    return cls(**kwargs)


def list_to_objects(cls: Type[T], data_list: List[dict], field_map: dict = None) -> List[T]:
    """
    将 List[dict] 转为 List[cls]
    :param cls: 目标类
    :param data_list: 原始数据列表
    :param field_map: 可选字段映射
    :return: List[cls]
    """
    return [dict_to_object(cls, d, field_map) for d in data_list]
