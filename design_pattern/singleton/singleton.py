# -*- coding: utf-8 -*-
# @Time    : 2025/2/20 22:08
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

class SingletonBase:
    _instances = {}  # 存储所有子类的单例实例

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]
