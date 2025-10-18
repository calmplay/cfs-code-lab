# -*- coding: utf-8 -*-
# @Time    : 2025/2/12 13:36
# @Author  : cfushn
# @Comments:
# @Software: PyCharm

import os
import re

def os_remove_by_regex(file_path_pattern: str):
    """仅支持文件名的正则"""
    # 拆分为目录和文件模式部分
    dir_path, file_pattern = os.path.split(file_path_pattern)
    file_pattern = r"" + file_pattern
    print(file_path_pattern, dir_path, file_pattern)
    dir_path = dir_path or "."  # 如果目录为空，则默认为当前目录

    for filename in os.listdir(dir_path):
        if re.match(file_pattern, filename):  # 匹配正则表达式
            file_path = os.path.join(dir_path, filename)  # 构造完整路径
            os.remove(file_path)  # 删除文件
            print(f"已删除文件: {file_path}")
