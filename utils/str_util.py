# -*- coding: utf-8 -*-
# @Time    : 2025/2/12 13:50
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import re

def to_snake_case(text: str) -> str:
    """
    将str转化为用下划线连接的字符串
    """

    # strip()去除首尾空格,并将多个空格替换为单个空格
    cleaned_str = re.sub(r'\s+', ' ', text.strip())

    # 替换空格和特殊字符为下划线，并将字符串转换为小写
    snake_case = re.sub(r'[^a-zA-Z0-9]+', '_', cleaned_str).lower()

    # 确保没有多余的下划线（strip('_')去除首尾下划线）
    snake_case = re.sub(r'_+', '_', snake_case).strip('_')

    return snake_case
