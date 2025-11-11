# -*- coding: utf-8 -*-
# @Time    : 2025/2/18 09:33
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import time
from datetime import datetime


def tttt():
    # 获取当前日期和时间
    now = datetime.now()

    # 格式化输出为 "YYYYMMDDHH" 格式
    formatted_date = now.strftime("%Y%m%d%H")
    print(formatted_date)


class TimerContext:
    def __init__(self, name: str = "---"):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        print(f"[{self.name}] 用时: {self.elapsed:.4f} 秒.")

    def seconds(self):
        return self.elapsed
