# -*- coding: utf-8 -*-
# @Time    : 2025/2/18 09:33
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

from datetime import datetime

# 获取当前日期和时间
now = datetime.now()

# 格式化输出为 "YYYYMMDDHH" 格式
formatted_date = now.strftime("%Y%m%d%H")
print(formatted_date)