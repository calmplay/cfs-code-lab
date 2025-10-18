# -*- coding: utf-8 -*-
# @Time    : 2025/6/9 14:24
# @Author  : CFuShn
# @Comments: 系统相关util
# @Software: PyCharm


import platform

def is_windows():
    return platform.system() == 'Windows'

def is_linux():
    return platform.system() == 'Linux'

def is_mac():
    return platform.system() == 'Darwin'

if __name__ == "__main__":
    print(is_windows())
    print(is_linux())
    print(is_mac())