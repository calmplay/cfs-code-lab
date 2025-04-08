# -*- coding: utf-8 -*-
# @Time    : 2025/2/20 22:09
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

from design_pattern.singleton.singleton import SingletonBase


class King(SingletonBase):
    name = "cfushn"


if __name__ == '__main__':
    k1 = King()
    k2 = King()
    print(k1)
    print(k2)
    print(k1.name)
    print(k1 is k2)
