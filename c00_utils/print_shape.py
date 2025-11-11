# -*- coding: utf-8 -*-
# @Time    : 2025/4/8 15:02
# @Author  : CFuShn
# @Comments: 
# @Software: PyCharm

from torchinfo import summary

if __name__ == "__main__":
    import os, sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from c05_basic_model.vgg import vgg8

    """
    'VGG8': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512,
              'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512,
              512, 512, 512, 'M'],
    """

    summary(model=vgg8(), input_size=(4, 3, 64, 64))  # depth默认=3
