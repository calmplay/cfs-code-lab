# -*- coding: utf-8 -*-
# @Time    : 2025/2/12 13:36
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import torch

device = ""
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # macOS m1 的 mps ≈ NVIDIA 1050Ti
else:
    device = "cpu"
