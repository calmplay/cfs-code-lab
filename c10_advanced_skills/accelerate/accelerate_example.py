# -*- coding: utf-8 -*-
# @Time    : 2025/4/15 22:10
# @Author  : CFuShn
# @Comments: 
# @Software: PyCharm

"""
https://huggingface.co/docs/accelerate/basic_tutorials/migration
大家后面实验可以尝试用 accelerate 包来训练模型，
利用其进行混合精度计算，可以极大减小显存占用，减少训练成本和时间。
"""
from accelerate import Accelerator

accelerator = Accelerator()
# The Accelerator also knows which device to move your PyTorch objects to,
# so it is recommended to let Accelerate handle this for you.
# device = "cuda"/"mps"
device = accelerator.device

print(device)


##
mixed_precision_type = 'fp16'
use_amp = True
accelerator = Accelerator(mixed_precision = mixed_precision_type if use_amp else "no")
from accelerate.utils import set_seed
set_seed(1234)
## prepare model, dataloader, optimizer with accelerator
# self.netG, self.netD, self.optG, self.optD = self.accelerator.prepare(self.netG, self.netD,
#                                                                       self.optG, self.optD)


# 使用多卡并行命令
# accelerate launch {script_name.py} --arg1 --arg2 ...