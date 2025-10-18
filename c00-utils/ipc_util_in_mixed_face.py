# -*- coding: utf-8 -*-
# @Time    : 2025/2/12 13:50
# @Author  : cfushn
# @Comments:
# @Software: PyCharm

import signal

# 两个自定义信号量, 供用户处理事件, 通过SIGUSR1 和 SIGUSR2 信号操作其值
__s1 = 0  # 对应SIGUSR1 = 10
__s2 = 0  # 对应SIGUSR2 = 12


def _signal_handler(signum, frame):
    global __s1, __s2
    if signum == signal.SIGUSR1:
        __s1 = 1 - __s1  # 在0,1之间switch
        print(f"\n================= Received signal {signum}, set s1 = {__s1} ==================")
    elif signum == signal.SIGUSR2:
        __s2 = 1 - __s2  # 在0,1之间switch
        print(f"\n================= Received signal {signum}, set s2 = {__s2} ==================")


def register_signal_handler():
    """注册用户自定义signal_handler,在接收信号时触发触发对应信号量的自定义操作"""
    signal.signal(signal.SIGUSR1, _signal_handler)
    signal.signal(signal.SIGUSR2, _signal_handler)


def get_s1():
    """对应信号:SIGUSR1 = 10"""
    return __s1


def get_s2():
    """对应信号:SIGUSR2 = 12"""
    return __s2


def switch_s1(num):
    """对应信号:SIGUSR1 = 10"""
    assert num == 0 or num == 1
    global __s1
    if __s1 != num:
        __s1 = num
        print(f"=================== use func to switch s1 to {__s1} ====================")


def switch_s2(num):
    """对应信号:SIGUSR2 = 12"""
    assert num == 0 or num == 1
    global __s2
    if __s2 != num:
        __s2 = num
        print(f"=================== use func to switch s2 to {__s2} ====================")


# ----------------------- accelerate相关进程间通信 -----------------------
from config import ctx

import torch
import torch.distributed as dist

accelerator = ctx.accelerator
device = accelerator.device

# 进程间通信容器,它们的值通过广播发送与接收
_ipc_flags = [torch.tensor([0], device=device) for _ in range(100)]


def broadcast(flag_index, value=0) -> int:
    if not dist.is_initialized():
        return value  # 单进程下直接返回默认值

    """注意,广播要注意死锁问题,broadcast必须生产消费一致!"""
    if accelerator.is_main_process:  # 仅主进程可设置flag值
        _ipc_flags[flag_index].fill_(value)
    # 主进程(src=0)广播该tensor值给所有进程; 其他进程接收这个值
    dist.broadcast(_ipc_flags[flag_index], src=0)
    return _ipc_flags[flag_index].item()


if __name__ == "__main__":

    import sys
    import time

    # 必须先注册, 信号操作才会生效
    register_signal_handler()

    for i in range(50):
        get_s2() > 0 and sys.exit("SIGUSR2 is active! This process exit!")
        for j in range(10):
            time.sleep(1)
            get_s1() and print("SIGUSR1 is active!")
            print(i, j)
