# -*- coding: utf-8 -*-
# @Time    : 2025/2/12 15:18
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt


def PlotLoss(loss, filename):
    """
    绘制训练损失曲线
    """
    x_axis = np.arange(start=1, stop=len(loss) + 1)
    plt.switch_backend('agg')  # 适用于无 GUI 环境
    mpl.style.use('seaborn')
    ax = plt.subplot(111)
    ax.plot(x_axis, np.array(loss))
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.legend()
    plt.savefig(filename)

if __name__ == "__main__":
    losses = [0.8, 0.6, 0.5, 0.4, 0.3]
    PlotLoss(losses, "loss_curve.png")
