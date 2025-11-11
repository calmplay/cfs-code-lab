# -*- coding: utf-8 -*-
# @Time    : 2025/2/12 15:12
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import sys

import numpy as np
import torch


class SimpleProgressBar:
    """
    ProgressBar
    """

    def __init__(self, width=50):
        self.last_x = -1
        self.width = width

    def update(self, x):
        assert 0 <= x <= 100  # `x`: progress in percent ( between 0 and 100)
        if self.last_x == int(x): return
        self.last_x = int(x)
        pointer = int(self.width * (x / 100.0))
        sys.stdout.write('\r%d%% [%s]' % (int(x), '#' * pointer + '.' * (self.width - pointer)))
        sys.stdout.flush()
        if x == 100:
            print('')


class ImgsDataset(torch.utils.data.Dataset):
    """
    torch dataset from numpy array
    """

    def __init__(self, images, labels=None, normalize=False):
        super(ImgsDataset, self).__init__()

        self.images = images
        self.n_images = len(self.images)
        self.labels = labels
        if labels is not None and (len(self.images) != len(self.labels)):
            raise Exception('images (' + str(len(self.images)) + ') and labels (' + str(
                len(self.labels)) + ') do not have the same length!!!')
        self.normalize = normalize

    def __getitem__(self, index):

        image = self.images[index]

        if self.normalize:
            image = image / 255.0
            image = (image - 0.5) / 0.5

        if self.labels is not None:
            label = self.labels[index]
            return (image, label)
        else:
            return image

    def __len__(self):
        return self.n_images


# example
# images = np.random.randint(0, 255, (100, 3, 32, 32))  # 100 张 32x32 RGB 图像
# labels = np.random.randint(0, 10, 100)  # 100 个类别标签
#
# dataset = IMGs_dataset(images, labels, normalize=True)
# image, label = dataset[0]
# print(image.shape, label)  # 输出: (3, 32, 32) 0

def compute_entropy(labels, base=None):
    """
    计算类别熵
    """
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = np.e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


def predict_class_labels(net, images, batch_size=500, verbose=False, num_workers=0):
    """
    预测图像类别
    :param net: 分类网络
    :param images: 输入图像
    :param batch_size:
    :param verbose: 是否打印详细的输出信息/进度条
    :param num_workers:
    :return: predict_class_labels
    """
    net = net.cuda()
    net.eval()

    n = len(images)
    if batch_size > n:
        batch_size = n
    dataset_pred = ImgsDataset(images, normalize=False)
    dataloader_pred = torch.utils.data.DataLoader(dataset_pred, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)

    class_labels_pred = np.zeros(n + batch_size)
    with torch.no_grad():
        nimgs_got = 0
        if verbose:
            pb = SimpleProgressBar()
        for batch_idx, batch_images in enumerate(dataloader_pred):
            batch_images = batch_images.type(torch.float).cuda()
            batch_size_curr = len(batch_images)

            outputs, _ = net(batch_images)
            _, batch_class_labels_pred = torch.max(outputs.data, 1)
            class_labels_pred[nimgs_got:(
                        nimgs_got + batch_size_curr)] = batch_class_labels_pred.detach().cpu().numpy().reshape(
                -1)

            nimgs_got += batch_size_curr
            if verbose:
                pb.update((float(nimgs_got) / n) * 100)
    class_labels_pred = class_labels_pred[0:n]
    return class_labels_pred


if __name__ == "__main__":
    pb = SimpleProgressBar()
    import time

    for i in range(101):
        time.sleep(1)
        pb.update(i)
