# -*- coding: utf-8 -*-
# @Time    : 2025/2/12 13:48
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import os
import re

import torch

from c00_utils.log_util import cy_log


def save_checkpoint(model, optimizer, epoch, loss):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    folder = '/home/cy/workdir/cfushn_ldm/model_save'
    cy_log(f'save check point (epoch:{epoch})...')
    torch.save(state, os.path.join(folder, f"checkpoint_{epoch}.pth"))
    cy_log('save successfully.\n')


def load_checkpoint(unet, optimizer=None, checkpoint_path: str = None):
    """
    读取模型的 checkpoint 文件并加载权重、优化器状态及其他训练信息。

    Args:
        unet (torch.nn.Module): 需要加载权重的模型实例。
        optimizer (torch.optim.Optimizer, optional): 如果需要恢复优化器状态，请传入优化器实例。
        checkpoint_path (str): 默认None,会加载特定目录中最后一个checkpoint

    Returns:
        dict: 包含额外信息的字典，例如 epoch 和损失。
    """
    if unet is None:
        raise Exception("请初始化模型实例!")

    if checkpoint_path is None:
        folder = '/home/cy/workdir/cfushn_ldm/model_save'
        files = os.listdir(folder)
        matched_files = []
        for file in files:
            match = re.match("checkpoint_(\d+).pth", file)
            if match:
                # 提取文件名中的 epoch 值并保存
                epoch = int(match.group(1))  # 假设第一个捕获组是数字
                matched_files.append((epoch, file))
        # 如果没有匹配文件
        if not matched_files:
            raise FileNotFoundError(f"there has no matched models in '{folder}'")
        # 按 epoch 值排序并返回最后一个文件的完整路径
        matched_files.sort(key=lambda x: x[0])  # 按 epoch 升序排序
        latest_file = matched_files[-1][1]  # 获取最后一个文件名
        checkpoint_path = os.path.join(folder, latest_file)

    # 加载 checkpoint 文件
    check_point = torch.load(checkpoint_path)

    unet.load_state_dict(check_point["state_dict"])
    unet.train()

    # 如果提供了优化器实例，加载优化器状态
    optimizer is None or optimizer.load_state_dict(check_point["optimizer"])

    last_epoch = check_point["epoch"]
    loss = check_point["loss"]
    # 返回其他训练信息
    print(f"-------------load checkpoint: epoch:{last_epoch},loss:{loss}---------------")
    return unet, optimizer, last_epoch, loss

def load_ckpt_compatibly(
    model,
    checkpoint_or_path,
    key: str | None = None,
    map_location="cpu",
    strict: bool = True,
    verbose: bool = True,
):
    """
    Load checkpoint into a model in a DP/DDP-friendly way.

    Args:
        model: nn.Module. The target model to load weights into.
        checkpoint_or_path: str or dict.
            - If str: treated as checkpoint path, will call torch.load.
            - If dict: treated as an already-loaded checkpoint.
        key: str or None.
            - If not None, use checkpoint[key] as state_dict.
            - If None, try to auto-detect a state_dict-like entry.
        map_location: passed to torch.load when checkpoint_or_path is a path.
        strict: passed to model.load_state_dict.
        verbose: print debug information if True.

    Returns:
        (missing_keys, unexpected_keys) from load_state_dict.
    """
    # 1) load checkpoint
    if isinstance(checkpoint_or_path, str):
        ckpt = torch.load(checkpoint_or_path, map_location=map_location, weights_only=True)
    else:
        ckpt = checkpoint_or_path

    # 2) get raw state_dict
    if key is not None:
        if key not in ckpt:
            raise KeyError(f"Key '{key}' not found in checkpoint. Available keys: {list(ckpt.keys())}")
        state_dict = ckpt[key]
    else:
        # If ckpt itself looks like a state_dict (all values are tensors), use it directly
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
        else:
            # Try common field names
            candidate_keys = [
                "state_dict",
                "net_state_dict",
                "model_state_dict",
                "net_encoder_state_dict",
                "encoder_state_dict",
                "module",
                "model",
            ]
            found = None
            for k in candidate_keys:
                if k in ckpt:
                    found = k
                    break
            if found is None:
                raise ValueError(
                    f"Cannot find a state_dict in checkpoint. Tried keys: {candidate_keys}. "
                    f"Available keys: {list(ckpt.keys())}"
                )
            state_dict = ckpt[found]
            if verbose:
                print(f"[load_compatible_ckpt] Auto-detected state_dict key: '{found}'")

    # 3) handle 'module.' prefix (DP/DDP vs single-GPU)
    model_keys = list(model.state_dict().keys())
    loaded_keys = list(state_dict.keys())

    def _count_prefix(keys, prefix: str) -> int:
        return sum(k.startswith(prefix) for k in keys)

    model_has_module = _count_prefix(model_keys, "module.") > len(model_keys) * 0.5
    loaded_has_module = _count_prefix(loaded_keys, "module.") > len(loaded_keys) * 0.5

    new_state_dict = {}

    if loaded_has_module and not model_has_module:
        # Strip 'module.' from checkpoint keys
        if verbose:
            print("[load_compatible_ckpt] Stripping 'module.' prefix from checkpoint keys.")
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_k = k[len("module."):]
            else:
                new_k = k
            new_state_dict[new_k] = v

    elif model_has_module and not loaded_has_module:
        # Add 'module.' prefix to checkpoint keys
        if verbose:
            print("[load_compatible_ckpt] Adding 'module.' prefix to checkpoint keys.")
        for k, v in state_dict.items():
            new_k = "module." + k
            new_state_dict[new_k] = v

    else:
        # No prefix adjustment needed
        new_state_dict = state_dict
        if verbose:
            print("[load_compatible_ckpt] No 'module.' prefix adjustment needed.")

    # 4) actually load
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)

    if verbose:
        if missing_keys:
            print("[load_compatible_ckpt] Missing keys:", missing_keys)
        if unexpected_keys:
            print("[load_compatible_ckpt] Unexpected keys:", unexpected_keys)
        if not missing_keys and not unexpected_keys:
            print("[load_compatible_ckpt] Loaded successfully with strict =", strict)

    return missing_keys, unexpected_keys