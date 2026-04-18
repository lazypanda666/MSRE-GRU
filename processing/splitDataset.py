#!/usr/bin/env python
# coding=utf-8
"""
@Description   
@Author        lazypanda666
@Date          2026-04-15 20:18:51
"""

import os
import numpy as np
from const import cfg
from utils.log import logger
from utils.wrapper import calTimes


def buildSequences(data: list, seq_len: int, step: int = 1) -> np.array:
    """
    @description 构建时间序列数据集
    @param {list} data 输入的一维时间序列数据
    @param {int} seq_len  输入的一维时间序列数据
    @param {int} step  输滑动窗口的步长
    @return {np.array}
    """    
    seqs = []
    for i in range(0, len(data) - seq_len + 1, step):
        seqs.append(data[i:i + seq_len])
    
    return np.array(seqs, dtype=np.float32)


@calTimes(logger, "序列级数据集划分完成")
def splitTheDataset(window_dir: str = cfg.WINDOW["file_path"],
                    entropy_dir: str = cfg.DISTRIBUTION["file_path"],
                    output_dir: str = cfg.DATA["file_path"],
                    seq_len: int = cfg.MODEL["seq_len"] ) -> None:
    """
    @description 划分数据集
    @param {str} window_dir window文件目录
    @param {str} entropy_dir entropy文件目录
    @param {str} output_dir 划分数据集存储文件目录
    @param {int} seq_len 时间窗口长度
    @return {None}
    """    

    X_all, E_all, DE_all, y_all = [], [], [], []

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(entropy_dir):

        if not file.startswith("sampled_") or not file.endswith("_e.npy"):
            continue

        base_prefix = file.replace("_e.npy", "")

        # e(t)
        e_path = os.path.join(entropy_dir, base_prefix + "_e.npy")
        de_path = os.path.join(entropy_dir, base_prefix + "_de.npy")

        if not os.path.exists(de_path):
            continue

        e_data = np.load(e_path).astype(np.float32)     
        de_data = np.load(de_path).astype(np.float32)   

        # x(t)
        x_path_npy = os.path.join(window_dir, base_prefix + "_X.npy")

        if not os.path.exists(x_path_npy):
            continue

        x_data = np.load(x_path_npy).astype(np.float32)  

        # 对齐长度
        min_len = min(len(x_data), len(e_data), len(de_data))

        x_data = x_data[:min_len]
        e_data = e_data[:min_len]
        de_data = de_data[:min_len]

        # 序列级标准化
        x_data = (x_data - np.mean(x_data, axis=0)) / (np.std(x_data, axis=0) + 1e-6)
        e_data = (e_data - np.mean(e_data, axis=0)) / (np.std(e_data, axis=0) + 1e-6)
        de_data = (de_data - np.mean(de_data, axis=0)) / (np.std(de_data, axis=0) + 1e-6)

        # label
        label = int(file.split("_")[1])

        # 构建序列
        X_seq = buildSequences(x_data, seq_len)
        E_seq = buildSequences(e_data, seq_len)
        DE_seq = buildSequences(de_data, seq_len)

        if len(X_seq) == 0:
            continue

        y_seq = np.full((X_seq.shape[0],), label, dtype=np.int64)

        X_all.append(X_seq)
        E_all.append(E_seq)
        DE_all.append(DE_seq)
        y_all.append(y_seq)


    # 拼接
    X_all = np.concatenate(X_all, axis=0)
    E_all = np.concatenate(E_all, axis=0)
    DE_all = np.concatenate(DE_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)


    # shuffle
    idx = np.random.permutation(len(X_all))
    X_all = X_all[idx]
    E_all = E_all[idx]
    DE_all = DE_all[idx]
    y_all = y_all[idx]


    # split
    split_idx = int(0.7 * len(X_all))

    np.save(os.path.join(output_dir, "X_train.npy"), X_all[:split_idx])
    np.save(os.path.join(output_dir, "E_train.npy"), E_all[:split_idx])
    np.save(os.path.join(output_dir, "DE_train.npy"), DE_all[:split_idx])
    np.save(os.path.join(output_dir, "y_train.npy"), y_all[:split_idx])

    np.save(os.path.join(output_dir, "X_test.npy"), X_all[split_idx:])
    np.save(os.path.join(output_dir, "E_test.npy"), E_all[split_idx:])
    np.save(os.path.join(output_dir, "DE_test.npy"), DE_all[split_idx:])
    np.save(os.path.join(output_dir, "y_test.npy"), y_all[split_idx:])

    print(f"[Dataset] X:{X_all.shape}, E:{E_all.shape}, DE:{DE_all.shape}, y:{y_all.shape}")