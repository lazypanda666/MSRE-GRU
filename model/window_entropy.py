#!/usr/bin/env python
# coding=utf-8
"""
@Description   网络流量的窗口化，并计算RENYI_entropy
@Author        lazypanda666
@Date          2026-04-15 15:25:00
"""

import os
import csv
import numpy as np
from typing import List, Callable

import processing.featureExtract.feature as feature

from const import cfg
from utils.log import logger
from utils.wrapper import calTimes

# 特征编码
def feature_encode(pkt: dict) -> np.ndarray:
    def safe_float(x):
        if x is None:
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        return 0.0

    return np.array([safe_float(pkt[f]) for f in feature.FIELDNAMES], dtype=float)


# ψ_x（内容编码）
def psi_x(W: np.ndarray) -> np.ndarray:

    mean = np.mean(W, axis=0)
    std = np.std(W, axis=0)
    max_val = np.max(W, axis=0)
    return np.concatenate([mean, std, max_val])


# 计算Rényi熵
def renyi_entropy(p: np.ndarray, alpha: float) -> float:
    p_safe = p + 1e-12

    # Shannon 熵
    if alpha == 1.0:
        return -np.sum(p_safe * np.log(p_safe))

    # 数值稳定计算：log-sum-exp
    log_p = np.log(p_safe)
    a = alpha * log_p

    # 防止overflow
    a_max = np.max(a)
    sum_exp = np.sum(np.exp(a - a_max))

    return (1.0 / (1 - alpha)) * (np.log(sum_exp) + a_max)


# 窗口Rényi特征
def compute_window_renyi(W: np.ndarray, alphas: List[float], num_bins: int) -> np.ndarray:
    w, m = W.shape
    K = len(alphas)

    H = np.zeros((K, m))

    for j in range(m):
        feature = W[:, j]

        # 跳过常数特征
        if np.all(feature == feature[0]):
            continue

        hist, _ = np.histogram(feature, bins=num_bins)

        if np.sum(hist) == 0:
            continue

        p = hist / np.sum(hist)

        for k, alpha in enumerate(alphas):
            val = renyi_entropy(p, float(alpha))

            H[k, j] = val

    return H.flatten()



@calTimes(logger, "窗口内容 + Rényi特征构建完成")
def extractWindowFeaturesAndEntropy(
    input_dir: str = cfg.FIXED_LENGTH["file_path"],
    l2_mode: str = cfg.WINDOW["l2_mode"],
    window_size: int = cfg.WINDOW["window_size"],
    step_size: int = cfg.WINDOW["step_size"],
    encoder: Callable = psi_x,
    output_dir1: str = cfg.WINDOW["file_path"],
    output_dir2: str = cfg.ENTROPY["file_path"]
) -> None:
    """
    @description 网络流量的窗口化
    @param {str} input_dir 输入文件路径
    @param {str} l2_mode 链路层解析模式
    @param {int} window_size 窗口大小
    @param {int} step_size 滑动步长
    @param {Callable} encoder 编码器 
    @param {str} output_dir1 输出WINDOW文件路径
    @param {str} output_dir2 输出ENTROPY文件路径
    @return {None}
    """
    
    # 调用参数
    alphas = cfg.ENTROPY["alphas"]
    num_bins = cfg.ENTROPY["num_bins"]

    print(f"窗口大小 w={window_size}, 滑动步长 s={step_size}, 阶数参数 alphas={alphas}")


    # 遍历文件
    for filename in os.listdir(input_dir):

        name, ext = os.path.splitext(filename)
        if ext != ".pcap":
            continue

        print(f"正在处理 {filename}")

        pcap_path = os.path.join(input_dir, filename)
        
        save_path1 = os.path.join(output_dir1, name + "_window.csv")
        save_path2 = os.path.join(output_dir2, name + "_entropy.csv")

        pkt_generator = feature._parsePcapPkts(pcap_path, l2_mode)

        buffer: List[np.ndarray] = []
        results_window = []
        results_entropy = []

        # packet流处理
        for pkt in pkt_generator:

            # dict → vector
            x_k = feature_encode(pkt)
            buffer.append(x_k)

            # window构造
            if len(buffer) >= window_size:
                W = np.stack(buffer[:window_size], axis=0)

                # 标准化
                mean = np.mean(W, axis=0)
                std = np.std(W, axis=0)
                W_norm = (W - mean) / (std + 1e-6)

                #纯窗口特征
                x = encoder(W_norm)
                results_window.append(x)

                #窗口 + Rényi
                H = compute_window_renyi(W_norm, alphas, num_bins)
                z = np.concatenate([x, H])
                results_entropy.append(z)

                buffer = buffer[step_size:]
        
        # 保存 WINDOW
        if len(results_window) > 0:
            with open(save_path1, "w", newline="") as f:
                writer = csv.writer(f)
                dim = len(results_window[0])
                writer.writerow([f"f{i}" for i in range(dim)])
                writer.writerows(results_window)

        # 保存 ENTROPY
        if len(results_entropy) > 0:
            with open(save_path2, "w", newline="") as f:
                writer = csv.writer(f)
                dim = len(results_entropy[0])
                writer.writerow([f"z{i}" for i in range(dim)])
                writer.writerows(results_entropy)