#!/usr/bin/env python
# coding=utf-8
"""
@Description   网络流量的窗口化，并计算雷尼熵
@Author        lazypanda666
@Date          2026-04-15 15:25:00
"""

import os
import numpy as np
from typing import Callable,Optional

import processing.featureExtract.feature as feature

from const import cfg
from utils.log import logger
from utils.wrapper import calTimes



def featureEncode(pkt: dict) -> np.ndarray:
    """
    @description 特征编码
    @param {dict} pkt 数据包字典 
    @return {np.ndarray}
    """    
    def safeFloat(x):
        """
        @description 将输入值转换为浮点数
        @return {float}
        """        
        if x is None:
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        return 0.0

    return np.array([safeFloat(pkt[f]) for f in feature.FIELDNAMES], dtype=np.float32)



def psi_x(W: np.ndarray) -> np.ndarray:
    """
    @description 内容编码
    @return {np.ndarray}
    """    
    mean = np.mean(W, axis=0)
    std = np.std(W, axis=0)
    max_val = np.max(W, axis=0)
    min_val = np.min(W, axis=0)

    return np.concatenate([mean, std, max_val, min_val]).astype(np.float32)



def renyiEntropy(p: np.ndarray, alpha: float) -> float:
    """
    @description 计算概率分布的雷尼熵
    @param {float} alpha 雷尼熵的阶数参数
    @return {float}
    """    
    p = p + 1e-12

    if abs(alpha - 1.0) < 1e-6:
        return -np.sum(p * np.log(p))

    log_p = np.log(p)
    a = alpha * log_p

    a_max = np.max(a)
    sum_exp = np.sum(np.exp(a - a_max))

    return (1.0 / (1 - alpha)) * (np.log(sum_exp) + a_max)



def adaptiveBinning(feature: np.ndarray, num_bins: int) -> Optional[np.ndarray]:
    """
    @description 使用分位数分箱
    @param {int} num_bins 分箱数量
    @return {Optional[np.ndarray]}
    """    
    
    try:
        bins = np.unique(np.quantile(feature, np.linspace(0, 1, num_bins + 1)))
        if len(bins) <= 2:
            return None
        hist, _ = np.histogram(feature, bins=bins)
        return hist
    except Exception:
        return None



def computeWindowRenyi(W: np.ndarray, alphas, num_bins: int) -> np.array:
    """
    @description 计算滑动窗口数据的雷尼熵特征矩阵
    @param {list} alphas 雷尼熵的阶数参数的列表
    @param {int} num_bins 分箱数量
    @return {np.array}
    """    

    w, m = W.shape
    K = len(alphas)

    H = np.zeros((K, m), dtype=np.float32)

    for j in range(m):
        feature_j = W[:, j]

        # 常数特征跳过
        if np.allclose(feature_j, feature_j[0]):
            continue

        # 自适应分箱
        hist = adaptiveBinning(feature_j, num_bins)

        if hist is None or np.sum(hist) == 0:
            continue

        p = hist.astype(np.float32)
        p = p / (np.sum(p) + 1e-12)

        for k, alpha in enumerate(alphas):
            H[k, j] = renyiEntropy(p, float(alpha))

    # 标准化
    H = (H - np.mean(H)) / (np.std(H) + 1e-6)

    return H.flatten()



def computeGlobalStats(all_windows: list) -> tuple:
    """
    @description 计算全局统计特征
    @param {list} all_windows 窗口数据列表
    @return {tuple}
    """    
    W_all = np.concatenate(all_windows, axis=0)

    mean = np.mean(W_all, axis=0)
    std = np.std(W_all, axis=0)

    # 防止除0
    std = np.where(std < 1e-6, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)



@calTimes(logger, "窗口 + Rényi特征构建完成")
def extractWindowFeaturesAndEntropy(input_dir: str = cfg.FIXED_LENGTH["file_path"],
                                    l2_mode: str = cfg.WINDOW["l2_mode"],
                                    window_size: int = cfg.WINDOW["window_size"],
                                    step_size: int = cfg.WINDOW["step_size"],
                                    encoder: Callable = psi_x,
                                    output_dir_window: str = cfg.WINDOW["file_path"],
                                    output_dir_entropy: str = cfg.ENTROPY["file_path"] ) -> None:
    """
    @description 窗口化+雷尼熵特征构建
    @param {str} input_dir 定长化后的输入文件路径
    @param {str} l2_mode 协议处理模式
    @param {int} window_size 滑动窗口大小
    @param {int} step_size 滑动步长
    @param {Callable} encoder 数据包编码器函数
    @param {str} output_dir_window 窗口特征输出文件路径
    @param {str} output_dir_entropy 熵特征输出文件路径
    @return {None}
    """    

    alphas = cfg.ENTROPY["alphas"]
    num_bins = cfg.ENTROPY["num_bins"]

    print(f"[INFO] window={window_size}, step={step_size}, alphas={alphas}")

    for filename in os.listdir(input_dir):

        if not filename.endswith(".pcap"):
            continue

        print(f"[Processing] {filename}")

        name = filename.split(".")[0]

        pcap_path = os.path.join(input_dir, filename)

        save_x = os.path.join(output_dir_window, name + "_X.npy")
        save_e = os.path.join(output_dir_entropy, name + "_E.npy")

        pkt_generator = feature._parsePcapPkts(pcap_path, l2_mode)

        buffer = []
        windows = []


        # 窗口构建
        for pkt in pkt_generator:
            x_k = featureEncode(pkt)
            buffer.append(x_k)

            if len(buffer) >= window_size:
                W = np.stack(buffer[:window_size], axis=0)
                windows.append(W)
                buffer = buffer[step_size:]

        if len(windows) == 0:
            continue


        # 全局统计
        mean, std = computeGlobalStats(windows)

        X_list = []
        E_list = []


        # 特征提取
        for W in windows:

            # 内容特征（归一化）
            W_norm = (W - mean) / (std + 1e-6)
            x = encoder(W_norm)

            # Rényi（原始分布）
            e = computeWindowRenyi(W, alphas, num_bins)

            X_list.append(x)
            E_list.append(e)

        X = np.array(X_list, dtype=np.float32)
        E = np.array(E_list, dtype=np.float32)


        # 序列级标准化
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6)
        E = (E - np.mean(E, axis=0)) / (np.std(E, axis=0) + 1e-6)


        # 保存
        np.save(save_x, X)
        np.save(save_e, E)

        print(f"[Saved] {name} -> X:{X.shape}, E:{E.shape}")