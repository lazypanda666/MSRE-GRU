#!/usr/bin/env python
# coding=utf-8
"""
@Description   网络流量的窗口化
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


# ψ_x（窗口编码）
def psi_x(W: np.ndarray) -> np.ndarray:
    mean = np.mean(W, axis=0)
    std = np.std(W, axis=0)
    max_val = np.max(W, axis=0)
    return np.concatenate([mean, std, max_val])



@calTimes(logger, "窗口构建与特征编码完成")
def extractWindowFeatures(
    input_dir: str = cfg.FIXED_LENGTH["file_path"],
    l2_mode: str = cfg.EXTRACT["l2_mode"],
    output_dir: str = cfg.WINDOW["file_path"],
    window_size: int = cfg.WINDOW["window_size"],
    step_size: int = cfg.WINDOW["step_size"],
    encoder: Callable = psi_x
) -> None:

    print(f"窗口大小 w={window_size}, 滑动步长 s={step_size}")


    # 遍历文件
    for filename in os.listdir(input_dir):

        name, ext = os.path.splitext(filename)
        if ext != ".pcap":
            continue

        print(f"正在处理 {filename}")

        pcap_path = os.path.join(input_dir, filename)
        save_path = os.path.join(output_dir, filename.replace(".pcap", "_window.csv"))


        pkt_generator = feature._parsePcapPkts(pcap_path, l2_mode)

        buffer: List[np.ndarray] = []
        results: List[np.ndarray] = []

        # packet流处理
        for pkt in pkt_generator:

            # dict → vector
            x_k = feature_encode(pkt)
            buffer.append(x_k)

            # window构造
            if len(buffer) >= window_size:

                W_t = np.stack(buffer[:window_size], axis=0)
                x_t = encoder(W_t)

                results.append(x_t)

                buffer = buffer[step_size:]

        # 写入CSV

        with open(save_path, "w", newline="") as f:
            if len(results) == 0:
                continue

            dim = len(results[0])
            writer = csv.writer(f)

            # header
            writer.writerow([f"f{i}" for i in range(dim)])

            # data
            writer.writerows(results)