#!/usr/bin/env python
# coding=utf-8
"""
@Description   采样方法
@Author        Alex_McAvoy
@Date          2025-12-02 15:51:16
"""
import random
from scapy.all import RawPcapReader
import numpy as np

from utils.log import logger
from utils.wrapper import calTimes

def _countPktsStats(pkts: list, valid_pkts: list, sampled_pkts: list, unsampled_pkts: list) -> dict:
    """
    @description 统计采样过程中的包数量信息
    @param {list} pkts 原始扫描包列表
    @param {list} valid_pkts 过滤后的有效包列表
    @param {list} sampled_pkts 采样包列表
    @param {list} unsampled_pkts 未采样包列表
    @return {dict} 返回采样统计信息字典，包括扫描包数、有效包数、采样包数、未采样包数
    """
    return {
        "num_scanned_pkts": len(pkts),
        "num_valid_pkts": len(valid_pkts),
        "num_sampled_pkts": len(sampled_pkts),
        "num_unsampled_pkts": len(unsampled_pkts)
    }

@calTimes(logger, "分组随机采样完成")
def randomGSamplePkts(pcap_path: str, min_len: int, sample_beta: float) -> tuple[list, list, dict]:
    """
    @description 从PCAP文件中根据有效样本数分组随机采样
    @param {str} pcap_path PCAP文件路径
    @param {int} min_len 采样包的最小长度约束
    @param {float} sample_beta 有效样本数计算超参
    @return {tuple[list, list, dict]} 返回 (采样包列表, 未采样包列表, 包数量统计字典)
    """
    # 将整个 pcap 文件读入内存 
    pkts = []
    for pkt, pkt_meta in RawPcapReader(pcap_path):
        time = pkt_meta.sec + pkt_meta.usec / 1_000_000
        pkts.append((pkt, time))
    # 过滤掉长度小于 min_len 的包
    valid_pkts = [(pkt, ts) for (pkt, ts) in pkts if len(pkt) >= min_len]
    # 计算有效样本数 E
    E = int((1 - np.power(sample_beta, len(valid_pkts))) / (1 - sample_beta))

    # 计算区间大小
    step = len(valid_pkts) / E
    # 分组随机采样
    sampled_indices = []
    for i in range(E):
        # 区间起始与结束索引
        start = int(i * step)
        end = int((i + 1) * step)
        # 防止最后一个区间越界
        if end > len(valid_pkts):
            end = len(valid_pkts)
        # 在区间内随机选一个包
        index = random.randint(start, end - 1)
        sampled_indices.append(index)

    # 采样与未采样包
    sampled_pkts = [valid_pkts[i] for i in sampled_indices]
    unsampled_pkts = [valid_pkts[i] for i in range(len(valid_pkts)) if i not in sampled_indices]

    # 包数量统计
    pkts_stats_info = _countPktsStats(pkts, valid_pkts, sampled_pkts, unsampled_pkts)

    return sampled_pkts, unsampled_pkts, pkts_stats_info
