#!/usr/bin/env python
# coding=utf-8
"""
@Description   数据集采样
@Author        Alex_McAvoy
@Date          2025-09-28 14:40:37
"""
import os
import random
import pickle
import numpy as np
from scapy.all import RawPcapReader
import dpkt

import sample.method as sam
from const import cfg
from utils.log import logger
from utils.wrapper import calTimes

@calTimes(logger, "采样完毕")
def sample(pcap_dir: str = cfg.DATASET["pcap_dir"],
           seed: int = cfg.COMMON["seed"], 
           sample_beta: float = cfg.SAMPLE["beta"],
           min_len: int = cfg.SAMPLE["min_len"],
           unsample_switch: bool = cfg.SAMPLE["unsample_switch"],
           save_path: str = cfg.SAMPLE["file_path"]) -> None:
    """
    @description 对PCAP文件采样
    @param {str} pcap_dir PCAP文件目录
    @param {int} seed 随机种子
    @param {float} sample_beta 有效样本数超参
    @param {int} min_len 采样包的最小长度约束
    @param {bool} unsample_switch 是否保存未采样数据
    @param {str} save_path 采样后PCAP文件存储路径
    @return {None} 
    """
    print(f"有效样本数超参：{sample_beta}，采样包的最小长度约束：{min_len}，随机种子：{seed}")

    # 随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 采样包数量统计信息
    pkts_stat_info_dict = dict()

    # 遍历所有pcap文件
    for filename in os.listdir(pcap_dir):
        print(f"正在处理{filename}")
        
        # 分割文件名和后缀
        name, ext = os.path.splitext(filename)
        
        # 拼接当前pcap文件路径
        pcap_path = pcap_dir + filename

        # 获取输入文件的 linktype
        reader = RawPcapReader(pcap_path)
        linktype = reader.linktype
        reader.close()

        # 分组随机采样
        sampled, unsampled, pkts_stat_info = sam.randomGSamplePkts(pcap_path, min_len, sample_beta)
        pkts_stat_info_dict[name] = pkts_stat_info

        # 保存
        with open(f"{save_path}pkts_stat_info.pkl", "wb") as f:
            pickle.dump(pkts_stat_info_dict, f)
        with open(f"{save_path}sampled_{filename}", "wb") as f:
            writer = dpkt.pcap.Writer(f)
            for (pkt, ts) in sampled:
                writer.writepkt(pkt, ts)
        if unsample_switch:
            with open(f"{save_path}unsampled_{filename}", "wb") as f:
                writer = dpkt.pcap.Writer(f)
                for (pkt, ts) in unsampled:
                    writer.writepkt(pkt, ts)
