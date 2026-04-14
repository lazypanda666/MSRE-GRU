#!/usr/bin/env python
# coding=utf-8
"""
@Description   数据处理
@Author        Alex_McAvoy
@Date          2025-09-20 16:09:55
"""

import os
import pandas as pd
from scapy.all import Ether, IP, IPv6, TCP, UDP
from scapy.all import RawPcapReader, PcapWriter
from processing.featureExtract.common import detectL3Offset

from const import cfg
from utils.log import logger
from utils.wrapper import calTimes

@calTimes(logger, "匿名化处理完成")
def anonymizePcap(input_dir: str = cfg.SAMPLE["file_path"], 
                  sampled: str = "sampled",
                  output_dir: str = cfg.ANONYMOUS["file_path"]) -> None:
    """
    @description PCAP 文件匿名化
    @param {str} input_dir PCAP文件夹路径
    @param {str} sample_way 采样方式
    @param {str} sampled 数据获取方式，sampled采样，unsampled未采样
    @param {str} output_dir 匿名化文件夹路径
    @return {None}
    """

    for filename in os.listdir(input_dir):
        # 分割文件名和后缀
        name, ext = os.path.splitext(filename)

        # 仅处理pcap文件
        if ext != ".pcap":
            continue

        # 分割文件名
        sampled_val, label = name.split("_")
        # 仅处理与数据获取方式相同的数据
        if sampled_val != sampled:
            continue

        print(f"正在处理{filename}")
        # PCAP文件路径
        pcap_path = input_dir + filename
        # 输出文件路径
        save_path = output_dir + filename
        
        # 创建writer
        writer = PcapWriter(save_path, append=False, sync=True)
        for i, (pkt_data, pkt_metadata) in enumerate(RawPcapReader(pcap_path)):
            
            # 使用自定义的检测函数来判断帧类型
            result = detectL3Offset(pkt_data)
            if result is None:
                continue
            else:
                offset, pkt_type = result

            # Ethernet网
            if pkt_type == "ethernet":
                pkt = Ether(pkt_data)
            # 裸IPv4
            elif pkt_type == "ipv4":
                pkt_ip = None
                eth_type = None

                # 解析IP层
                try:
                    # IPv4
                    pkt_ip = IP(pkt_data)
                    eth_type = 0x0800
                except:
                    # IPv6
                    try:
                        pkt_ip = IPv6(pkt_data)
                        eth_type = 0x86DD
                    # 既不是 IPv4 也不是 IPv6，跳过
                    except Exception:
                        continue

                # 补Ethernet头
                pkt = Ether(dst="00:00:00:00:00:00", src="00:00:00:00:00:00", type=eth_type) / pkt_ip
            else:
                continue

            # MAC匿名化化
            if Ether in pkt:
                pkt[Ether].src = "00:00:00:00:00:00"
                pkt[Ether].dst = "00:00:00:00:00:00"

            # IPv4匿名化
            if IP in pkt:
                pkt[IP].src = "0.0.0.0"
                pkt[IP].dst = "0.0.0.0"
                # 删除校验和，保存时让scapy包重新计算
                if hasattr(pkt[IP], "chksum"):
                    del pkt[IP].chksum

            # TCP、UDP校验和删除，保存时让scapy包重新计算
            if TCP in pkt and hasattr(pkt[TCP], "chksum"):
                del pkt[TCP].chksum
            if UDP in pkt and hasattr(pkt[UDP], "chksum"):
                del pkt[UDP].chksum

            # 时间戳写回
            pkt.time = pkt_metadata.sec + pkt_metadata.usec / 1_000_000

            # 存入
            writer.write(pkt)

        # 关闭 writer
        writer.close()

@calTimes(logger, "截断/补齐数据包已完成")
def truncatePcapPackets(input_dir: str = cfg.ANONYMOUS["file_path"], 
                        sampled: str = "sampled",
                        n: int = cfg.FIXED_LENGTH["n"],
                        output_dir: str = cfg.FIXED_LENGTH["file_path"]) -> None:
    """
    @description 对PCAP文件进行截断、补齐
    @param {str} input_dir PCAP文件夹路径
    @param {str} sample_way 采样方式
    @param {str} sampled 数据获取方式，sampled采样，unsampled未采样
    @param {int} n 截断/补零数据包的固定长度
    @param {str} output_dir 存储路径
    @return {None}
    """
    print(f"数据包固定长度为：{n} 字节")
    for filename in os.listdir(input_dir):
        # 分割文件名和后缀
        name, ext = os.path.splitext(filename)

        # 分割文件名
        sampled_val, label = name.split("_")
        # 仅处理与数据获取方式相同的数据
        if sampled_val != sampled:
            continue

        print(f"正在处理{filename}")
        # PCAP文件路径
        pcap_path = input_dir + filename
        # 输出文件路径
        save_path = output_dir + filename

        # 创建writer
        writer = PcapWriter(save_path, append=False, sync=True)
        for i, (pkt_data, pkt_metadata) in enumerate(RawPcapReader(pcap_path)):
            # 对数据包截断或补零到固定长度n
            if len(pkt_data) >= n:
                pkt_bytes = pkt_data[:n]
            else:
                pkt_bytes = pkt_data + bytes([0] * (n - len(pkt_data)))
            
            # 使用截断或补零后的字节重新构造以太网帧
            pkt = Ether(pkt_bytes)
            # 时间戳写回
            pkt.time = pkt_metadata.sec + pkt_metadata.usec / 1_000_000

            # 存入
            writer.write(pkt)
            
        # 关闭writer
        writer.close()

