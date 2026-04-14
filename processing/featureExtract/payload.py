#!/usr/bin/env python
# coding=utf-8
"""
@Description   PCAP文件payload提取
@Author        Alex_McAvoy
@Date          2025-12-29 18:49:17
"""
import os
import struct
from typing import Generator
import numpy as np

from processing.featureExtract.pcapStruct import PcapHeader
import processing.featureExtract.common as common

from const import cfg
from utils.log import logger
from utils.wrapper import calTimes

def _parsePcapPktsPayload(pcap_path: str, l2_mode: str) -> Generator:
    """
    @description 解析pcap包的payload
    @param {str} pcap_path pcap包路径
    @param {str} l2_mode 链路层解析模式
    @return {Generator} 逐包yield解析后的payload列表
    """
    
    # 以二进制格式读取
    with open(pcap_path, "rb") as f:
        data = f.read()

    # 创建PcapHeader对象，读取字节序与链路层类型
    gh = PcapHeader(data[:24])
    # 获取32位整数的unpcak格式
    fmt_I = gh.fmt_I
    # 包长度
    pcap_len = len(data)
    # 数据指针初始位置，全局头24字节后
    ptr = 24

    # 按位读取整个pcap包
    while ptr + 16 <= pcap_len:
        # =============== Header解析 ===============
        # 帧在文件中的实际长度
        incl_len = struct.unpack(fmt_I, data[ptr + 8:ptr + 12])[0]
        # 移动指针到frame开始位置
        ptr += 16
        # 越界检查
        if ptr + incl_len > pcap_len:
            break

        # 提取帧字节
        frame = data[ptr:ptr + incl_len]
        # 指针跳过frame
        ptr += incl_len

        # =============== 根据链路层解析模式，决定如何解析L2层 ===============
        effective_mode  = l2_mode
        if effective_mode == "pcap_header":
            # 以太网
            if gh.linktype == 1:
                effective_mode = "ethernet"
            # Raw Ipv4
            elif gh.linktype == 228:
                effective_mode = "ipv4"
            # 自动判断
            else:
                effective_mode = "auto"

        # ======= Ethernet解析 =======
        if effective_mode == "ethernet":
            # 帧太短，无法解析MAC头，跳过
            if len(frame) < 14:
                continue
            # 以太类型
            eth_type = struct.unpack("!H", frame[12:14])[0]
            # 只处理IPv4
            if eth_type != 0x0800:
                continue
            # IPv4头起始偏移14字节
            l3_offset = 14
        # ======= Raw Ipv4解析 =======
        elif l2_mode == "ipv4":
            # L3层偏移从0开始
            l3_offset = 0
        # ======= 自动解析解析 =======
        elif l2_mode == "auto":
            # 自动检测IPv4头偏移
            detected = common.detectL3Offset(frame)
            # 未检测到IPv4头，跳过
            if detected is None:
                continue
            # 偏移与类型
            l3_offset, _ = detected
        else:
            raise ValueError("未知的解析类型")

        # =============== IPv4解析 ===============

        # ======= IPv4头 =======
        if len(frame) < l3_offset + 20:
            continue
        ip_hdr = frame[l3_offset:]
        # 版本与IHL字段
        ver_ihl = ip_hdr[0]
        # IHL为低4bit，单位为4字节
        ihl = ver_ihl & 0x0F
        ip_header_len = ihl * 4
        # 越界检查
        if len(ip_hdr) < ip_header_len:
            continue

        # ======= IPv4 内容 =======
        # 总长度
        total_len = struct.unpack("!H", ip_hdr[2:4])[0]
        # L4协议号
        protocol = ip_hdr[9]

        # L4 起始偏移
        l4_offset = l3_offset + ip_header_len
        if len(frame) < l4_offset:
            continue
        
        # =============== TCP解析 ===============
        if protocol == 6:
            # TCP头检查
            if len(frame) < l4_offset + 20:
                continue
            
            # TCP头
            tcp_hdr = frame[l4_offset:]
            # 数据偏移字段，单位4字节
            data_offset = (tcp_hdr[12] >> 4) & 0x0F
            tcp_header_len = data_offset * 4

            # 越界检查
            if len(tcp_hdr) < tcp_header_len:
                continue
            
            # L4层总长度
            l4_total_len = max(total_len - ip_header_len, 0)
            # payload长度
            payload_len = max(l4_total_len - tcp_header_len, 0)

            if payload_len <= 0:
                continue

            # payload偏移
            payload_offset = l4_offset + tcp_header_len
            # payload
            payload = frame[payload_offset: payload_offset + payload_len]
            yield list(payload)

        # =============== UDP解析 ===============
        if protocol == 17:
            # UDP头检查
            if len(frame) < l4_offset + 8:
                continue

            # UDP头
            udp_hdr = frame[l4_offset:]
            # UDP长度
            udp_len = struct.unpack("!H", udp_hdr[4:6])[0]
            # payload长度
            payload_len = max(udp_len - 8, 0)

            if payload_len <= 0:
                continue

            payload_offset = l4_offset + 8
            payload = frame[payload_offset: payload_offset + payload_len]
            yield list(payload)

@calTimes(logger, "数据包payload提取完毕")
def extractPcapPktsPayload(input_dir: str = cfg.ANONYMOUS["file_path"],
                       sample_way: str = cfg.SAMPLE["way"],
                       l2_mode: str = cfg.EXTRACT["l2_mode"],
                       max_len: int = 1460,
                       model_name: str = cfg.MODEL["name"],
                       output_dir: str = cfg.EXTRACT["file_path"]) -> None:
    """
    @description 对PCAP文件进行数据包payload提取，用于对比实验（Transformer模型）
    @param {str} input_dir PCAP文件夹路径
    @param {str} sample_way 采样方式
    @param {str} l2_mode 链路层解析模式
    @param {int} max_len payload最大长度
    @param {str} model_name 模型名
    @param {str} output_dir 存储路径
    @return {None}
    """
    
    for filename in os.listdir(input_dir):
        # 分割文件名和后缀
        name, ext = os.path.splitext(filename)

        # 仅处理pcap文件
        if ext != ".pcap":
            continue

        # 分割文件名
        way, sampled_val, label = name.split("_")
        # 仅处理采样方式相同的 pcap 文件
        if way != sample_way:
            continue
        # 仅处理采样数据
        if sampled_val != "sampled":
            continue

        print(f"正在处理{filename}")

        # PCAP文件路径
        pcap_path = input_dir + filename

        # 使用迭代器解析整个pcap文件
        payload_iter = _parsePcapPktsPayload(pcap_path, l2_mode)

        # 转为list
        payload_ls = list(payload_iter)
        num_samples = len(payload_ls)

        if num_samples == 0:
            print(f"{filename}中未提取到有效的payload")
            continue

        # 对齐max_len
        payload_np = np.zeros((num_samples, max_len), dtype=np.uint8)
        for i, payload in enumerate(payload_ls):
            length = min(len(payload), max_len)
            payload_np[i, :length] = payload[:length]

        # 保存
        np.save(output_dir + f"{model_name}_{name}_payload.npy", payload_np)

