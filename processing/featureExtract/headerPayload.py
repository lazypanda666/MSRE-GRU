#!/usr/bin/env python
# coding=utf-8
"""
@Description   数据包Header与Payload提取
@Author        Alex_McAvoy
@Date          2025-12-03 21:00:22
"""
import os
import struct
from typing import Generator
import numpy as np

from processing.featureExtract.pcapStruct import PcapHeader
from processing.featureExtract.payload import _parsePcapPktsPayload
import processing.featureExtract.common as common

from const import cfg
from utils.log import logger
from utils.wrapper import calTimes



def _parsePcapPktsHeader(pcap_path: str, l2_mode: str) -> Generator:
    """
    @description 解析pcap包header(IPv4+TCP/UDP)
    @param {str} pcap_path pcap包路径
    @param {str} l2_mode 链路层解析模式
    @return {Generator} 逐包yield解析后的header字节列表
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
        effective_mode = l2_mode
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
        elif effective_mode == "ipv4":
            # L3层偏移从0开始
            l3_offset = 0
        # ======= 自动解析解析 =======
        elif effective_mode == "auto":
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
        # 版本与IHL字段
        ip_hdr = frame[l3_offset:]
        # IHL为低4bit，单位为4字节
        ihl = (ip_hdr[0] & 0x0F)
        ip_header_len = ihl * 4
        # 越界检查
        if len(ip_hdr) < ip_header_len:
            continue
        # ======= IPv4 内容 =======
        # 协议号
        protocol = ip_hdr[9]

        # IPv4头最小20，最大60，固定为60补齐
        ipv4_fixed = bytearray([0xFF]*60)
        cp = min(ip_header_len, 60)
        ipv4_fixed[:cp] = ip_hdr[:cp]

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
            
            # TCP头最小20，最大60，固定为60补齐
            tcp_fixed = bytearray([0xFF]*60)
            cp = min(tcp_header_len, 60)
            tcp_fixed[:cp] = tcp_hdr[:cp]
            # 组合Ipv4头和TCP头
            hdr = ipv4_fixed + tcp_fixed

            # IPv4+TCP补全128字节
            if len(hdr) < 128:
                hdr = hdr + bytearray([0xFF]*(128-len(hdr)))
            # 返回
            yield list(hdr[:128])

        # =============== UDP解析 ===============
        elif protocol == 17:
            # UDP头检查
            if len(frame) < l4_offset + 8:
                continue
            
            # UDP头永远固定为8字节，统一表示成与TCP一样的60字节
            udp_hdr = frame[l4_offset:l4_offset+8]
            udp_fixed = bytearray([0xFF]*60)
            udp_fixed[:8] = udp_hdr
            # 组合Ipv4头和UDP头
            hdr = ipv4_fixed + udp_fixed

            # IPv4+UCP补全128字节
            if len(hdr) < 128:
                hdr = hdr + bytearray([0xFF]*(128-len(hdr)))
            # 返回
            yield list(hdr)
        else:
            continue

@calTimes(logger, "数据包Header与Payload提取完毕")
def extractPacpPktsHeaderAndPayload(input_dir: str = cfg.ANONYMOUS["file_path"],
                                    sample_way: str = cfg.SAMPLE["way"],
                                    l2_mode: str = cfg.EXTRACT["l2_mode"],
                                    max_len_header: int = 128,
                                    max_len_payload: int = 64,
                                    model_name: str = cfg.MODEL["name"],
                                    output_dir: str = cfg.EXTRACT["file_path"]) -> None:
    """
    @description 对PCAP文件进行数据包header和payload提取，用于对比实验（AMLHP模型）
    @param {str} input_dir PCAP文件夹路径
    @param {str} sample_way 采样方式
    @param {str} l2_mode 链路层解析模式
    @param {int} max_len_header header最大长度
    @param {int} max_len_payload payload最大长度
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
        # 仅处理与采样方式相同的文件
        if way != sample_way:
            continue
        # 仅处理采样数据
        if sampled_val != "sampled":
            continue

        print(f"正在处理{filename}")
         
        # pacp文件路径
        pcap_path = input_dir + filename

        # 启动两个迭代器，同步到zip中
        header_iter = _parsePcapPktsHeader(pcap_path, l2_mode)
        payload_iter = _parsePcapPktsPayload(pcap_path, l2_mode)

        header_ls = []
        payload_ls = []

        # 使用 zip 同步逐包抽取
        for header, payload in zip(header_iter, payload_iter):
            # 对齐header并加入到list中
            header_ls.append(header[:max_len_header])
            
            # 对齐payload并加入到list中
            arr = np.zeros(max_len_payload, dtype=np.uint8)
            L = min(len(payload), max_len_payload)
            arr[:L] = payload[:L]
            payload_ls.append(arr)

        if len(header_ls) == 0:
            print(f"{filename} 中没有有效数据，跳过")
            continue

        # 转numpy并合并
        header_np = np.array(header_ls, dtype=np.uint8)
        payload_np = np.array(payload_ls, dtype=np.uint8)
        combined_np = np.concatenate([header_np, payload_np], axis=1)

        # 保存
        np.save(output_dir + f"{model_name}_{name}_header+payload.npy", combined_np)
