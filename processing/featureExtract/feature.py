#!/usr/bin/env python
# coding=utf-8
"""
@Description   数据包包级特征提取
@Author        Alex_McAvoy
@Date          2025-12-03 21:00:22
"""
import os
import csv
import struct
from typing import Generator

from processing.featureExtract.pcapStruct import PcapHeader
import processing.featureExtract.common as common

from const import cfg
from utils.log import logger
from utils.wrapper import calTimes

# 特征字段
FIELDNAMES = [
    "pkt_id",
    "ts_sec",
    "ts_usec",
    "timestamp",
    "eth_src",
    "eth_dst",
    "eth_type",
    "ip_src",
    "ip_dst",
    "ip_proto",
    "ip_total_len",
    "ip_ttl",
    "ip_id",
    "ip_flags_df",
    "ip_flags_mf",
    "ip_frag_offset",
    "src_port",
    "dst_port",
    "l4_len",
    "payload_len",
    "tcp_seq",
    "tcp_ack",
    "tcp_window",
    "tcp_checksum",
    "tcp_urg_ptr",
    "tcp_flags_byte",
    "tcp_fin",
    "tcp_syn",
    "tcp_rst",
    "tcp_psh",
    "tcp_ack_flag",
    "tcp_urg",
    "tcp_ece",
    "tcp_cwr",
    "udp_len",
    "udp_checksum",
    "udp_is_dns",
    "udp_is_dhcp",
    "udp_is_ntp",
]

def _parsePcapPkts(pcap_path: str, l2_mode: str) -> Generator:
    """
    @description 解析pcap包
    @param {str} pcap_path pcap包路径
    @param {str} l2_mode 链路层解析模式
    @return {Generator} 逐包yield解析后的特征字典
    """
    
    # 以二进制格式读取
    with open(pcap_path, "rb") as f:
        data = f.read()

    # 解析数据包全局头
    header = PcapHeader(data[:24])

    # 创建PcapHeader对象，读取字节序与链路层类型
    gh = PcapHeader(data[:24])
    # 获取32位整数的unpcak格式
    fmt_I = gh.fmt_I

    # 包长度
    pcap_len = len(data)
    # 数据指针初始位置，全局头24字节后
    ptr = 24
    # 包序号
    pkt_id = 0

    # 按位读取整个pcap包
    while ptr + 16 <= pcap_len:
        # =============== Header解析 ===============
        # 解析时间戳秒
        ts_sec = struct.unpack(fmt_I, data[ptr:ptr + 4])[0]
        # 解析时间戳微秒
        ts_usec = struct.unpack(fmt_I, data[ptr + 4:ptr + 8])[0]
        # 帧在文件中的实际长度
        incl_len = struct.unpack(fmt_I, data[ptr + 8:ptr + 12])[0]
        # 帧未截断前的原始长度
        _orig_len = struct.unpack(fmt_I, data[ptr + 12:ptr + 16])[0]

        # 移动指针到frame开始位置
        ptr += 16
        # 越界检查
        if ptr + incl_len > pcap_len:
            break

        # 提取帧字节
        frame = data[ptr:ptr + incl_len]
        # 指针跳过frame
        ptr += incl_len
        # 包计数+1
        pkt_id += 1

        # =============== 根据链路层解析模式，决定如何解析L2层 ===============
        effective_mode  = l2_mode
        l3_offset = None
        eth_src = ""
        eth_dst = ""
        eth_type = 0
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
            # 目的MAC地址
            eth_dst = common.macAddr(frame[0:6])
            # 源MAC地址
            eth_src = common.macAddr(frame[6:12])
            # 以太类型
            eth_type = struct.unpack("!H", frame[12:14])[0]
            # 只处理IPv4
            if eth_type != 0x0800:
                continue
            # IPv4头起始偏移14字节
            l3_offset = 14
        # ======= Raw Ipv4解析 =======
        elif l2_mode == "ipv4":
            # 无MAC层，全部为空
            eth_src = ""
            eth_dst = ""
            eth_type = 0x0800
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
            l3_offset, detected_mode = detected
            # Ethernet模式
            if detected_mode == "ethernet":
                eth_dst = common.macAddr(frame[0:6])
                eth_src = common.macAddr(frame[6:12])
                eth_type = struct.unpack("!H", frame[12:14])[0]
            # Raw IPv4模式
            else:
                eth_src = ""
                eth_dst = ""
                eth_type = 0x0800
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
        # 数据包ID
        identification = struct.unpack("!H", ip_hdr[4:6])[0]
        # flags与帧偏移
        flags_frag = struct.unpack("!H", ip_hdr[6:8])[0]
        # TTL
        ttl = ip_hdr[8]
        # L4协议号
        protocol = ip_hdr[9]
        # 源IP
        src_ip = common.ipAddr(ip_hdr[12:16])
        # 目的IP
        dst_ip = common.ipAddr(ip_hdr[16:20])
        # IP flags
        ip_flags = (flags_frag & 0xE000) >> 13
        # 分片偏移量
        frag_offset = flags_frag & 0x1FFF
        # DF标志位
        flag_df = bool(ip_flags & 0b010)
        # MF标志位
        flag_mf = bool(ip_flags & 0b001)

        # L4 起始偏移
        l4_offset = l3_offset + ip_header_len
        if len(frame) < l4_offset:
            continue
        
        # =============== 封装基础字段 ===============
        base = {
            "pkt_id": pkt_id,
            "ts_sec": ts_sec,
            "ts_usec": ts_usec,
            "timestamp": ts_sec + ts_usec / 1_000_000.0,
            "eth_src": eth_src,
            "eth_dst": eth_dst,
            "eth_type": eth_type,
            "ip_src": src_ip,
            "ip_dst": dst_ip,
            "ip_proto": protocol,
            "ip_total_len": total_len,
            "ip_ttl": ttl,
            "ip_id": identification,
            "ip_flags_df": int(flag_df),
            "ip_flags_mf": int(flag_mf),
            "ip_frag_offset": frag_offset,
        }

        # =============== TCP解析 ===============
        if protocol == 6:
            # TCP头检查
            if len(frame) < l4_offset + 20:
                continue
            
            # TCP头
            tcp_hdr = frame[l4_offset:]
            # 源端口
            src_port = struct.unpack("!H", tcp_hdr[0:2])[0]
            # 目的端口
            dst_port = struct.unpack("!H", tcp_hdr[2:4])[0]
            # 序列号
            seq = struct.unpack("!I", tcp_hdr[4:8])[0]
            # 确认号
            ack = struct.unpack("!I", tcp_hdr[8:12])[0]
            # 数据偏移字段，单位4字节
            data_offset = (tcp_hdr[12] >> 4) & 0x0F
            tcp_header_len = data_offset * 4

            # 越界检查
            if len(tcp_hdr) < tcp_header_len:
                continue
            
            # TCP flags
            flags_byte = tcp_hdr[13]
            # 窗口大小
            window_size = struct.unpack("!H", tcp_hdr[14:16])[0]
            # 校验和
            checksum = struct.unpack("!H", tcp_hdr[16:18])[0]
            # 紧急指针
            urg_ptr = struct.unpack("!H", tcp_hdr[18:20])[0]

            # 各flags解析
            cwr = bool(flags_byte & 0x80)
            ece = bool(flags_byte & 0x40)
            urg = bool(flags_byte & 0x20)
            ack_flag = bool(flags_byte & 0x10)
            psh = bool(flags_byte & 0x08)
            rst = bool(flags_byte & 0x04)
            syn = bool(flags_byte & 0x02)
            fin = bool(flags_byte & 0x01)

            # L4层总长度
            l4_total_len = max(total_len - ip_header_len, 0)
            # payload长度
            payload_len = max(l4_total_len - tcp_header_len, 0)

            # 合并字段
            row = dict(base)
            row.update(
                {
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "l4_len": l4_total_len,
                    "payload_len": payload_len,
                    # TCP真实字段
                    "tcp_seq": seq,
                    "tcp_ack": ack,
                    "tcp_window": window_size,
                    "tcp_checksum": checksum,
                    "tcp_urg_ptr": urg_ptr,
                    "tcp_flags_byte": flags_byte,
                    "tcp_fin": int(fin),
                    "tcp_syn": int(syn),
                    "tcp_rst": int(rst),
                    "tcp_psh": int(psh),
                    "tcp_ack_flag": int(ack_flag),
                    "tcp_urg": int(urg),
                    "tcp_ece": int(ece),
                    "tcp_cwr": int(cwr),
                    # UDP 默认值
                    "udp_len": 0,
                    "udp_checksum": 0,
                    "udp_is_dns": 0,
                    "udp_is_dhcp": 0,
                    "udp_is_ntp": 0,
                }
            )
            yield row

        # =============== UDP解析 ===============
        if protocol == 17:
            # UDP头检查
            if len(frame) < l4_offset + 8:
                continue

            # UDP头
            udp_hdr = frame[l4_offset:]
            # 源端口
            src_port = struct.unpack("!H", udp_hdr[0:2])[0]
            # 目的端口
            dst_port = struct.unpack("!H", udp_hdr[2:4])[0]
            # UDP长度
            udp_len = struct.unpack("!H", udp_hdr[4:6])[0]
            # 校验和
            udp_checksum = struct.unpack("!H", udp_hdr[6:8])[0]
            # payload长度
            payload_len = max(udp_len - 8, 0)

            # DNS端口
            is_dns = int(src_port == 53 or dst_port == 53)
            # DHCP端口
            is_dhcp = int(src_port in (67, 68) or dst_port in (67, 68))
            # NTP端口
            is_ntp = int(src_port == 123 or dst_port == 123)

            # 合并字段
            row = dict(base)
            row.update(
                {
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "l4_len": udp_len,
                    "payload_len": payload_len,
                    # TCP字段默认值
                    "tcp_seq": 0,
                    "tcp_ack": 0,
                    "tcp_window": 0,
                    "tcp_checksum": 0,
                    "tcp_urg_ptr": 0,
                    "tcp_flags_byte": 0,
                    "tcp_fin": 0,
                    "tcp_syn": 0,
                    "tcp_rst": 0,
                    "tcp_psh": 0,
                    "tcp_ack_flag": 0,
                    "tcp_urg": 0,
                    "tcp_ece": 0,
                    "tcp_cwr": 0,
                    # UDP真实字段
                    "udp_len": udp_len,
                    "udp_checksum": udp_checksum,
                    "udp_is_dns": is_dns,
                    "udp_is_dhcp": is_dhcp,
                    "udp_is_ntp": is_ntp,
                }
            )
            yield row

@calTimes(logger, "数据包特征提取完毕")
def extractPcapPktsFeature(input_dir: str = cfg.ANONYMOUS["file_path"], 
                           l2_mode: str = cfg.EXTRACT["l2_mode"],
                           output_dir: str = cfg.EXTRACT["file_path"]) -> None:
    """
    @description 对PCAP文件进行数据包特征提取，用于采样实验
    @param {str} input_dir PCAP文件夹路径
    @param {str} l2_mode 链路层解析模式
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
        sampled_val, label = name.split("_")
        # 仅处理采样数据
        if sampled_val != "sampled":
            continue

        print(f"正在处理{filename}")

        # PCAP文件路径
        pcap_path = input_dir + filename
        
        # 使用迭代器解析整个pcap文件
        rows = _parsePcapPkts(pcap_path, l2_mode)

        # 保存csv数据
        with open(output_dir + name + ".csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
