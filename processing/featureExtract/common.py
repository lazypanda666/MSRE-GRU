#!/usr/bin/env python
# coding=utf-8
"""
@Description   PCAP文件特征提取通用函数
@Author        Alex_McAvoy
@Date          2025-12-29 18:49:17
"""

import struct

def macAddr(b: bytes) -> str:
    """
    @description 解析MAC地址
    @param {bytes} b 字节数据
    @return {str} MAC地址
    """
    return ":".join(f"{x:02x}" for x in b)

def ipAddr(b: bytes) -> str:
    """
    @description 解析IP地址
    @param {bytes} b 字节数据
    @return {str} IP地址
    """
    return ".".join(str(x) for x in b)


def detectL3Offset(frame: bytes) -> tuple:
    """
    @description 猜测帧是Ethernet+IP，还是裸IP
    @param {bytes} frame 帧数据
    @return {tuple} 元组包含：
                  - {int} 偏移量
                  - {str} 帧类型
    """
    # 以太网
    if len(frame) >= 14:
        eth_type = struct.unpack("!H", frame[12:14])[0]
        if eth_type == 0x0800 and len(frame) >= 14 + 20:
            ver = frame[14] >> 4
            if ver in (4, 6):
                return 14, "ethernet"

    # 裸IP
    if len(frame) >= 20:
        ver = frame[0] >> 4
        if ver in (4, 6):
            return 0, "ipv4"

    return None