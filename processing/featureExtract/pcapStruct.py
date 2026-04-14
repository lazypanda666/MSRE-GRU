#!/usr/bin/env python
# coding=utf-8
"""
@Description   PCAP文件结构体
@Author        Alex_McAvoy
@Date          2025-12-29 18:49:17
"""


import struct

class PcapHeader:
    def __init__(self, data: bytes) -> None:
        """
        @description 构造函数
        @param {*} self 类实例对象
        @param {bytes} data 比特数据
        @return {None}
        """

        # 原始字节数据
        self.raw = data
        # magic number，判断文件字节序
        self.magic = data[:4]

        # 判断是否为大端格式
        if self.magic == b"\xa1\xb2\xc3\xd4":
            self.endian = ">"
        # 判断是否为小端格式
        elif self.magic == b"\xd4\xc3\xb2\xa1":
            self.endian = "<"  # little endian

        # 构造unpack格式字符串，解析32位无符号整数
        self.fmt_I = self.endian + "I"
        # 从偏移的16~20字节解析最大可捕获包长度 snaplen
        self.snaplen = struct.unpack(self.fmt_I, data[16:20])[0]
        # 从偏移的20~24字节解析链路层类型linktype
        self.linktype = struct.unpack(self.fmt_I, data[20:24])[0]
