#!/usr/bin/env python
# coding=utf-8
"""
@Description   实验主函数
@Author        Alex_McAvoy
@Date          2025-12-02 16:18:22
"""

import sample.sampling as sample
import processing.dataProcessing as processing
import model.window as window

from const import cfg

if __name__ == "__main__":
    # print("--------------------------------------------")
    # print(f"正在采样 {cfg.DATASET['name']} 数据集")
    # sample.sample()
    # print("--------------------------------------------")

    # print(f"正在匿名化")
    # processing.anonymizePcap()
    # print("--------------------------------------------")
    
    # print(f"正在定长化")
    # processing.truncatePcapPackets()
    # print("--------------------------------------------")

    print(f"正在窗口化")
    window.extractWindowFeatures()
    print("--------------------------------------------")
