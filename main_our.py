#!/usr/bin/env python
# coding=utf-8
"""
@Description   实验主函数
@Author        Alex_McAvoy
@Date          2025-12-02 16:18:22
"""

import sample.sampling as sample
import processing.dataProcessing as processing
import model.window_entropy as window
import model.window_distribution_state_of_Renyi_entropy as distribution
import processing.split_the_dataset as split
import model.GRU as gru
import evaluate.evaluate as evaluate

from const import cfg

if __name__ == "__main__":
    print("--------------------------------------------")
    print(f"正在采样 {cfg.DATASET['name']} 数据集")
    sample.sample()
    print("--------------------------------------------")

    print(f"正在匿名化")
    processing.anonymizePcap()
    print("--------------------------------------------")
    
    print(f"正在定长化")
    processing.truncatePcapPackets()
    print("--------------------------------------------")

    print(f"正在窗口化")
    window.extractWindowFeatures()
    print("--------------------------------------------")

    print(f"正在窗口化与雷尼熵计算")
    window.extractWindowFeaturesAndEntropy()
    print("--------------------------------------------")

    print(f"正在雷尼熵的窗口分布状态计算")
    distribution.buildEntropyStateVector()
    print("--------------------------------------------")

    划分数据集
    print(f"正在划分数据集")
    split.SplitTheDataset()
    print("--------------------------------------------")

    print(f"正在训练模型")
    gru.train()
    print("--------------------------------------------")

    print(f"正在测试模型")
    evaluate.evaluateModel()
    print("--------------------------------------------")