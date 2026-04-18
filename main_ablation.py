#!/usr/bin/env python
# coding=utf-8
"""
@Description   消融实验
@Author        lazypanda666
@Date          2026-04-18 17:01:25
"""


import sample.sampling as sample
import processing.dataProcessing as processing
import model.windowEntropy as window
import model.entropyState as distribution
import processing.splitDataset as split
import model.gruOnlyXAblation as gru_x
import model.gruOnlyEntropyAblation as gru_entropy
import evaluate.evaluateOnlyEntropyAblation as evaluate_entropy
import evaluate.evaluateOnlyXAblation as evaluate_x

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

    print(f"正在窗口化与雷尼熵计算")
    window.extractWindowFeaturesAndEntropy()
    print("--------------------------------------------")

    print(f"正在雷尼熵的窗口分布状态计算")
    distribution.buildEntropyStateVector()
    print("--------------------------------------------")

    print(f"正在划分数据集")
    split.splitTheDataset()
    print("--------------------------------------------")

    print(f"正在训练Only X模型")
    gru_x.train()
    print("--------------------------------------------")

    print(f"正在测试模型")
    evaluate_x.evaluateModel()
    print("--------------------------------------------")

    print(f"正在训练Only Entropy模型")
    gru_entropy.train()
    print("--------------------------------------------")
    
    print(f"正在测试模型")
    evaluate_entropy.evaluateModel()
    print("--------------------------------------------")