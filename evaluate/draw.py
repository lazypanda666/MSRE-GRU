#!/usr/bin/env python
# coding=utf-8
"""
@Description   绘制模型评估相关图像
@Author        Alex_McAvoy
@Date          2025-07-26 17:55:59
"""

import numpy as np
import pandas as pd
import plotly.express as px

def plotConfusionMatrix(filename_suffix: str, 
                        confusion_matrix: np.ndarray, 
                        labels: list, 
                        save_path: str) -> None:
    """
    @description 绘制混淆矩阵热力图
    @param {str} filename_suffix 存储文件名前缀
    @param {np.ndarray} confusion_matrix 混淆矩阵
    @param {list} labels 分类标签名称（顺序需与混淆矩阵行列对应）
    @param {str} save_path 绘图存储路径
    @return {None}
    """

    df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

    fig = px.imshow(
        df,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="预测标签", y="真实标签", color="样本数"),
        x=labels,
        y=labels
    )

    fig.update_layout(
        title="混淆矩阵",
        xaxis=dict(tickangle=45),
        width=700,
        height=700,
        font=dict(family="Microsoft YaHei", size=14)
    )

    fig.write_image(save_path + f"{filename_suffix}_cm.png")
    fig.write_html(save_path + f"{filename_suffix}_cm.html")