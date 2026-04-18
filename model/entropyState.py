#!/usr/bin/env python
# coding=utf-8
"""
@Description   雷尼熵的窗口分布状态
@Author        lazypanda666
@Date          2026-04-15 18:11:45
"""

import os
import numpy as np
import torch
import torch.nn as nn

from const import cfg
from utils.log import logger
from utils.wrapper import calTimes



class EntropyEncoder(nn.Module):
    """
    @description 实现了一个简单的神经网络编码器
    @param input_dim: 输入数据的维度
    @param embed_dim: 嵌入维度
    @return: 返回模型
    """    
    def __init__(self, input_dim, embed_dim) -> None:
        """
        @description 定义了模型的网络结构，两层线性变换和层归一化
        @param input_dim: 输入数据的维度
        @param embed_dim: 嵌入维度
        @return: {None}
        """ 

        super().__init__()

        self.net = nn.Sequential(
        nn.Linear(input_dim, embed_dim),
        nn.LayerNorm(embed_dim),
        nn.ReLU(),
        nn.Linear(embed_dim, embed_dim),
        nn.LayerNorm(embed_dim)
        )

    def forward(self, x) -> torch.Tensor:
        """
        @description 前向传播函数
        @param x: 输入数据
        @return: {torch.Tensor}
        """
        return self.net(x)



def emaSmooth(x, alpha=0.9) -> np.ndarray:
    """
    @description 指数加权平均
    @param x: 输入数据
    @param alpha: 平滑系数，控制历史数据对当前值的影响程度
    @return: {np.ndarray}
    """
    x_smooth = np.zeros_like(x)
    x_smooth[0] = x[0]

    for t in range(1, len(x)):
        x_smooth[t] = alpha * x_smooth[t - 1] + (1 - alpha) * x[t]

    return x_smooth



@calTimes(logger, "窗口分布状态向量 e(t), Δe(t) 构建完成")
def buildEntropyStateVector(
    input_dir: str = cfg.ENTROPY["file_path"],
    output_dir: str = cfg.DISTRIBUTION["file_path"] ) -> None:
    """
    @description 从指定目录加载雷尼熵数据，使用编码器对数据进行处理，生成平滑后的雷尼熵状态向量 e(t) 和其变化量 Δe(t)
    @param input_dir: 输入数据的目录，包含网络流量的雷尼熵特征文件
    @param output_dir: 输出目录，用于保存生成的 e(t) 和 Δe(t) 文件
    @return: {None}
    """      

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.COMMON["seed"])

    os.makedirs(output_dir, exist_ok=True)

    encoder_path = os.path.join(output_dir, "entropy_encoder.pth")

 
    # 自动推断输入维度
    sample_file = None
    for f in os.listdir(input_dir):
        if f.endswith("_E.npy"):
            sample_file = f
            break

    if sample_file is None:
        raise RuntimeError("No entropy files found")

    sample_data = np.load(os.path.join(input_dir, sample_file))
    input_dim = sample_data.shape[1]

    d_e = cfg.DISTRIBUTION["entropy_embed_dim"]

    encoder = EntropyEncoder(input_dim, d_e).to(device)

    encoder_path = os.path.join(output_dir, "entropy_encoder.pth")

    if os.path.exists(encoder_path):
        state_dict = torch.load(encoder_path, map_location=device)

        try:
            encoder.load_state_dict(state_dict)
            print("[INFO] Loaded encoder (new architecture)")
        except RuntimeError:
            print("[INFO] Converting old encoder weights...")

            new_state_dict = encoder.state_dict()


            if "linear.weight" in state_dict:
                new_state_dict["net.0.weight"] = state_dict["linear.weight"]
                new_state_dict["net.0.bias"] = state_dict["linear.bias"]

            encoder.load_state_dict(new_state_dict, strict=False)
            print("[INFO] Loaded encoder (converted from old architecture)")

    encoder.eval()


    # 遍历文件
    for filename in os.listdir(input_dir):

        if not filename.endswith("_E.npy"):
            continue

        print(f"[Processing] {filename}")

        path = os.path.join(input_dir, filename)

        H = np.load(path).astype(np.float32)   


        # 输入归一化
        H = (H - np.mean(H, axis=0)) / (np.std(H, axis=0) + 1e-6)

        H_tensor = torch.from_numpy(H).to(device)


        # e(t)
        with torch.no_grad():
            e = encoder(H_tensor).cpu().numpy()   


        # EMA 平滑
        e_smooth = emaSmooth(e, alpha=0.9)


        # Δe(t)
        de = np.zeros_like(e_smooth)
        de[1:] = e_smooth[1:] - e_smooth[:-1]


        # 再标准化
        e_smooth = (e_smooth - np.mean(e_smooth, axis=0)) / (np.std(e_smooth, axis=0) + 1e-6)
        de = (de - np.mean(de, axis=0)) / (np.std(de, axis=0) + 1e-6)


        # 保存
        base_name = filename.replace("_E.npy", "")

        save_e = os.path.join(output_dir, base_name + "_e.npy")
        save_de = os.path.join(output_dir, base_name + "_de.npy")

        np.save(save_e, e_smooth.astype(np.float32))
        np.save(save_de, de.astype(np.float32))

        print(f"[Saved] {base_name} -> e:{e_smooth.shape}, de:{de.shape}")

    # 保存 encoder
    torch.save(encoder.state_dict(), encoder_path)