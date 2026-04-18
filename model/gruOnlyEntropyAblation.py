#!/usr/bin/env python
# coding=utf-8
"""
@Description   only_Entropy GRU 模型构建
@Author        lazypanda666
@Date          2026-04-18 18:06:08
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from const import cfg
from sklearn.utils.class_weight import compute_class_weight
import random



def loadSequenceData(split="train"):
    """
    @description 加载特征和标签数据
    @param split: 数据集的划分
    @return E: 特征矩阵
    @return DE: 特征变化量矩阵
    @return y: 标签数组
    """
    E = np.load(os.path.join(cfg.DATA["file_path"], f"E_{split}.npy"))
    DE = np.load(os.path.join(cfg.DATA["file_path"], f"DE_{split}.npy"))
    y = np.load(os.path.join(cfg.DATA["file_path"], f"y_{split}.npy"))
    return E, DE, y



class EntropyGRUCell(nn.Module):
    """
    @description 定义基于熵的GRU单元
    @param entropy_dim: 输入特征的维度
    @param hidden_dim: 隐藏层维度
    """
    def __init__(self, entropy_dim, hidden_dim):
        """
        @description 初始化熵驱动的GRU单元，设置其参数。
        @param entropy_dim: 输入特征的维度
        @param hidden_dim: 隐藏层维度
        """       
        super().__init__()

        self.hidden_dim = hidden_dim


        self.W_z = nn.Linear(entropy_dim, hidden_dim)
        self.W_r = nn.Linear(entropy_dim, hidden_dim)
        self.W_h = nn.Linear(entropy_dim, hidden_dim)

        self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_h = nn.Linear(hidden_dim, hidden_dim, bias=False)


        self.dropout = nn.Dropout(0.3)

    def forward(self, e_t, de_t, h_prev):
        """
        @description 计算单时间步的隐藏状态更新。
        @param e_t: 当前时刻的输入特征
        @param de_t: 当前时刻的特征变化量
        @param h_prev: 上一时刻的隐藏状态
        @return h: 当前时刻的隐藏状态
        """

        x = e_t + 0.5 * de_t

        z = torch.sigmoid(self.W_z(x) + self.U_z(h_prev))
        r = torch.sigmoid(self.W_r(x) + self.U_r(h_prev))

        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r * h_prev))

        h = (1 - z) * h_prev + z * h_tilde

        return self.dropout(h)



class EntropyGRU(nn.Module):
    """
    @description 定义基于熵的GRU模型
    """
    def __init__(self, entropy_dim, hidden_dim, num_classes):
        """
        @description 初始化熵驱动的GRU模型。
        @param entropy_dim: 输入特征的维度
        @param hidden_dim: 隐藏层维度
        @param num_classes: 类别数
        """
        super().__init__()

        hidden_dim = hidden_dim // 2 

        self.cell = EntropyGRUCell(entropy_dim, hidden_dim)

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, e, de):
        """
        @description 执行前向传播过程
        @param e: 输入特征矩阵
        @param de: 特征变化量矩阵
        @return: 分类结果
        """

        B, T, _ = e.shape
        h = torch.zeros(B, self.cell.hidden_dim, device=e.device)

        H = []

        for t in range(T):
            h = self.cell(e[:, t], de[:, t], h)
            H.append(h)

        H = torch.stack(H, dim=1)

        
        score = self.attn(H)
        weight = torch.softmax(score, dim=1)
        h_final = torch.sum(weight * H, dim=1)

        h_final = self.dropout(h_final)

        return self.fc(h_final)



def train(seed=cfg.COMMON["seed"]) -> None:
    """
    @description 训练模型
    @param seed: 随机种子
    @return: None
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    E, DE, y = loadSequenceData("train")

    E = torch.tensor(E, dtype=torch.float32).to(device)
    DE = torch.tensor(DE, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    model = EntropyGRU(
        entropy_dim=E.shape[-1],
        hidden_dim=cfg.MODEL["hidden_dim"],
        num_classes=len(torch.unique(y))
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.MODEL["lr"] * 0.7,
        weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.MODEL["epochs"]
    )

    weights = compute_class_weight(
        "balanced",
        classes=np.unique(y.cpu().numpy()),
        y=y.cpu().numpy()
    )
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    for epoch in range(cfg.MODEL["epochs"]):

        model.train()

        logits = model(E, DE)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean().item()

        print(f"[Epoch {epoch+1}] Loss={loss.item():.4f} Acc={acc:.4f}")

    
    os.makedirs(cfg.MODEL["file_path"], exist_ok=True)
    save_path = os.path.join(cfg.MODEL["file_path"], "GRU_only_entropy.pth")
    torch.save(model.state_dict(), save_path)

    print("Saved:", save_path)