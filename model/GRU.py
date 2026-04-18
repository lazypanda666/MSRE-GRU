#!/usr/bin/env python
# coding=utf-8
"""
@Description   
@Author        lazypanda666
@Date          2026-04-15 21:46:35
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from const import cfg
from sklearn.utils.class_weight import compute_class_weight
import random


def loadSequenceData(split = "train"):
    """
    @description 从指定路径加载序列建模所需的数据
    @param split: 数据划分标识（"train" / "test" / "val"）
    @return:X: 网络流量特征序列
            E: 熵嵌入序列 e(t)
            DE: 熵变化量序列 Δe(t)
            y: 标签，形状 [N]
    """
    X = np.load(os.path.join(cfg.DATA["file_path"], f"X_{split}.npy"))
    E = np.load(os.path.join(cfg.DATA["file_path"], f"E_{split}.npy"))
    DE = np.load(os.path.join(cfg.DATA["file_path"], f"DE_{split}.npy"))
    y = np.load(os.path.join(cfg.DATA["file_path"], f"y_{split}.npy"))

    return X, E, DE, y



class MSREGRUCell(nn.Module):
    """
    @description MSRE-GRU单元
    """
    def __init__(self, input_dim, hidden_dim, entropy_dim) -> None:
        """
        @param input_dim: 输入特征维度 d_x
        @param hidden_dim: 隐状态维度 d_h
        @param entropy_dim: 熵嵌入维度 d_e
        @return: {None}
        """

        super().__init__()

        self.hidden_dim = hidden_dim

        # GRU
        self.W_z = nn.Linear(input_dim, hidden_dim)
        self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_r = nn.Linear(input_dim, hidden_dim)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_h = nn.Linear(input_dim, hidden_dim)
        self.U_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # γ（缩放）
        self.P_z = nn.Sequential(
            nn.Linear(entropy_dim, hidden_dim),
            nn.Tanh()
        )
        self.P_r = nn.Sequential(
            nn.Linear(entropy_dim, hidden_dim),
            nn.Tanh()
        )
        self.P_h = nn.Sequential(
            nn.Linear(entropy_dim, hidden_dim),
            nn.Tanh()
        )

        # β（偏置）
        self.Q_z = nn.Linear(entropy_dim, hidden_dim)
        self.Q_r = nn.Linear(entropy_dim, hidden_dim)
        self.Q_h = nn.Linear(entropy_dim, hidden_dim)

        # 初始化
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x_t, h_prev, e_t, delta_e_t) -> torch.Tensor:
        """
        @description 单时间步前向传播
        @param x_t: 当前输入
        @param h_prev: 上一时刻隐状态
        @param e_t: 当前熵嵌入
        @param delta_e_t: 当前熵变化量
        @return: {torch.Tensor}
        """

        gamma_z = torch.sigmoid(self.P_z(delta_e_t))
        gamma_r = torch.sigmoid(self.P_r(delta_e_t))
        gamma_h = torch.sigmoid(self.P_h(delta_e_t))

        beta_z = self.Q_z(e_t)
        beta_r = self.Q_r(e_t)
        beta_h = self.Q_h(e_t)

        z_t = torch.sigmoid(
            gamma_z * (self.W_z(x_t) + self.U_z(h_prev)) + beta_z
        )

        r_t = torch.sigmoid(
            gamma_r * (self.W_r(x_t) + self.U_r(h_prev)) + beta_r
        )

        h_tilde = torch.tanh(
            gamma_h * (
                self.W_h(x_t) + self.U_h(r_t * h_prev)
            ) + beta_h
        )

        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        return h_t


class MSREGRU(nn.Module):
    """
    @description MSRE-GRU 时序建模模型。
    @param input_dim: 输入特征维度
    @param hidden_dim: GRU 隐状态维度
    @param entropy_dim: 熵特征维度
    @param num_classes: 分类类别数
    """
    def __init__(self, input_dim, hidden_dim, entropy_dim, num_classes) ->{None}:
        """
        @description 初始化 MSRE-GRU 模型
        @return: {None}
        """
        super().__init__()

        self.cell = MSREGRUCell(input_dim, hidden_dim, entropy_dim)

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, e, delta_e) -> torch.Tensor:
        """
        @description 模型前向传播过程。
        @param x: 输入序列
        @param e: 熵嵌入序列
        @param delta_e: 熵变化量序列
        @return: {torch.Tensor}
        """
        B, T, _ = x.shape
        h = torch.zeros(B, self.cell.hidden_dim, device=x.device)

        h_list = []

        for t in range(T):
            h = self.cell(
                x[:, t, :],
                h,
                e[:, t, :],
                delta_e[:, t, :]
            )
            h_list.append(h)

        H = torch.stack(h_list, dim=1)  

        # Attention pooling
        attn_score = self.attn(H)             
        attn_weight = torch.softmax(attn_score, dim=1)
        h_final = torch.sum(attn_weight * H, dim=1)

        out = self.fc(h_final)
        return out


def train(seed: int = cfg.COMMON["seed"]) -> None:
    """
    @description MSRE-GRU 模型训练流程
    @param seed: 随机种子
    @return: {None}
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, E, DE, y = loadSequenceData("train")

    X = torch.tensor(X, dtype=torch.float32).to(device)
    E = torch.tensor(E, dtype=torch.float32).to(device)
    DE = torch.tensor(DE, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    model = MSREGRU(
        input_dim=X.shape[-1],
        hidden_dim=cfg.MODEL["hidden_dim"],
        entropy_dim=E.shape[-1],
        num_classes=len(torch.unique(y))
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.MODEL["lr"], weight_decay=1e-4)

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

        logits = model(X, E, DE)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()
        scheduler.step()

        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean().item()

        print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    # 保存模型
    os.makedirs(cfg.MODEL["file_path"], exist_ok=True)
    save_path = os.path.join(cfg.MODEL["file_path"], "ecgru.pth")
    torch.save(model.state_dict(), save_path)

    print(f"Model saved to: {save_path}")
