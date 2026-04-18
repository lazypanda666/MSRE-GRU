#!/usr/bin/env python
# coding=utf-8
"""
@Description   only-X GRU 模型构建
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


def loadSequenceData(split="train"):
    """
    @description 导入数据集
    @param split: 数据集划分标识
    @return X: 输入特征矩阵
    @return y: 标签
    """    
    X = np.load(os.path.join(cfg.DATA["file_path"], f"X_{split}.npy"))
    y = np.load(os.path.join(cfg.DATA["file_path"], f"y_{split}.npy"))
    return X, y


class GRUCell(nn.Module):
    """
    @description 定义了GRU单元
    """ 
    def __init__(self, input_dim, hidden_dim):
        """
        @description 初始化GRU单元
        @param input_dim: 输入特征的维度
        @param hidden_dim: 隐藏状态的维度
        """     
        super().__init__()

        self.hidden_dim = hidden_dim

        
        self.W_z = nn.Linear(input_dim, hidden_dim)
        self.W_r = nn.Linear(input_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim, hidden_dim)

        self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

        
        self.dropout = nn.Dropout(0.4)

    def forward(self, x_t, h_prev):
        """
        @description 执行单时间步的前向传播，计算新的隐藏状态
        @param x_t: 当前时刻的输入特征
        @param h_prev: 上一时刻的隐藏状态
        @return h: 当前时刻的隐藏状态
        """      

        z = torch.sigmoid(self.W_z(x_t) + self.U_z(h_prev))
        r = torch.sigmoid(self.W_r(x_t) + self.U_r(h_prev))

        h_tilde = torch.tanh(self.W_h(x_t) + self.U_h(r * h_prev))

        h = (1 - z) * h_prev + z * h_tilde

        
        h = self.dropout(h)

        return h



class GRUModel(nn.Module):
    """
    @description 定义整个GRU模型，使用GRU单元进行序列数据的处理
    """    
    def __init__(self, input_dim, hidden_dim, num_classes) -> None:
        """
        @description 初始化GRU模型
        @param input_dim: 输入特征的维度
        @param hidden_dim: 隐藏状态的维度
        @param num_classes: 类别数
        @return {None}
        """      
        super().__init__()

        
        hidden_dim = hidden_dim // 2

        self.cell = GRUCell(input_dim, hidden_dim)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        @description 前向传播，通过GRU单元处理序列数据。
        @param x: 输入的特征序列
        @return out: 分类结果
        """     

        B, T, _ = x.shape
        h = torch.zeros(B, self.cell.hidden_dim, device=x.device)

        
        for t in range(T):
            h = self.cell(x[:, t], h)

        
        h_final = h  

        h_final = self.dropout(h_final)

        out = self.fc(h_final)
        return out



def train(seed=cfg.COMMON["seed"]) -> None:
    """
    @description 训练GRU模型
    @param seed: 随机种子
    @return: {None}
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = loadSequenceData("train")

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    model = GRUModel(
        input_dim=X.shape[-1],
        hidden_dim=cfg.MODEL["hidden_dim"],
        num_classes=len(torch.unique(y))
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.MODEL["lr"] * 0.5,  
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

        logits = model(X)
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
    save_path = os.path.join(cfg.MODEL["file_path"], "GRU_only_x.pth")
    torch.save(model.state_dict(), save_path)

    print("Saved:", save_path)