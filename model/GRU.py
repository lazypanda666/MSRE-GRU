import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from const import cfg

import random
    

# 1. 数据处理
def load_entropy_sequence(path, T):
    data = np.load(path)

    X = data[:, :-1]   # e(t)
    y = data[:, -1]

    # 构造序列
    N, D = X.shape
    new_N = N // T

    X = X[:new_N*T].reshape(new_N, T, D)
    y = y[:new_N*T:T]

    # Δe(t)
    delta_e = np.zeros_like(X)
    delta_e[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]

    return X, delta_e, y


# 2. EC-GRU Cell
class ECGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, entropy_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        # GRU
        self.W_z = nn.Linear(input_dim, hidden_dim)
        self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_r = nn.Linear(input_dim, hidden_dim)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_h = nn.Linear(input_dim, hidden_dim)
        self.U_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # γ
        self.P_z = nn.Linear(entropy_dim, hidden_dim)
        self.P_r = nn.Linear(entropy_dim, hidden_dim)
        self.P_h = nn.Linear(entropy_dim, hidden_dim)

        # β
        self.Q_z = nn.Linear(entropy_dim, hidden_dim)
        self.Q_r = nn.Linear(entropy_dim, hidden_dim)
        self.Q_h = nn.Linear(entropy_dim, hidden_dim)

    def forward(self, x_t, h_prev, e_t, delta_e_t):
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


# 3. EC-GRU Model
class ECGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, entropy_dim, num_classes):
        super().__init__()

        self.cell = ECGRUCell(input_dim, hidden_dim, entropy_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, e, delta_e):
        B, T, _ = x.shape
        h = torch.zeros(B, self.cell.hidden_dim, device=x.device)

        for t in range(T):
            h = self.cell(
                x[:, t, :],
                h,
                e[:, t, :],
                delta_e[:, t, :]
            )

        out = self.fc(h)
        return out


# 4. 训练函数
def train(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = os.path.join(cfg.DATA["file_path"], "train.npy")

    X, delta_e, y = load_entropy_sequence(
        train_path,
        cfg.MODEL["seq_len"]
    )

    X = torch.tensor(X, dtype=torch.float32).to(device)
    delta_e = torch.tensor(delta_e, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    model = ECGRU(
        input_dim=X.shape[-1],
        hidden_dim=cfg.MODEL["hidden_dim"],
        entropy_dim=X.shape[-1],
        num_classes=len(torch.unique(y))
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.MODEL["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.MODEL["epochs"]):
        model.train()

        logits = model(X, X, delta_e)  # x = e
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")

    # 保存模型
    os.makedirs(cfg.MODEL["file_path"], exist_ok=True)
    save_path = os.path.join(cfg.MODEL["file_path"], "ecgru.pth")
    torch.save(model.state_dict(), save_path)

    print(f"Model saved to: {save_path}")