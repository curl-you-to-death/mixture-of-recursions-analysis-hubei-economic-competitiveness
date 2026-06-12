import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import gc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

input_dim       = 22
dim             = 128
num_heads       = 4
num_recursions  = 3
mlp_ratio       = 4
capacity_ratios = [1.0, 2/3, 1/3]
lr              = 1e-4
epochs          = 300
run_times       = 20

def load_data():
    X = pd.read_csv("input_72.csv").iloc[:, 2:].values
    y = pd.read_csv("output_18.csv").iloc[:, 2].values
    X = X.reshape(18, 4, 22)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X, y = load_data()
y_true = y.cpu().numpy()

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, cached_k=None, cached_v=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if cached_k is not None and cached_v is not None:
            k, v = cached_k, cached_v
        else:
            k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = (attn_probs @ v).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        return out, k, v

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x, cached_k=None, cached_v=None):
        attn_out, k, v = self.attn(self.norm1(x), cached_k, cached_v)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, k, v

class ExpertChoiceRouter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score_fc = nn.Linear(dim, 1)

    def forward(self, x, ratio):
        B, T, C = x.shape
        score = torch.sigmoid(self.score_fc(x))
        keep_num = max(1, int(T * ratio))
        topk_idx = torch.topk(score, keep_num, dim=1).indices
        mask = torch.zeros_like(score)
        mask.scatter_(1, topk_idx, 1.0)
        active_x = x * mask
        return active_x

class MoR_Complete(nn.Module):
    def __init__(self, attn_layers=1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, dim)
        self.first_blocks = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_ratio) for _ in range(attn_layers)])
        self.rec_block = TransformerBlock(dim, num_heads, mlp_ratio)
        self.last_blocks = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_ratio) for _ in range(attn_layers)])
        self.router = ExpertChoiceRouter(dim)
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.first_blocks:
            x, _, _ = block(x)
        cached_k, cached_v = None, None
        for i in range(num_recursions):
            x = self.router(x, capacity_ratios[i])
            if i == 0:
                x, cached_k, cached_v = self.rec_block(x)
            else:
                x, _, _ = self.rec_block(x, cached_k, cached_v)
        for block in self.last_blocks:
            x, _, _ = block(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze()

def run_one_model(attn_layers):
    model = MoR_Complete(attn_layers=attn_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        pred = model(X)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_scores = model(X).cpu().numpy()

    rmse = np.sqrt(mean_squared_error(y_true, pred_scores))
    mae = mean_absolute_error(y_true, pred_scores)
    r2 = r2_score(y_true, pred_scores)

    del model, opt
    gc.collect()
    return round(float(rmse),4), round(float(mae),4), round(float(r2),4)

def get_attention_layers_result():
    print("\n===== 注意力层数对比 开始20轮重复训练（每轮300次）=====\n")
    res1 = []
    print("==================== 1 层注意力 ====================")
    for i in range(run_times):
        rmse, mae, r2 = run_one_model(1)
        res1.append((rmse, mae, r2))
        print(f"第{i+1}轮完成 | RMSE:{rmse:.4f} | MAE:{mae:.4f} | R²:{r2:.4f}")

    res2 = []
    print("\n==================== 2 层注意力 ====================")
    for i in range(run_times):
        rmse, mae, r2 = run_one_model(2)
        res2.append((rmse, mae, r2))
        print(f"第{i+1}轮完成 | RMSE:{rmse:.4f} | MAE:{mae:.4f} | R²:{r2:.4f}")

    res3 = []
    print("\n==================== 3 层注意力 ====================")
    for i in range(run_times):
        rmse, mae, r2 = run_one_model(3)
        res3.append((rmse, mae, r2))
        print(f"第{i+1}轮完成 | RMSE:{rmse:.4f} | MAE:{mae:.4f} | R²:{r2:.4f}")

    m1 = np.mean(res1, axis=0)
    m2 = np.mean(res2, axis=0)
    m3 = np.mean(res3, axis=0)

    return {
        "layer1": {"rmse": round(float(m1[0]),4), "mae": round(float(m1[1]),4), "r2": round(float(m1[2]),4)},
        "layer2": {"rmse": round(float(m2[0]),4), "mae": round(float(m2[1]),4), "r2": round(float(m2[2]),4)},
        "layer3": {"rmse": round(float(m3[0]),4), "mae": round(float(m3[1]),4), "r2": round(float(m3[2]),4)},
    }