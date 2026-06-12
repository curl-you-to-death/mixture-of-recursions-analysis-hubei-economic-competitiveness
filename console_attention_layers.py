# ==============================================================
# MoR 注意力层数1/2/3对比实验 | 20轮平均值
# ==============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
import gc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ======================== 超参数========================
input_dim       = 22
dim             = 128
num_heads       = 4
num_recursions  = 3
mlp_ratio       = 4
capacity_ratios = [1.0, 2/3, 1/3]
lr              = 1e-4
epochs          = 300
run_times       = 20

# ======================== CPU 温和模式 ========================
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

# ======================== 数据加载 ========================
def load_data():
    X = pd.read_csv("input_72.csv").iloc[:, 2:].values
    y = pd.read_csv("output_18.csv").iloc[:, 2].values
    X = X.reshape(18, 4, 22)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X, y = load_data()
y_true = y.cpu().numpy()

# ======================== MoR模型代码 ========================
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

# ======================== 训练函数 ========================
# 功能：针对不同注意力层数的 MoR 模型进行单次训练与评估
# 参数：
#   attn_layers: 当前使用的注意力层数（1/2/3层）
#   run_idx:    第几次重复运行（用于20轮取平均）
# 返回：
#   rmse, mae, r2: 模型在测试集上的三大评价指标
def run_one_model(attn_layers, run_idx):
    print(f"\n==============================================")
    print(f"注意力层数 = {attn_layers} | 第 {run_idx+1}/{run_times} 次训练")
    print(f"==============================================")

    # 初始化模型：根据传入的注意力层数构建对应 MoR 模型
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

        time.sleep(0.005)  # CPU 温和延时：防止训练速度过快导致CPU占用过高

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Loss = {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        pred_scores = model(X).cpu().numpy()    # 获取预测分数并转为numpy格式

    rmse = np.sqrt(mean_squared_error(y_true, pred_scores))
    mae = mean_absolute_error(y_true, pred_scores)
    r2 = r2_score(y_true, pred_scores)

    # 资源释放：删除模型与变量，回收内存，防止多次运行导致内存泄漏
    del model, opt, pred_scores
    gc.collect()

    time.sleep(0.01)

    return rmse, mae, r2

# ======================== 运行 20 轮 ========================
if __name__ == "__main__":
    print("==== MoR 注意力层数对比 | 20轮平均 | 高精度模式 ====")
    results = {1: [], 2: [], 3: []}

    for layer in [1, 2, 3]:
        for i in range(run_times):
            rmse, mae, r2 = run_one_model(layer, i)
            results[layer].append((rmse, mae, r2))

    # ======================== 输出最终平均结果 ========================
    print("\n==============================================")
    print("          注意力层数对比结果（20次平均）")
    print("==============================================")

    for layer in [1,2,3]:
        rmses = [x[0] for x in results[layer]]
        maes  = [x[1] for x in results[layer]]
        r2s   = [x[2] for x in results[layer]]
        print(f"层数{layer} | 平均RMSE: {np.mean(rmses):.4f} | 平均MAE: {np.mean(maes):.4f} | 平均R²: {np.mean(r2s):.4f}")