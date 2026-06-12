# ==============================================================
# 输入指标维度对比实验
# ==============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
import gc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ======================== 维度划分 ========================
DIMENSION_CONFIG = {
    "经济实力": [0,1,2,9,10,18,19],
    "产业结构": [3,4,14,15],
    "创新投入": [21],
    "对外开放": [7,8,16,17],
    "人才支撑": [11,12,20],
    "财政保障": [5,6,13]
}

# ======================== 超参数 ========================
dim             = 128
num_heads       = 4
num_recursions  = 3
mlp_ratio       = 4
capacity_ratios = [1.0, 2/3, 1/3]
lr              = 1e-4
epochs          = 300
run_times       = 20

# ======================== CPU 保护 ========================
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

# ======================== 数据加载 ========================
def load_data():
    X = pd.read_csv("input_72.csv").iloc[:, 2:].values
    y = pd.read_csv("output_18.csv").iloc[:, 2].values
    return X, y

X_all, y = load_data()
y_tensor = torch.tensor(y, dtype=torch.float32)

# ======================== MoR 模型 100% 复制 mor4 ========================
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
        return self.out_proj(out), k, v

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*mlp_ratio), nn.GELU(), nn.Linear(dim*mlp_ratio, dim))

    def forward(self, x, cached_k=None, cached_v=None):
        a, k, v = self.attn(self.norm1(x), cached_k, cached_v)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, k, v

class ExpertChoiceRouter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score_fc = nn.Linear(dim, 1)
    def forward(self, x, ratio):
        s = torch.sigmoid(self.score_fc(x))
        knum = max(1, int(x.shape[1]*ratio))
        idx = torch.topk(s, knum, dim=1).indices
        mask = torch.zeros_like(s)
        mask.scatter_(1, idx, 1.0)
        return x * mask

class MoR_Complete(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, dim)
        self.first_block = TransformerBlock(dim, num_heads, mlp_ratio)
        self.rec_block = TransformerBlock(dim, num_heads, mlp_ratio)
        self.last_block = TransformerBlock(dim, num_heads, mlp_ratio)
        self.router = ExpertChoiceRouter(dim)
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _, _ = self.first_block(x)
        ck, cv = None, None
        for i in range(num_recursions):
            x = self.router(x, capacity_ratios[i])
            if i == 0:
                x, ck, cv = self.rec_block(x)
            else:
                x, _, _ = self.rec_block(x, ck, cv)
        x, _, _ = self.last_block(x)
        return self.fc(x.mean(1)).squeeze()

# ======================== 单轮训练 ========================
# 功能：针对某一组输入特征维度，执行一次完整的模型训练与指标计算
# 参数：X_dim：当前维度配置对应的输入特征张量
# 返回：rmse, mae, r2 三大回归评估指标
def run_one_dimension(X_dim):

    # 初始化MoR模型：输入维度自动适配当前所选特征的维度
    model = MoR_Complete(input_dim=X_dim.shape[-1])

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        pred = model(X_dim)
        loss = criterion(pred, y_tensor)
        opt.zero_grad()
        loss.backward()
        opt.step()
        time.sleep(0.005)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Loss = {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        pred = model(X_dim)

    rmse = np.sqrt(mean_squared_error(y, pred.numpy()))
    mae = mean_absolute_error(y, pred.numpy())
    r2 = r2_score(y, pred.numpy())

    del model, opt, pred
    gc.collect()
    time.sleep(0.01)
    return rmse, mae, r2

# ======================== 主运行：20轮平均 ========================
if __name__ == "__main__":
    print("======= 输入指标维度对比实验（20轮平均） =======")
    final_result = {}

    # 遍历所有预设的维度配置（经济实力、产业结构等6大维度）
    for name, idx_list in DIMENSION_CONFIG.items():

        # 根据索引列表提取当前维度对应的原始特征
        X_sub = X_all[:, idx_list]

        # 重塑形状：[18地区, 4时间步, 特征维度]，适配模型输入格式
        X_sub = X_sub.reshape(18, 4, -1)

        # 转换为PyTorch张量
        X_tensor = torch.tensor(X_sub, dtype=torch.float32)

        print(f"\n==============================================")
        print(f"当前维度 = {name} | 第 1/{run_times} 次训练")
        print(f"==============================================")

        rmses, maes, r2s = [], [], []
        for t in range(run_times):
            rm, ma, r2 = run_one_dimension(X_tensor)
            rmses.append(rm)
            maes.append(ma)
            r2s.append(r2)

            # ========== 【第2~20轮也按统一格式打印】 ==========
            if t+1 < run_times:
                print(f"\n==============================================")
                print(f"当前维度 = {name} | 第 {t+2}/{run_times} 次训练")
                print(f"==============================================")

        final_result[name] = {
            "rmse": np.mean(rmses),
            "mae": np.mean(maes),
            "r2": np.mean(r2s)
        }

    # ======================== 最终输出 ========================
    print("\n==============================================")
    print("        各维度评价指标对比（20次平均）")
    print("==============================================")
    for name, res in final_result.items():
        print(f"{name:6s} | 平均RMSE: {res['rmse']:.4f} | 平均MAE: {res['mae']:.4f} | 平均R²: {res['r2']:.4f}")