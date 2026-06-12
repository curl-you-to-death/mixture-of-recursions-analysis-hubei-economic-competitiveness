# ==============================================================
#输入比例调整实验
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
input_dim = 22
dim = 128
num_heads = 4
num_recursions = 3
mlp_ratio = 4
capacity_ratios = [1.0, 2 / 3, 1 / 3]
lr = 1e-4
epochs = 300
run_times = 20

# ======================== CPU 保护 ========================
torch.set_num_threads(4)
torch.set_num_interop_threads(4)


# ======================== 数据加载 + 指标比例调整 ========================
def load_data():
    X = pd.read_csv("input_72.csv").iloc[:, 2:].values
    y = pd.read_csv("output_18.csv").iloc[:, 2].values

    # 政策支持：×1.2
    policy_up = [5, 7, 21, 11]  # 财政预算收入、实际外商直接投资、财政科学技术支出、高等教育在校人数
    # 外部冲击：×0.8
    shock_down = [16, 18, 12, 13]  # 旅游人数、城乡人口总数、城镇居民人均可支配收入、人均金融机构存款余额

    X[:, policy_up] *= 1.2
    X[:, shock_down] *= 0.8

    X = X.reshape(18, 4, 22)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


X, y = load_data()
y_true = y.cpu().numpy()


# ======================== MoR 模型  ========================
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
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

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
        knum = max(1, int(x.shape[1] * ratio))
        idx = torch.topk(s, knum, dim=1).indices
        mask = torch.zeros_like(s)
        mask.scatter_(1, idx, 1.0)
        return x * mask


class MoR_Complete(nn.Module):
    def __init__(self):
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
def run_one_round():
    model = MoR_Complete()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        pred = model(X)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        time.sleep(0.005)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Loss = {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        pred = model(X)

    rmse = np.sqrt(mean_squared_error(y_true, pred.cpu().numpy()))
    mae = mean_absolute_error(y_true, pred.cpu().numpy())
    r2 = r2_score(y_true, pred.cpu().numpy())

    del model, opt, pred
    gc.collect()
    time.sleep(0.01)

    return rmse, mae, r2


# ======================== 主运行：20轮 ========================
if __name__ == "__main__":
    print("======= 输入比例调整实验（20轮平均） =======")
    rmses, maes, r2s = [], [], []

    for i in range(run_times):
        print(f"\n==============================================")
        print(f"输入比例调整 | 第 {i + 1}/{run_times} 次训练")
        print(f"==============================================")
        rm, ma, r2 = run_one_round()
        rmses.append(rm)
        maes.append(ma)
        r2s.append(r2)

    # ======================== 最终输出指标 ========================
    print("\n==============================================")
    print("     输入比例调整后预测指标（20次平均）")
    print("==============================================")
    print(f"平均RMSE: {np.mean(rmses):.4f}")
    print(f"平均MAE:  {np.mean(maes):.4f}")
    print(f"平均R²:   {np.mean(r2s):.4f}")