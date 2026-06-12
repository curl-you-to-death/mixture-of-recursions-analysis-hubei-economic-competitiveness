# ==============================================================
# 区域经济竞争力预测 - MoR 模型
# 仅新增：训练完成后返回4位小数得分的接口函数
# ==============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# ===================== 论文超参数 =====================
input_dim = 22
dim = 128
num_heads = 4
num_recursions = 3
mlp_ratio = 4
capacity_ratios = [1.0, 2 / 3, 1 / 3]
lr = 1e-4
epochs = 300


# ===================== 1. 多头注意力层 =====================
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


# ===================== 2. Transformer 块 =====================
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x, cached_k=None, cached_v=None):
        attn_out, k, v = self.attn(self.norm1(x), cached_k, cached_v)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, k, v


# ===================== 3. Expert-Choice 路由 =====================
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


# ===================== 4. MoR 模型 =====================
class MoR_Economic(nn.Module):
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
        cached_k, cached_v = None, None
        for i in range(num_recursions):
            x = self.router(x, capacity_ratios[i])
            if i == 0:
                x, cached_k, cached_v = self.rec_block(x)
            else:
                x, _, _ = self.rec_block(x, cached_k, cached_v)
        x, _, _ = self.last_block(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze()


# ===================== 5. 数据加载 =====================
def load_data():
    X = pd.read_csv("input_72.csv").iloc[:, 2:].values
    y = pd.read_csv("output_18.csv").iloc[:, 2].values
    X = X.reshape(18, 4, 22)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ===================== 6. 前端调用函数：训练300轮+返回4位小数得分 =====================
def get_mor_prediction():
    X, y = load_data()
    model = MoR_Economic()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("==== MoR 模型训练开始 ====")
    for epoch in range(epochs):
        model.train()
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss = {loss.item():.6f}")

    # 最终预测 + 强制保留4位小数
    model.eval()
    with torch.no_grad():
        final_scores = model(X).cpu().numpy()

    # 严格保留4位小数，和你的运行结果完全一致
    return [round(float(score), 4) for score in final_scores]


# ===================== 7. 本地测试 =====================
if __name__ == "__main__":
    scores = get_mor_prediction()
    print("\n湖北省18地区经济竞争力预测结果")
    for i, s in enumerate(scores):
        print(f"地区 {i + 1} : {s}")