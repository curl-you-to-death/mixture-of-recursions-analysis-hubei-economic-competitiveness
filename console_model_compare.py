# ==============================================================
# RNN/LSTM/MoR对比实验
# ==============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ====================== 全局统一超参数 ======================
input_dim = 22
seq_len = 4
dim = 128
num_heads = 4
num_layers = 2
lr = 1e-4
epochs = 300
#repeat_times = 1  3个模型仅训练1轮，都能较好预测得分
num_recursions = 3
mlp_ratio = 4
capacity_ratios = [1.0, 2/3, 1/3]

# ====================== 数据加载 ======================
X = pd.read_csv("input_72.csv").iloc[:, 2:].values
y = pd.read_csv("output_18.csv").iloc[:, 2].values
X = X.reshape(18, 4, 22)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# ==============================================================================
# 完整版 MoR(参考mor.py)
# ==============================================================================
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
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
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

class FullMoR(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(input_dim, dim)
        self.first_block = TransformerBlock(dim, num_heads, mlp_ratio)
        self.rec_block = TransformerBlock(dim, num_heads, mlp_ratio)
        self.last_block = TransformerBlock(dim, num_heads, mlp_ratio)
        self.router = ExpertChoiceRouter(dim)
        self.out = nn.Linear(dim, 1)

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
        return self.out(x).squeeze()

# ====================== 完整版 LSTM（无Dropout，小样本专用）======================
# 结构：嵌入层 → 归一化 → 激活 → LSTM → 归一化 → 输出
class FullLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        # 输入嵌入层：将原始22维经济指标 → 映射到模型高维空间 dim=128
        self.emb = nn.Linear(input_dim, dim)

        # 输入特征归一化：稳定训练、加速收敛
        self.norm_in = nn.LayerNorm(dim)

        # LSTM 核心层（时序特征提取）
        # batch_first=True：输入形状为 [批次, 时间步, 维度]
        # dropout=0.0：小样本数据必须关闭，防止丢失有效信息
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,  # 关闭
        )

        # 输出特征归一化
        self.norm_out = nn.LayerNorm(dim)

        # 最终全连接层：输出1维经济竞争力预测分数
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):

        # 输入特征嵌入映射：低维指标 → 高维特征
        x = self.emb(x)

        # 输入归一化
        x = self.norm_in(x)

        # GELU激活函数：非线性特征转换
        x = F.gelu(x)

        # LSTM时序建模：输出所有时间步的特征
        out, _ = self.lstm(x)

        # 取最后一个时间步的特征作为全局时序表示
        feat = out[:, -1, :]

        # 输出特征归一化
        feat = self.norm_out(feat)

        # 终预测：输出经济竞争力得分
        return self.fc(feat).squeeze()

# ====================== 完整版 RNN（无Dropout，小样本专用）======================
# 与 LSTM 结构完全对称，保证对比实验公平性
class FullRNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 输入嵌入层：22维指标 → dim=128维特征
        self.emb = nn.Linear(input_dim, dim)

        # 输入特征归一化
        self.norm_in = nn.LayerNorm(dim)

        # RNN 核心层（基础时序模型）
        # 结构与 LSTM 保持一致，仅将 LSTM 替换为 RNN，保证对照公平
        self.rnn = nn.RNN(
            input_size=dim,
            hidden_size=dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,  # 关闭
        )

        # 输出特征归一化
        self.norm_out = nn.LayerNorm(dim)

        # 预测头：输出1维经济分数
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):

        # 特征嵌入映射
        x = self.emb(x)

        # 输入归一化
        x = self.norm_in(x)

        # GELU非线性激活
        x = F.gelu(x)

        # RNN时序特征提取
        out, _ = self.rnn(x)

        # 取最后一个时间步作为全局时序特征
        feat = out[:, -1, :]

        # 输出归一化
        feat = self.norm_out(feat)

        # 输出预测结果
        return self.fc(feat).squeeze()

# ====================== 统一训练预测函数 ======================
# 功能：为所有模型（RNN/LSTM/MoR）提供统一的训练与预测流程
# 保证对比实验公平：相同优化器、相同学习率、相同训练轮数
def train_and_predict(model, name):
    print(f"\n===== 训练 {name} =====")

    # 定义优化器：Adam 自适应优化器（所有模型统一使用）
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # 定义损失函数：均方误差损失（回归任务标准损失）
    loss_fn = nn.MSELoss()

    for e in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e % 50 == 0:
            print(f"{name} | Epoch {e:3d} | Loss {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        pred = model(X).cpu().numpy()
    return pred

# ====================== 运行模型 ======================
# 将真实标签从GPU张量转为numpy数组（统一格式计算指标）
y_true = y.cpu().numpy()

# 分别训练并预测三个对比模型
pred_rnn = train_and_predict(FullRNN(), "Full RNN")
pred_lstm = train_and_predict(FullLSTM(), "Full LSTM")
pred_mor = train_and_predict(FullMoR(), "Full MoR")

# ====================== 评价指标 ======================
# 功能：输入真实值和预测值，计算三大回归评价指标
# RMSE：均方根误差（误差越小越好）
# MAE：平均绝对误差（误差越小越好）
# R²：决定系数（越接近1越好，拟合优度）
def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# 分别计算三个模型的评价指标
rmse_rnn, mae_rnn, r2_rnn = metrics(y_true, pred_rnn)
rmse_lstm, mae_lstm, r2_lstm = metrics(y_true, pred_lstm)
rmse_mor, mae_mor, r2_mor = metrics(y_true, pred_mor)

# ====================== 输出指标表格 ======================
print("\n==============================================")
print("          模型量化对比结果")
print("==============================================")
print(f"RNN   | RMSE: {rmse_rnn:.4f} | MAE: {mae_rnn:.4f} | R²: {r2_rnn:.4f}")
print(f"LSTM  | RMSE: {rmse_lstm:.4f} | MAE: {mae_lstm:.4f} | R²: {r2_lstm:.4f}")
print(f"MoR   | RMSE: {rmse_mor:.4f} | MAE: {mae_mor:.4f} | R²: {r2_mor:.4f}")

# ====================== 输出18地区预测对比 ======================
print("\n==============================================")
print("        湖北省18地区经济竞争力预测对比")
print("==============================================")
for i in range(18):
    print(f"地区{i+1:2d} | 真实:{y_true[i]:.4f} | RNN:{pred_rnn[i]:.4f} | LSTM:{pred_lstm[i]:.4f} | MoR:{pred_mor[i]:.4f}")