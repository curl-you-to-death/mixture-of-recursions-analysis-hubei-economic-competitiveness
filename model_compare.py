import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# ======================== 限制CPU线程（防过热） ========================
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

# ====================== 全局超参数 ======================
input_dim = 22
seq_len = 4
dim = 128
num_heads = 4
num_layers = 2
lr = 1e-4
epochs = 300
repeat_times = 20   # 3个模型各训练20轮，每轮300次,算20轮的指标平均值
num_recursions = 3
mlp_ratio = 4
capacity_ratios = [1.0, 2 / 3, 1 / 3]


# ====================== 数据加载 ======================
def load_data():
    X = pd.read_csv("input_72.csv").iloc[:, 2:].values
    y = pd.read_csv("output_18.csv").iloc[:, 2].values
    X = X.reshape(18, 4, 22)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y


X, y = load_data()
y_true = y.cpu().numpy()


# ==============================================================================
# MoR 模型
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
        self.emb = nn.Linear(input_dim, dim)
        self.first_block = TransformerBlock(dim, num_heads, mlp_ratio)
        self.rec_block = TransformerBlock(dim, num_heads, mlp_ratio)
        self.last_block = TransformerBlock(dim, num_heads, mlp_ratio)
        self.router = ExpertChoiceRouter(dim)
        self.out = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.emb(x)
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


# ====================== LSTM/RNN 模型 ======================
class FullLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(input_dim, dim)
        self.norm_in = nn.LayerNorm(dim)
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.norm_out = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.emb(x)
        x = self.norm_in(x)
        x = F.gelu(x)
        out, _ = self.lstm(x)
        feat = out[:, -1, :]
        feat = self.norm_out(feat)
        return self.fc(feat).squeeze()


class FullRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(input_dim, dim)
        self.norm_in = nn.LayerNorm(dim)
        self.rnn = nn.RNN(
            input_size=dim,
            hidden_size=dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.norm_out = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.emb(x)
        x = self.norm_in(x)
        x = F.gelu(x)
        out, _ = self.rnn(x)
        feat = out[:, -1, :]
        feat = self.norm_out(feat)
        return self.fc(feat).squeeze()


# ====================== 统一训练预测函数 ======================
def train_one_round(model, model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_np = model(X).cpu().numpy()
    time.sleep(0.01)  # 仅保留极短休眠（10ms）
    return pred_np


# ====================== 指标计算 ======================
def calculate_metrics(true_v, pred_v):
    rmse = np.sqrt(mean_squared_error(true_v, pred_v))
    mae = mean_absolute_error(true_v, pred_v)
    r2 = r2_score(true_v, pred_v)
    return rmse, mae, r2


# ====================== 20轮训练求平均（逻辑完全不变） ======================
def get_avg_metrics(model_class, model_name):
    rmse_list = []
    mae_list = []
    r2_list = []

    print(f"\n===== {model_name} 开始20轮重复训练（每轮300次）=====")
    for i in range(repeat_times):
        model = model_class()
        pred = train_one_round(model, f"{model_name}-{i + 1}")
        rmse, mae, r2 = calculate_metrics(y_true, pred)
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        print(f"第{i + 1}轮完成 | RMSE:{rmse:.4f} | MAE:{mae:.4f} | R²:{r2:.4f}")

    avg_rmse = np.mean(rmse_list)
    avg_mae = np.mean(mae_list)
    avg_r2 = np.mean(r2_list)

    print(f"\n===== {model_name} 20轮平均指标 =====")
    print(f"平均RMSE:{avg_rmse:.4f} | 平均MAE:{avg_mae:.4f} | 平均R²:{avg_r2:.4f}")
    return avg_rmse, avg_mae, avg_r2


# ===================== Flask 接口 =====================
def get_compare_result():
    rnn_rmse, rnn_mae, rnn_r2 = get_avg_metrics(FullRNN, "Full RNN")
    lstm_rmse, lstm_mae, lstm_r2 = get_avg_metrics(FullLSTM, "Full LSTM")
    mor_rmse, mor_mae, mor_r2 = get_avg_metrics(FullMoR, "Full MoR")

    return {
        "models": ["RNN", "LSTM", "MoR"],
        "rmse": [round(rnn_rmse, 4), round(lstm_rmse, 4), round(mor_rmse, 4)],
        "mae": [round(rnn_mae, 4), round(lstm_mae, 4), round(mor_mae, 4)],
        "r2": [round(rnn_r2, 4), round(lstm_r2, 4), round(mor_r2, 4)]
    }


# ===================== 主函数 =====================
if __name__ == "__main__":
    rnn_rmse, rnn_mae, rnn_r2 = get_avg_metrics(FullRNN, "Full RNN")
    lstm_rmse, lstm_mae, lstm_r2 = get_avg_metrics(FullLSTM, "Full LSTM")
    mor_rmse, mor_mae, mor_r2 = get_avg_metrics(FullMoR, "Full MoR")

    print("\n==============================================")
    print("          20轮×300次训练 平均指标结果")
    print("==============================================")
    print(f"RNN   | RMSE: {rnn_rmse:.4f} | MAE: {rnn_mae:.4f} | R²: {rnn_r2:.4f}")
    print(f"LSTM  | RMSE: {lstm_rmse:.4f} | MAE: {lstm_mae:.4f} | R²: {lstm_r2:.4f}")
    print(f"MoR   | RMSE: {mor_rmse:.4f} | MAE: {mor_mae:.4f} | R²: {mor_r2:.4f}")