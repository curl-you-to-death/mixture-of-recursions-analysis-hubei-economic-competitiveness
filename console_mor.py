# ==============================================================
# 湖北省区域经济竞争力预测 - MoR 模型
# 实现三大核心机制：
# 1. Middle-Cycle 权重共享（首尾独立、中间递归共享）
# 2. Expert-Choice 动态路由（逐层筛选Token）
# 3. Recursive Sharing KV 缓存（所有递归复用第1轮KV）
# ==============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# ==============================================
# 一、全局超参数定义
# ==============================================
input_dim       = 22        # 输入特征维度：22个经济指标
dim             = 128       # 模型嵌入维度（特征映射维度）
num_heads       = 4         # 多头注意力头数
num_recursions  = 3         # 递归次数 Nr=3（论文核心参数）
mlp_ratio       = 4         # MLP 中间层放大倍数
capacity_ratios = [1.0, 2/3, 1/3]  # 三层递归的容量衰减比例
lr              = 1e-4      # 学习率
epochs          = 300       # 每轮训练迭代次数

# ==============================================
# 二、核心模块1：多头注意力机制
# 功能：提取时空特征、捕捉指标间的依赖关系
# ==============================================
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads          # 注意力头数
        self.head_dim = dim // num_heads    # 每个头的维度

        # Q/K/V 线性映射
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # 输出投影层
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, cached_k=None, cached_v=None):

        # x：输入特征 [批次, 时间步, 维度]
        B, T, C = x.shape

        # Q 线性投影 + 多头拆分
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # ===========================
        # 核心创新：KV 缓存机制（Recursive Sharing）
        # 第一次递归计算 KV，后续递归直接复用
        # ===========================
        if cached_k is not None and cached_v is not None:
            k, v = cached_k, cached_v
        else:
            k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力分数计算 + softmax 归一化
        attn_scores = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 注意力输出 + 多头拼接
        out = (attn_probs @ v).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)

        # 返回：注意力结果、缓存的K、缓存的V
        return out, k, v

# ==============================================
# 三、核心模块2：Transformer 块
# 结构：LayerNorm → 多头注意力 → 残差 → MLP
# ==============================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)      # 第一层归一化
        self.attn = MultiHeadAttention(dim, num_heads)  # 注意力
        self.norm2 = nn.LayerNorm(dim)      # 第二层归一化

        # MLP 模块：升维 → GELU激活 → 降维
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x, cached_k=None, cached_v=None):

        # 注意力计算（带KV缓存）
        attn_out, k, v = self.attn(self.norm1(x), cached_k, cached_v)

        # 残差连接1
        x = x + attn_out

        # MLP + 残差连接2
        x = x + self.mlp(self.norm2(x))

        return x, k, v

# ==============================================
# 四、核心模块3：Expert-Choice Router（专家选择路由）
# 功能：根据重要性分数，动态保留一定比例的关键特征(token)
# ==============================================
class ExpertChoiceRouter(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # 线性层：为每个特征位置计算一个重要性得分
        self.score_fc = nn.Linear(dim, 1)

    def forward(self, x, ratio):

        # x：输入特征图，形状 [B, T, C]
        # B：批次大小（地区数量）
        # T：时间步/特征长度
        # C：特征维度
        B, T, C = x.shape

        # 对每个特征位置计算重要性分数，并通过sigmoid归一化到0~1
        score = torch.sigmoid(self.score_fc(x))

        # 计算需要保留的特征数量（根据传入的ratio比例）
        # 至少保留1个，防止维度为0
        keep_num = max(1, int(T * ratio))

        # 选取分数最高的 top-k 个特征的索引
        topk_idx = torch.topk(score, keep_num, dim=1).indices

        # 构造掩码矩阵：保留的位置为1，其余为0
        mask = torch.zeros_like(score)
        mask.scatter_(1, topk_idx, 1.0)

        # 掩码过滤：只保留高分特征，其余置0
        active_x = x * mask

        # 返回筛选后的活跃特征
        return active_x

# ==============================================
# 五、MoR 模型总架构
# ==============================================
class MoR_Economic(nn.Module):
    def __init__(self):
        super().__init__()

        # 输入嵌入层：将原始22维指标映射到模型维度
        self.embedding = nn.Linear(input_dim, dim)

        # ============ Middle-Cycle 结构 ============
        self.first_block = TransformerBlock(dim, num_heads, mlp_ratio)   # 独立输入层
        self.rec_block = TransformerBlock(dim, num_heads, mlp_ratio)     # 共享递归层
        self.last_block = TransformerBlock(dim, num_heads, mlp_ratio)    # 独立输出层

        # 专家选择路由模块
        self.router = ExpertChoiceRouter(dim)

        # 最终预测全连接层
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):

        # 输入特征嵌入变换
        x = self.embedding(x)

        # 首层独立 Transformer 编码
        x, _, _ = self.first_block(x)

        # ===========================
        # 核心：Recursive Sharing KV 缓存
        # 第 1 轮生成 KV，后续直接复用
        # ===========================
        cached_k, cached_v = None, None

        # 开始递归循环（共 num_recursions 轮）
        for i in range(num_recursions):

            # 专家路由：根据容量比例动态筛选活跃特征
            x = self.router(x, capacity_ratios[i])

            # ===========================
            # 递归共享：仅第1次计算KV并缓存，后续复用
            # ===========================
            if i == 0:
                x, cached_k, cached_v = self.rec_block(x)
            else:
                x, _, _ = self.rec_block(x, cached_k, cached_v)

        # 末层独立 Transformer 精炼特征
        x, _, _ = self.last_block(x)

        # 全局平均池化 + 全连接输出预测得分
        x = x.mean(dim=1)
        return self.fc(x).squeeze()

# ==============================================
# 六、数据加载模块
# 功能：读取输入输出CSV，构建模型需要的张量格式
# 输入：input_72.csv  (18地区 × 4时段 × 22指标)
# 输出：output_18.csv (18地区经济竞争力得分)
# ==============================================
def load_data():

    # 读取输入特征（从第3列开始是有效指标）
    X = pd.read_csv("input_72.csv").iloc[:, 2:].values

    # 读取预测目标（经济竞争力得分）
    y = pd.read_csv("output_18.csv").iloc[:, 2].values

    # 形状转换：[18地区, 4时间步, 22指标] → 适配模型输入
    X = X.reshape(18, 4, 22)

    # 转为 PyTorch 张量
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ==============================================
# 七、训练和预测
# ==============================================
if __name__ == "__main__":

    # 加载数据集（输入特征 + 真实标签）
    X, y = load_data()

    # 初始化 MoR 模型
    model = MoR_Economic()

    # 优化器：Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 损失函数：回归任务使用均方误差损失 MSELoss
    criterion = nn.MSELoss()

    # 模型训练：共训练 epochs=300 次
    print("==== MoR 模型训练开始 ====")
    for epoch in range(epochs):

        # 切换为训练模式
        model.train()

        # 前向传播：输入数据得到预测结果
        pred = model(X)

        # 计算预测值与真实值之间的损失
        loss = criterion(pred, y)

        # 梯度清零 + 反向传播 + 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每 50 轮打印一次训练损失
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss = {loss.item():.6f}")

    # 模型预测输出
    model.eval()                    # 切换为评估模式（不训练）
    with torch.no_grad():           # 关闭梯度计算
        final_scores = model(X)     # 只做预测，不学习

    # 输出 18 个地区经济竞争力预测结果（保留 4 位小数）
    print("\n湖北省18地区经济竞争力预测结果")
    for i in range(18):
        print(f"地区 {i+1} : {final_scores[i].item():.4f}")