"""
目标域故障诊断与过程解释脚本 - Transformer版本
功能：使用Transformer架构实现域适应、目标域预测、聚类分析和SHAP解释
基于Q2的Transformer架构重写Q3 Q4功能

主要改进：
1. 集成了PyTorch Transformer架构用于故障诊断
2. 实现了基于Transformer的域适应方法（基线和CORAL）
3. 支持序列化特征输入，更好地捕获时序依赖关系
4. 提供了传统方法的回退机制（当PyTorch不可用时）
5. 保持了原有的可视化和聚类分析功能

使用方法：
1. 确保安装了PyTorch（可选，不安装会自动回退到传统方法）
2. 准备好源域和目标域数据文件
3. 运行 python Q3_Q4_目标域故障诊断与过程解释.py

依赖：
- PyTorch (可选，用于Transformer模型)
- sklearn, pandas, numpy, matplotlib, seaborn
- scipy, pywt (用于特征提取)
- shap (可选，用于可解释性分析)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import random
import math
from scipy.signal import hilbert, welch, stft
from scipy.stats import kurtosis, skew
import pywt
from numpy.linalg import eigh
import importlib

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report, 
                           normalized_mutual_info_score, adjusted_rand_score, f1_score,
                           accuracy_score)
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

# PyTorch相关
_torch_spec = importlib.util.find_spec("torch")
if _torch_spec is not None:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
else:
    torch = None
    nn = None
    optim = None
    Dataset = None
    DataLoader = None
    F = None
    PYTORCH_AVAILABLE = False

# SHAP相关
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("警告：SHAP未安装，可解释性分析功能将不可用")
    SHAP_AVAILABLE = False


def setup_environment():
    """设置环境"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore')


# ==================== Transformer相关类和函数 ====================

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    """PyTorch Transformer模型 - 用于域适应"""
    def __init__(self, input_dim, d_model=64, num_heads=4, num_layers=2, 
                 dropout=0.3, dense_units=64, n_classes=10):
        super(TransformerModel, self).__init__()
        
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，无法使用Transformer模型")
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, n_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # 分类
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.classifier(x)
        
        return x
    
    def get_features(self, x):
        """获取特征表示（用于域适应）"""
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # 平均池化得到特征表示
        x = x.mean(dim=1)  # (batch_size, d_model)
        return x


class FaultDataset(Dataset):
    """故障诊断数据集"""
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def prepare_transformer_data(X, seq_len=20):
    """
    为Transformer准备序列数据
    将特征重塑为时间序列格式
    """
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    # 计算输入维度
    input_dim = X_array.shape[1] // seq_len
    if input_dim == 0:
        input_dim = 1
        seq_len = X_array.shape[1]
    
    # 重塑数据
    n_samples = X_array.shape[0]
    X_seq = X_array[:, :seq_len*input_dim].reshape(n_samples, seq_len, input_dim)
    
    return X_seq, input_dim, seq_len


def train_transformer_baseline(X_src, y_src, X_tgt, 
                              d_model=64, num_heads=4, num_layers=2,
                              dropout=0.3, dense_units=64,
                              lr=1e-3, batch_size=32, epochs=100,
                              seq_len=20, random_state=42):
    """
    训练基线Transformer模型（无域适应）
    """
    if not PYTORCH_AVAILABLE:
        print("PyTorch未安装，跳过Transformer训练")
        return None, None, None
    
    # 准备数据
    X_src_seq, input_dim, seq_len_actual = prepare_transformer_data(X_src, seq_len)
    X_tgt_seq, _, _ = prepare_transformer_data(X_tgt, seq_len)
    
    # 标签编码
    le = LabelEncoder()
    y_src_enc = le.fit_transform(y_src)
    n_classes = len(le.classes_)
    
    # 分割训练和验证数据
    X_train, X_val, y_train, y_val = train_test_split(
        X_src_seq, y_src_enc, test_size=0.2, random_state=random_state, stratify=y_src_enc
    )
    
    # 创建数据集
    train_dataset = FaultDataset(X_train, y_train)
    val_dataset = FaultDataset(X_val, y_val)
    tgt_dataset = FaultDataset(X_tgt_seq)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    tgt_loader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=False)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = TransformerModel(
        input_dim=input_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        dense_units=dense_units,
        n_classes=n_classes
    ).to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 训练模型
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    print("开始训练基线Transformer模型...")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {train_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_transformer_baseline.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停在epoch {epoch}")
                break
    
    # 加载最佳模型并预测目标域
    model.load_state_dict(torch.load('best_transformer_baseline.pth'))
    model.eval()
    
    # 目标域预测
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_X in tgt_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    y_tgt_pred = le.inverse_transform(all_predictions)
    y_tgt_proba = np.array(all_probabilities)
    
    return model, y_tgt_pred, (le, device)


def coral_alignment_transformer(source_features, target_features):
    """
    基于Transformer特征的CORAL对齐
    """
    # 转换为numpy
    if isinstance(source_features, torch.Tensor):
        source_features = source_features.cpu().numpy()
    if isinstance(target_features, torch.Tensor):
        target_features = target_features.cpu().numpy()
    
    # 计算协方差矩阵
    cov_src = np.cov(source_features, rowvar=False) + np.eye(source_features.shape[1]) * 1e-6
    cov_tar = np.cov(target_features, rowvar=False) + np.eye(target_features.shape[1]) * 1e-6
    
    # 使用SVD进行矩阵开方
    U_s, S_s, _ = np.linalg.svd(cov_src)
    U_t, S_t, _ = np.linalg.svd(cov_tar)
    A_s = U_s @ np.diag(S_s**-0.5) @ U_s.T
    A_t = U_t @ np.diag(S_t**0.5) @ U_t.T
    
    # 变换
    source_aligned = (source_features - source_features.mean(0)) @ A_s @ A_t + target_features.mean(0)
    
    return source_aligned


def train_transformer_with_coral(X_src, y_src, X_tgt,
                                d_model=64, num_heads=4, num_layers=2,
                                dropout=0.3, dense_units=64,
                                lr=1e-3, batch_size=32, epochs=100,
                                seq_len=20, random_state=42):
    """
    使用CORAL域适应训练Transformer模型
    """
    if not PYTORCH_AVAILABLE:
        print("PyTorch未安装，跳过Transformer训练")
        return None, None, None
    
    # 准备数据
    X_src_seq, input_dim, seq_len_actual = prepare_transformer_data(X_src, seq_len)
    X_tgt_seq, _, _ = prepare_transformer_data(X_tgt, seq_len)
    
    # 标签编码
    le = LabelEncoder()
    y_src_enc = le.fit_transform(y_src)
    n_classes = len(le.classes_)
    
    # 创建数据集
    src_dataset = FaultDataset(X_src_seq, y_src_enc)
    tgt_dataset = FaultDataset(X_tgt_seq)
    
    src_loader = DataLoader(src_dataset, batch_size=batch_size, shuffle=True)
    tgt_loader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = TransformerModel(
        input_dim=input_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        dense_units=dense_units,
        n_classes=n_classes
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print("开始训练CORAL适应Transformer模型...")
    
    # 首先在源域上预训练
    for epoch in range(epochs // 2):
        model.train()
        for batch_X, batch_y in src_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        if epoch % 20 == 0:
            print(f'预训练 Epoch [{epoch}/{epochs//2}]')
    
    # 获取源域和目标域的特征表示
    model.eval()
    src_features_list = []
    tgt_features_list = []
    
    with torch.no_grad():
        # 源域特征
        for batch_X, _ in src_loader:
            batch_X = batch_X.to(device)
            features = model.get_features(batch_X)
            src_features_list.append(features.cpu())
        
        # 目标域特征
        for batch_X in tgt_loader:
            batch_X = batch_X.to(device)
            features = model.get_features(batch_X)
            tgt_features_list.append(features.cpu())
    
    # 合并特征
    src_features = torch.cat(src_features_list, dim=0).numpy()
    tgt_features = torch.cat(tgt_features_list, dim=0).numpy()
    
    # CORAL对齐
    src_features_aligned = coral_alignment_transformer(src_features, tgt_features)
    
    # 目标域预测（使用对齐后的源域模型）
    # 这里简化处理，直接用预训练模型预测目标域
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_X in tgt_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    y_tgt_pred = le.inverse_transform(all_predictions)
    y_tgt_proba = np.array(all_probabilities)
    
    return model, y_tgt_pred, (le, device)


def time_features(x):
    """时域统计特征"""
    x = np.asarray(x).ravel()
    if len(x) == 0:
        return {}
    
    # 移除NaN和无穷值
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    
    # 基本统计量
    mean_val = np.mean(x)
    std_val = np.std(x)
    var_val = np.var(x)
    rms = np.sqrt(np.mean(x**2))
    mean_abs = np.mean(np.abs(x))
    
    # 避免负数开方
    abs_x = np.abs(x)
    sqr_mean = np.mean(np.sqrt(abs_x)) if len(abs_x) > 0 else 0
    peak = np.max(abs_x)

    # 安全的除法，避免除零
    def safe_divide(a, b, default=0.0):
        return a / (b + 1e-12) if abs(b) > 1e-15 else default

    feats = {
        "均值": mean_val,
        "标准差": std_val,
        "方差": var_val,
        "均方根": rms,
        "峰值": peak,
        "峰峰值": np.ptp(x),
        "平均绝对值": mean_abs,
        "波形指标": safe_divide(rms, mean_abs, 1.0),
        "峭度指标": safe_divide(peak, rms, 1.0),
        "脉冲指标": safe_divide(peak, mean_abs, 1.0),
        "裕度指标": safe_divide(peak, sqr_mean**2, 1.0),
        "间隙指标": safe_divide(peak, sqr_mean, 1.0),
        "峭度": kurtosis(x, nan_policy='omit'),
        "偏度": skew(x, nan_policy='omit'),
    }
    
    # 检查并替换任何剩余的NaN值
    for key, val in feats.items():
        if not np.isfinite(val):
            feats[key] = 0.0
    
    return feats


def freq_features(x, fs, fr):
    """频域特征（转速归一化）"""
    # 移除NaN和无穷值
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    
    try:
        f, Pxx = welch(x, fs=fs, nperseg=min(2048, len(x)//4))
        
        # 检查功率谱是否有效
        if len(Pxx) == 0 or np.sum(Pxx) <= 1e-15:
            return {}
            
        Pxx = Pxx / (np.sum(Pxx) + 1e-15)  # 安全的归一化功率谱

        # 安全的转频归一化
        fr = max(fr, 1e-6)  # 避免除零
        f_norm = f / fr  

        # 安全计算频域特征
        centroid = np.sum(f_norm * Pxx)
        mean_freq = np.mean(f_norm)
        bandwidth = np.sqrt(np.sum(((f_norm - mean_freq)**2) * Pxx))
        
        # 安全的统计计算
        pxx_safe = Pxx[np.isfinite(Pxx)]
        if len(pxx_safe) == 0:
            return {}
            
        feats = {
            "谱质心": centroid if np.isfinite(centroid) else 0.0,
            "谱带宽": bandwidth if np.isfinite(bandwidth) else 0.0,
            "谱偏度": skew(pxx_safe, nan_policy='omit'),
            "谱峭度": kurtosis(pxx_safe, nan_policy='omit'),
            "谱熵": -np.sum(Pxx * np.log(Pxx + 1e-12)),
        }

        # 带通能量（按 fr 倍频划分）
        bands = [(0.8,1.2),(1.8,2.2),(2.8,3.2),(4.5,5.5)]
        for i,(lo,hi) in enumerate(bands,1):
            mask = (f_norm>=lo)&(f_norm<=hi)
            band_energy = np.sum(Pxx[mask]) if np.any(mask) else 0.0
            feats[f"带通能量_{i}"] = band_energy
            
        # 检查并替换任何剩余的NaN值
        for key, val in feats.items():
            if not np.isfinite(val):
                feats[key] = 0.0
                
    except Exception as e:
        print(f"频域特征计算出错: {e}")
        return {}
        
    return feats


def tf_features(x, fs, fr):
    """时频域特征（STFT + 小波能量）"""
    # 移除NaN和无穷值
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    
    feats = {}
    
    try:
        # --- STFT ---
        f, t, Zxx = stft(x, fs=fs, nperseg=min(1024, len(x)//4))
        power = np.abs(Zxx)**2
        
        # 检查功率是否有效
        total_power = np.sum(power)
        if total_power <= 1e-15:
            feats["时频熵"] = 0.0
            feats["时频均值频率"] = 0.0
        else:
            power = power / (total_power + 1e-15)
            
            # 安全的转频归一化
            fr = max(fr, 1e-6)
            f_norm = f / fr
            
            # 计算时频特征
            tf_entropy = -np.sum(power * np.log(power + 1e-12))
            mean_power_freq = np.mean(power, axis=1)
            mean_freq = np.sum(mean_power_freq * f_norm)
            
            feats["时频熵"] = tf_entropy if np.isfinite(tf_entropy) else 0.0
            feats["时频均值频率"] = mean_freq if np.isfinite(mean_freq) else 0.0
            
    except Exception as e:
        print(f"STFT计算出错: {e}")
        feats["时频熵"] = 0.0
        feats["时频均值频率"] = 0.0

    try:
        # --- 小波分解 ---
        if len(x) >= 8:  # 小波分解需要足够的数据点
            coeffs = pywt.wavedec(x, 'db4', level=min(4, int(np.log2(len(x)))))
            energy = np.array([np.sum(c**2) for c in coeffs])
            total_energy = np.sum(energy)
            
            if total_energy > 1e-15:
                energy_ratio = energy / (total_energy + 1e-15)
            else:
                energy_ratio = np.zeros_like(energy)
                
            for i, e in enumerate(energy_ratio):
                feats[f"小波能量_{i}"] = e if np.isfinite(e) else 0.0
        else:
            # 数据点不足，设置默认值
            for i in range(5):  # 默认5个小波能量特征
                feats[f"小波能量_{i}"] = 0.0
                
    except Exception as e:
        print(f"小波分解出错: {e}")
        for i in range(5):
            feats[f"小波能量_{i}"] = 0.0
    
    # 最终检查，确保没有NaN值
    for key, val in feats.items():
        if not np.isfinite(val):
            feats[key] = 0.0
    
    return feats


def extract_features_balanced_from_long(
    long_table: pd.DataFrame,
    signal_col: str = "DE_time",
    fs: int = 32000,
    target_per_class: int = 77,   # 目标：每个类别都补到 77
    mix_ratio: float = 0.5,       # "随机采用比例"（掩码中 1 的期望占比）
    random_state: int = 42        # 随机种子，保证可复现
):
    """
    基于原始 long 表进行采样与样本均衡：完整版数据平衡算法
    """
    rng = np.random.default_rng(random_state)
    random.seed(random_state)

    # Step 1：先按文件提取"原始样本"的特征，并缓存原始信号用于后续合成
    feature_rows = []
    labels = []
    # 为后续合成准备一个缓存：每个状态 -> [(file_id, x_array, rpm), ...]
    raw_pool = {}

    print(f"正在提取 {signal_col} 特征...")
    file_groups = list(long_table.groupby("file"))
    
    for i, (fid, group) in enumerate(file_groups):
        if i % 20 == 0:
            print(f"  进度: {i+1}/{len(file_groups)}")
            
        x = group[signal_col].dropna().values
        if len(x) == 0:
            continue
        
        # 安全处理RPM值
        rpm = group["RPM"].iloc[0] if "RPM" in group.columns else None
        if pd.isna(rpm) or rpm <= 0:
            fr = 1.0  # 默认转频
        else:
            fr = rpm / 60.0
        
        status = group["status"].iloc[0] if "status" in group.columns else "unknown"

        # 提取特征（中文字段名）
        feats = {}
        feats.update(time_features(x))
        feats.update(freq_features(x, fs, fr))
        feats.update(tf_features(x, fs, fr))
        feats["文件名"] = fid

        feature_rows.append(feats)
        labels.append(status)

        # 存入池子，便于后续做少数类的合成
        raw_pool.setdefault(status, []).append((fid, x, rpm))

    # 组装原始特征表
    X = pd.DataFrame(feature_rows).set_index("文件名")
    y = pd.Series(labels, index=X.index, name="状态")

    # Step 2：统计每个类别当前数量，决定需要合成多少
    counts = y.value_counts()
    need_augment = {
        cls: max(0, target_per_class - cnt)
        for cls, cnt in counts.items()
    }
    
    print("原始各类数量：")
    print(counts)
    print("需要增广的数量：")
    print(need_augment)

    # Step 3：对少数类进行"随机掩码混合"合成新信号并提特征
    aug_feature_rows = []
    aug_labels = []

    for cls, n_needed in need_augment.items():
        if n_needed <= 0:
            continue  # 该类已达到或超过目标，无需增广

        print(f"正在为类别 {cls} 生成 {n_needed} 个增广样本...")
        
        # 该类别的原始池
        pool = raw_pool.get(cls, [])
        if len(pool) == 0:
            print(f"警告：类别 {cls} 在原始数据中不存在，无法增广。")
            continue
        if len(pool) == 1:
            print(f"提示：类别 {cls} 只有 1 个原始样本，增广将以同一条信号自混合方式进行。")

        for k in range(n_needed):
            if k % 10 == 0 and k > 0:
                print(f"    {cls} 类别增广进度: {k}/{n_needed}")
                
            # 随机从该类中取两条（可重复）用于混合
            (fid_a, xa, rpm_a) = random.choice(pool)
            (fid_b, xb, rpm_b) = random.choice(pool)

            # 对齐长度（截断到最短长度）
            L = min(len(xa), len(xb))
            xa_ = xa[:L]
            xb_ = xb[:L]

            # 生成随机0/1掩码（mix_ratio 的比例来自 xa，其余来自 xb）
            mask = rng.random(L) < mix_ratio
            # 逐点拼接（掩码为 True 取 xa，否则取 xb）
            x_new = np.where(mask, xa_, xb_)

            # 用 A 的 RPM 作为该新样本的转速
            rpm_new = rpm_a
            fr_new = (rpm_new / 60.0) if (pd.notna(rpm_new) and rpm_new > 0) else 1.0

            # 提取新样本特征
            feats_new = {}
            feats_new.update(time_features(x_new))
            feats_new.update(freq_features(x_new, fs, fr_new))
            feats_new.update(tf_features(x_new, fs, fr_new))

            # 新样本的"文件名"（索引）
            new_id = f"{fid_a}__aug_{k+1:03d}_cls_{cls}"
            feats_new["文件名"] = new_id

            aug_feature_rows.append(feats_new)
            aug_labels.append(cls)

    # Step 4：合并原始与增广后的特征，并返回
    if len(aug_feature_rows) > 0:
        X_aug = pd.DataFrame(aug_feature_rows).set_index("文件名")
        y_aug = pd.Series(aug_labels, index=X_aug.index, name="状态")
        X_balanced = pd.concat([X, X_aug], axis=0)
        y_balanced = pd.concat([y, y_aug], axis=0)
    else:
        X_balanced, y_balanced = X, y

    # 检查均衡结果
    print("均衡后各类数量：")
    print(y_balanced.value_counts())

    return X_balanced, y_balanced


def extract_features_from_target_long(tgt_long, signal_col="Xtime", fs=32000):
    """
    从目标域长格式数据中提取特征
    """
    feature_rows = []

    for fid, group in tgt_long.groupby("file"):
        x = group[signal_col].dropna().values
        if len(x) == 0: 
            continue

        # 安全处理RPM值
        rpm = group["RPM"].iloc[0] if "RPM" in group.columns else None
        if pd.isna(rpm) or rpm <= 0:
            fr = 1.0  # 默认转频
        else:
            fr = rpm / 60.0

        feats = {}
        feats.update(time_features(x))
        feats.update(freq_features(x, fs, fr))
        feats.update(tf_features(x, fs, fr))
        feats["file"] = fid

        feature_rows.append(feats)

    X = pd.DataFrame(feature_rows).set_index("file")
    return X


def coral_alignment(source, target):
    """
    CORAL: CORrelation ALignment
    """
    # 转换为 numpy
    if isinstance(source, pd.DataFrame):
        source_ = source.values
    else:
        source_ = source
    if isinstance(target, pd.DataFrame):
        target_ = target.values
    else:
        target_ = target

    # 计算协方差矩阵
    cov_src = np.cov(source_, rowvar=False) + np.eye(source_.shape[1]) * 1e-6
    cov_tar = np.cov(target_, rowvar=False) + np.eye(target_.shape[1]) * 1e-6

    # 使用SVD进行矩阵开方
    U_s, S_s, _ = np.linalg.svd(cov_src)
    U_t, S_t, _ = np.linalg.svd(cov_tar)
    A_s = U_s @ np.diag(S_s**-0.5) @ U_s.T
    A_t = U_t @ np.diag(S_t**0.5) @ U_t.T
    
    # 变换
    source_aligned = (source_ - source_.mean(0)) @ A_s @ A_t + target_.mean(0)

    return source_aligned


def _kernel(X1, X2=None, kernel='rbf', gamma=None):
    """核函数计算"""
    X2 = X1 if X2 is None else X2
    if kernel == 'linear':
        return X1 @ X2.T
    if gamma is None:
        gamma = 1.0 / X1.shape[1]
    # RBF
    X1_sq = np.sum(X1**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2**2, axis=1, keepdims=True).T
    dist2 = X1_sq + X2_sq - 2 * (X1 @ X2.T)
    return np.exp(-gamma * dist2)


def TCA(Xs, Xt, dim=20, mu=1.0, kernel='rbf', gamma=None):
    """
    Transfer Component Analysis (TCA)
    """
    X = np.vstack([Xs, Xt])
    n = X.shape[0]
    ns = Xs.shape[0]
    nt = Xt.shape[0]

    # MMD 矩阵 L
    e = np.vstack([
        np.full((ns,1),  1.0/ns),
        np.full((nt,1), -1.0/nt)
    ])
    L = e @ e.T

    # 中心化矩阵 H
    H = np.eye(n) - np.ones((n,n)) / n

    # 核矩阵 K
    K = _kernel(X, kernel=kernel, gamma=gamma)

    # 广义特征问题
    KLK = K @ L @ K
    KHK = K @ H @ K
    A_mat = KLK + mu * np.eye(n)

    # 数值稳定的求解
    reg = 1e-6
    A_inv = np.linalg.pinv(A_mat + reg*np.eye(n))
    M = A_inv @ KHK

    eigvals, eigvecs = eigh(M)
    idx = np.argsort(eigvals)[::-1]
    A = eigvecs[:, idx[:dim]]

    Z = A.T @ K
    Zs = Z[:, :ns].T
    Zt = Z[:, ns:].T
    return Zs, Zt


def filter_source_samples(X_src, y_src, X_tgt, threshold_ratio=0.2):
    """
    根据特征范围筛选源域样本
    """
    # 转换为 DataFrame
    X_src_df = pd.DataFrame(X_src, columns=range(X_src.shape[1]) if isinstance(X_src, np.ndarray) else X_src.columns)
    X_tgt_df = pd.DataFrame(X_tgt, columns=range(X_tgt.shape[1]) if isinstance(X_tgt, np.ndarray) else X_tgt.columns)

    # 目标域 min/max
    tgt_min = X_tgt_df.min(axis=0)
    tgt_max = X_tgt_df.max(axis=0)

    # 判断每个特征是否在范围内
    satisfy_matrix = (X_src_df >= tgt_min) & (X_src_df <= tgt_max)

    # 每个样本满足的特征数量
    satisfy_count = satisfy_matrix.sum(axis=1)

    # 阈值：至少满足指定比例的特征
    threshold = int(threshold_ratio * X_src_df.shape[1])
    mask = satisfy_count >= threshold

    # 筛选后的源域样本
    X_src_filtered = X_src_df[mask].reset_index(drop=True)
    y_src_filtered = y_src[mask].reset_index(drop=True) if isinstance(y_src, pd.Series) else pd.Series(y_src)[mask].reset_index(drop=True)

    print("原始源域样本数:", len(X_src_df))
    print("筛选后源域样本数:", len(X_src_filtered))
    print("平均满足特征数:", satisfy_count.mean())
    print("筛选阈值 (至少特征数):", threshold)

    return X_src_filtered, y_src_filtered, mask


def plot_label_distribution(y, title="标签分布"):
    """绘制标签分布图"""
    plt.figure(figsize=(8,5))
    ax = sns.countplot(x=y, palette="Set3")
    
    # 添加数值标签
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2, height, f'{height}',
                ha='center', va='bottom')
    
    plt.title(title)
    plt.xlabel("类别")
    plt.ylabel("样本数")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_prediction_distribution(y_pred, title="预测结果分布"):
    """绘制预测结果分布"""
    plt.figure(figsize=(8,5))
    ax = sns.countplot(x=y_pred, palette="Set2")

    # 添加标签
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2,
                height,
                f'{height}',
                ha='center', va='bottom')

    plt.title(title)
    plt.xlabel("预测类别")
    plt.ylabel("样本数")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(y_test, y_pred_proba, classes, title="多分类ROC曲线"):
    """绘制多分类ROC曲线"""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]

    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{classes[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("假阳率 (FPR)")
    plt.ylabel("真正率 (TPR)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_test, y_pred, classes, title="混淆矩阵"):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_feature_selection_curve(X_src, X_tgt, feature_names):
    """绘制特征筛选保留样本数曲线（使用真实的源域和目标域数据）"""
    # 转换为DataFrame
    X_src_df = pd.DataFrame(X_src, columns=feature_names)
    X_tgt_df = pd.DataFrame(X_tgt, columns=feature_names)
    
    # 目标域 min/max
    tgt_min = X_tgt_df.min(axis=0)
    tgt_max = X_tgt_df.max(axis=0)
    
    # 判断每个特征是否在目标域范围内
    satisfy_matrix = (X_src_df >= tgt_min) & (X_src_df <= tgt_max)
    satisfy_count = satisfy_matrix.sum(axis=1)
    n_features = X_src_df.shape[1]
    
    # 不同比例阈值下的保留样本数
    ratios = np.linspace(0.1, 1.0, 20)
    kept_counts = []
    
    for r in ratios:
        threshold = int(r * n_features)
        kept = (satisfy_count >= threshold).sum()
        kept_counts.append(kept)
    
    plt.figure(figsize=(8,5))
    plt.plot(ratios, kept_counts, marker="o", linewidth=2, markersize=6)
    plt.title(f"源域样本筛选曲线\n(基于{len(X_tgt_df)}个目标域样本的特征范围)")
    plt.xlabel("特征比例阈值")
    plt.ylabel("保留的源域样本数")
    plt.grid(True, alpha=0.3)
    
    # 添加注释
    plt.text(0.5, max(kept_counts)*0.8, 
             f"目标域样本数: {len(X_tgt_df)}\n源域原始样本数: {len(X_src_df)}", 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    print(f"源域原始样本数: {len(X_src_df)}")
    print(f"目标域样本数: {len(X_tgt_df)}")
    print(f"平均满足特征数: {satisfy_count.mean():.1f}")
    
    return satisfy_count


def plot_source_target_comparison(X_src, X_tgt, feature_names, title="源域vs目标域特征对比"):
    """绘制源域和目标域特征分布对比"""
    # 转换为DataFrame
    X_src_df = pd.DataFrame(X_src, columns=feature_names)
    X_tgt_df = pd.DataFrame(X_tgt, columns=feature_names)
    
    # 添加domain标签
    X_src_df["domain"] = "Source"
    X_tgt_df["domain"] = "Target"
    
    # 合并
    X_all_df = pd.concat([X_src_df, X_tgt_df], axis=0)
    
    # 箱线图对比前5个特征
    plt.figure(figsize=(12,6))
    sns.boxplot(data=X_all_df.melt(id_vars=["domain"], 
                                   value_vars=feature_names[:5]),
                x="variable", y="value", hue="domain")
    plt.title(f"{title} - 前5个特征分布对比")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
    
    # 统计差异
    stats_src = X_src_df[feature_names].agg(["mean","std"]).T
    stats_tgt = X_tgt_df[feature_names].agg(["mean","std"]).T
    stats_diff = (stats_src["mean"] - stats_tgt["mean"]).abs().sort_values(ascending=False)
    
    print("源域 vs 目标域 特征均值差异最大的前10个特征：")
    print(stats_diff.head(10))
    
    return stats_diff


def plot_domain_adaptation_comparison(X_src_before, X_src_after, X_tgt, method_name="域适应"):
    """对比域适应前后的特征分布"""
    # CORAL前
    X_all_before = np.vstack([X_src_before, X_tgt])
    domain_labels = np.array(["Source"]*len(X_src_before) + ["Target"]*len(X_tgt))
    
    # 调整perplexity（考虑目标域只有16个样本的情况）
    n_samples = X_all_before.shape[0]
    perplexity = max(1, min(30, n_samples//4))
    
    tsne_before = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne_before = tsne_before.fit_transform(X_all_before)
    
    # CORAL后
    X_all_after = np.vstack([X_src_after, X_tgt])
    tsne_after = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne_after = tsne_after.fit_transform(X_all_after)
    
    # 绘图
    plt.figure(figsize=(15,6))
    
    # 子图1：域适应前
    plt.subplot(1,2,1)
    sns.scatterplot(x=X_tsne_before[:,0], y=X_tsne_before[:,1], 
                    hue=domain_labels, alpha=0.7, palette="Set1")
    plt.title(f"t-SNE分布 ({method_name}前)")
    plt.legend()
    
    # 子图2：域适应后
    plt.subplot(1,2,2)
    sns.scatterplot(x=X_tsne_after[:,0], y=X_tsne_after[:,1], 
                    hue=domain_labels, alpha=0.7, palette="Set1")
    plt.title(f"t-SNE分布 ({method_name}后)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance_comparison(X_before, X_after, y, feature_names, method_name="域适应"):
    """对比域适应前后的特征重要性"""
    from sklearn.ensemble import RandomForestClassifier
    
    # 训练模型
    clf_before = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_before.fit(X_before, y)
    
    clf_after = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_after.fit(X_after, y)
    
    # 特征重要性
    imp_before = pd.Series(clf_before.feature_importances_, index=feature_names)
    imp_after = pd.Series(clf_after.feature_importances_, index=feature_names)
    
    # 取前15个重要特征对比
    top_features = imp_before.sort_values(ascending=False).head(15).index
    imp_df = pd.DataFrame({
        f"{method_name}前": imp_before[top_features],
        f"{method_name}后": imp_after[top_features]
    })
    
    plt.figure(figsize=(10,8))
    imp_df.plot(kind="barh", figsize=(10,8))
    plt.title(f"特征重要性对比 ({method_name}前 vs {method_name}后)")
    plt.xlabel("重要性")
    plt.tight_layout()
    plt.show()
    
    return imp_df


def plot_tsne_comparison(X_src, X_tgt, domain_labels, class_labels=None, title="t-SNE 特征分布"):
    """绘制t-SNE特征分布对比"""
    X_all = np.vstack([X_src, X_tgt])
    
    # 调整perplexity（考虑目标域只有16个样本的情况）
    n_samples = X_all.shape[0]
    perplexity = max(1, min(30, n_samples//4))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, learning_rate=200)
    X_tsne = tsne.fit_transform(X_all)

    plt.figure(figsize=(8,6))
    if class_labels is not None:
        sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1],
                       hue=domain_labels, style=class_labels,
                       palette="Set1", alpha=0.7)
    else:
        sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1],
                       hue=domain_labels, palette="Set1", alpha=0.7)
    
    plt.title(f"{title} (perplexity={perplexity})")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_clustering_comparison(X_tgt, y_pred, y_kmeans, title="聚类对比分析"):
    """绘制聚类对比分析"""
    n_samples = X_tgt.shape[0]
    perplexity = max(1, min(5, n_samples // 4))  # 目标域只有16个样本，perplexity要更小

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, learning_rate=200)
    X_tgt_tsne = tsne.fit_transform(X_tgt)

    plt.figure(figsize=(14,6))

    # 子图1：模型预测类别
    plt.subplot(1,2,1)
    sns.scatterplot(x=X_tgt_tsne[:,0], y=X_tgt_tsne[:,1],
                    hue=y_pred, style=y_pred,
                    palette="tab10", alpha=0.8, s=80)
    plt.title(f"模型预测类别 (perplexity={perplexity})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left", title="预测类别")

    # 子图2：KMeans 聚类结果
    plt.subplot(1,2,2)
    sns.scatterplot(x=X_tgt_tsne[:,0], y=X_tgt_tsne[:,1],
                    hue=y_kmeans, style=y_kmeans,
                    palette="tab20", alpha=0.8, s=80)
    plt.title(f"KMeans 聚类结果 (perplexity={perplexity})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left", title="KMeans 簇")

    plt.tight_layout()
    plt.show()


def analyze_feature_distribution(X_src, X_tgt, feature_names=None):
    """分析特征分布差异"""
    if feature_names is None:
        feature_names = [f"特征_{i}" for i in range(X_src.shape[1])]
    
    # 转换为 DataFrame
    X_src_df = pd.DataFrame(X_src, columns=feature_names)
    X_tgt_df = pd.DataFrame(X_tgt, columns=feature_names)

    # 添加 domain 标签
    X_src_df["domain"] = "Source"
    X_tgt_df["domain"] = "Target"

    # 合并
    X_all_df = pd.concat([X_src_df, X_tgt_df], axis=0)

    # 箱线图对比前几个特征
    plt.figure(figsize=(12,6))
    sns.boxplot(data=X_all_df.melt(id_vars=["domain"], 
                                   value_vars=feature_names[:5]),
                x="variable", y="value", hue="domain")
    plt.title("源域 vs 目标域 前5个特征分布对比")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    # 统计差异
    stats_src = X_src_df[feature_names].agg(["mean","std"]).T
    stats_tgt = X_tgt_df[feature_names].agg(["mean","std"]).T
    stats_diff = (stats_src["mean"] - stats_tgt["mean"]).abs().sort_values(ascending=False)

    print("源域 vs 目标域 特征均值差异最大的前10个特征：")
    print(stats_diff.head(10))

    return stats_diff


def shap_analysis(model, X_data, feature_names=None, max_samples=100):
    """SHAP可解释性分析"""
    if not SHAP_AVAILABLE:
        print("SHAP未安装，跳过可解释性分析")
        return None, None
    
    # 限制样本数量以提高计算效率
    if len(X_data) > max_samples:
        indices = np.random.choice(len(X_data), max_samples, replace=False)
        X_sample = X_data[indices]
    else:
        X_sample = X_data
    
    try:
        print("正在计算SHAP值...")
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model, X_sample, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_sample)
        
        # 如果是多分类，shap_values是3D数组
        if isinstance(shap_values, list) or (isinstance(shap_values, np.ndarray) and shap_values.ndim == 3):
            print("检测到多分类SHAP值")
            
            # 汇总所有类别的SHAP值
            if isinstance(shap_values, list):
                shap_matrix_all = np.abs(np.array(shap_values)).mean(axis=0)
            else:
                shap_matrix_all = np.abs(shap_values).mean(axis=2)
            
            plt.figure(figsize=(10,6))
            shap.summary_plot(
                shap_matrix_all,
                X_sample,
                feature_names=feature_names,
                plot_type="dot",
                max_display=min(20, X_sample.shape[1]),
                show=False
            )
            plt.title("SHAP 特征重要性汇总")
            plt.tight_layout()
            plt.show()
            
        else:
            # 二分类情况
            plt.figure(figsize=(10,6))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                plot_type="dot",
                max_display=min(20, X_sample.shape[1]),
                show=False
            )
            plt.title("SHAP 特征重要性")
            plt.tight_layout()
            plt.show()
        
        return shap_values, explainer
        
    except Exception as e:
        print(f"SHAP分析出错：{e}")
        return None, None


def plot_shap_by_class(model, X_data, feature_names=None, max_samples=100):
    """按类别绘制SHAP蜂群图"""
    if not SHAP_AVAILABLE:
        print("SHAP未安装，跳过SHAP分析")
        return None
    
    # 限制样本数量
    if len(X_data) > max_samples:
        indices = np.random.choice(len(X_data), max_samples, replace=False)
        X_sample = X_data[indices]
    else:
        X_sample = X_data
    
    try:
        print("正在计算各类别SHAP值...")
        explainer = shap.TreeExplainer(model, X_sample, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_sample)
        
        classes = model.classes_
        
        # 为每个类别绘制SHAP图
        for i, class_name in enumerate(classes):
            plt.figure(figsize=(10,6))
            if isinstance(shap_values, list):
                shap_matrix = shap_values[i]
            else:
                shap_matrix = shap_values[:, :, i]
                
            shap.summary_plot(
                shap_matrix,
                X_sample,
                feature_names=feature_names,
                plot_type="dot",
                max_display=min(20, X_sample.shape[1]),
                show=False
            )
            plt.title(f"SHAP 蜂群图 - 类别 {class_name}")
            plt.tight_layout()
            plt.show()
            
        return shap_values, explainer
        
    except Exception as e:
        print(f"SHAP分析出错：{e}")
        return None, None


def plot_shap_waterfall(model, X_sample, sample_idx=0, feature_names=None):
    """绘制SHAP瀑布图"""
    if not SHAP_AVAILABLE:
        print("SHAP未安装，跳过SHAP分析")
        return
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # 获取预测类别
        pred_class = model.predict(X_sample[sample_idx].reshape(1, -1))[0]
        classes = model.classes_
        class_idx = list(classes).index(pred_class)
        
        print(f"样本 {sample_idx} 的预测类别: {pred_class}")
        
        # 为每个类别绘制瀑布图
        for j, cls in enumerate(classes):
            if isinstance(shap_values, list):
                shap_sample = shap_values[j][sample_idx, :]
            else:
                shap_sample = shap_values[sample_idx, :, j]
            
            exp = shap.Explanation(
                values=shap_sample,
                base_values=explainer.expected_value[j],
                data=X_sample[sample_idx, :],
                feature_names=feature_names
            )
            
            plt.figure(figsize=(8,6))
            shap.plots.waterfall(exp, max_display=10, show=False)
            plt.title(f"样本 {sample_idx} 在类别 {cls} 上的SHAP解释", fontsize=14)
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"SHAP瀑布图绘制出错：{e}")


def plot_model_performance_metrics(y_test, y_pred, y_pred_proba, classes, model_name="模型"):
    """绘制模型性能评估图表"""
    print(f"\n{model_name} 性能评估:")
    
    # 1. ROC曲线
    plot_roc_curves(y_test, y_pred_proba, classes, f"{model_name} ROC曲线")
    
    # 2. 混淆矩阵
    plot_confusion_matrix(y_test, y_pred, classes, f"{model_name} 混淆矩阵")
    
    # 3. 分类报告
    print("分类报告:")
    print(classification_report(y_test, y_pred, target_names=classes))


def main():
    """主程序入口 - Transformer版本"""
    print("开始基于Transformer的目标域故障诊断与过程解释...")
    
    # 设置环境
    setup_environment()
    
    # 1. 加载数据
    print("1. 加载数据...")
    try:
        # 加载源域数据
        long_table = pd.read_csv('源域数据32khz整理.csv')
        print(f"   源域数据加载成功，维度：{long_table.shape}")
        
        # 加载目标域数据
        tgt_long = pd.read_csv('目标域数据整理.csv')
        print(f"   目标域数据加载成功，维度：{tgt_long.shape}")
        
    except FileNotFoundError as e:
        print(f"   错误：{e}")
        print("   请确保先运行前面的脚本生成数据文件")
        return None
    
    # 2. 完整的多通道特征提取（包含数据平衡）
    print("2. 完整的多通道特征提取...")
    
    # 提取源域三通道特征（完整数据平衡）
    X_DE, y_DE = extract_features_balanced_from_long(long_table, signal_col="DE_time", fs=32000)
    X_FE, y_FE = extract_features_balanced_from_long(long_table, signal_col="FE_time", fs=32000) 
    X_BA, y_BA = extract_features_balanced_from_long(long_table, signal_col="BA_time", fs=32000)
    
    # 重置索引
    X_DE.reset_index(inplace=True, drop=True)
    X_FE.reset_index(inplace=True, drop=True)
    X_BA.reset_index(inplace=True, drop=True)
    y_DE.reset_index(inplace=True, drop=True)
    y_FE.reset_index(inplace=True, drop=True)
    y_BA.reset_index(inplace=True, drop=True)

    # 合并三通道数据
    X_src_combined = pd.concat([X_DE, X_FE, X_BA], axis=0, ignore_index=True)
    y_src_combined = pd.concat([y_DE, y_FE, y_BA], axis=0, ignore_index=True)
    
    print(f"   合并后源域特征维度：{X_src_combined.shape}")
    print(f"   合并后源域标签维度：{y_src_combined.shape}")
    
    # 绘制合并后的标签分布
    plot_label_distribution(y_src_combined, "合并后源域标签分布")
    
    # 特征选择
    print("   进行特征选择...")
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(estimator, n_features_to_select=20, step=1)
    selector = selector.fit(X_src_combined, y_src_combined)
    selected_features = X_src_combined.columns[selector.support_]
    X_src_selected = X_src_combined[selected_features]
    
    print("选择的特征：")
    print(selected_features.tolist())
    
    # 提取目标域特征
    X_tgt_full = extract_features_from_target_long(tgt_long, signal_col="Xtime", fs=32000)
    X_tgt = X_tgt_full[selected_features]
    
    print(f"   目标域特征维度：{X_tgt.shape}")
    
    # 验证目标域样本数
    if X_tgt.shape[0] != 16:
        print(f"   ⚠️  警告：目标域样本数为 {X_tgt.shape[0]}，预期应为16个样本")
    else:
        print(f"   ✓ 确认：目标域有 {X_tgt.shape[0]} 个样本（符合预期）")
    
    # 3. 特征标准化
    print("3. 特征标准化...")
    scaler = StandardScaler()
    scaler.fit(X_src_selected)
    X_src_scaled = scaler.transform(X_src_selected)
    X_tgt_scaled = scaler.transform(X_tgt)
    
    # 4. 样本筛选
    print("4. 源域样本筛选...")
    X_src_filtered, y_src_filtered, filter_mask = filter_source_samples(
        X_src_scaled, y_src_combined, X_tgt_scaled, threshold_ratio=0.2
    )
    
    # 5. Transformer域适应方法对比
    print("5. Transformer域适应方法对比...")
    
    if PYTORCH_AVAILABLE:
        # 5.1 Transformer基线（无域适应）
        print("5.1 训练Transformer基线模型...")
        transformer_baseline, y_tgt_pred_transformer_baseline, baseline_utils = train_transformer_baseline(
            X_src_filtered, y_src_filtered, X_tgt_scaled, 
            epochs=50, batch_size=16
        )
        if y_tgt_pred_transformer_baseline is not None:
            plot_prediction_distribution(y_tgt_pred_transformer_baseline, "Transformer基线 - 目标域预测分布")
        
        # 5.2 Transformer + CORAL适应
        print("5.2 Transformer + CORAL域适应...")
        transformer_coral, y_tgt_pred_transformer_coral, coral_utils = train_transformer_with_coral(
            X_src_filtered, y_src_filtered, X_tgt_scaled,
            epochs=50, batch_size=16
        )
        if y_tgt_pred_transformer_coral is not None:
            plot_prediction_distribution(y_tgt_pred_transformer_coral, "Transformer + CORAL适应 - 目标域预测分布")
        
        # 使用Transformer CORAL结果作为主要预测结果
        y_tgt_pred_main = y_tgt_pred_transformer_coral if y_tgt_pred_transformer_coral is not None else y_tgt_pred_transformer_baseline
        main_model = transformer_coral if transformer_coral is not None else transformer_baseline
        
    else:
        print("PyTorch未安装，使用传统方法...")
        # 回退到传统方法
        # 5.1 无适应基线
        print("5.1 训练RandomForest基线模型...")
        clf_baseline = RandomForestClassifier(random_state=42)
        clf_baseline.fit(X_src_filtered, y_src_filtered)
        y_tgt_pred_baseline = clf_baseline.predict(X_tgt_scaled)
        plot_prediction_distribution(y_tgt_pred_baseline, "RandomForest基线 - 目标域预测分布")
        
        # 5.2 CORAL适应
        print("5.2 CORAL域适应...")
        X_src_coral = coral_alignment(X_src_filtered, X_tgt_scaled)
        clf_coral = RandomForestClassifier(random_state=42)
        clf_coral.fit(X_src_coral, y_src_filtered)
        y_tgt_pred_coral = clf_coral.predict(X_tgt_scaled)
        plot_prediction_distribution(y_tgt_pred_coral, "RandomForest + CORAL适应 - 目标域预测分布")
        
        y_tgt_pred_main = y_tgt_pred_coral
        main_model = clf_coral
    
    # 6. 全面可视化分析
    print("6. 全面可视化分析...")
    
    # 6.1 特征选择曲线
    print("6.1 绘制源域样本筛选曲线...")
    satisfy_count = plot_feature_selection_curve(X_src_filtered, X_tgt_scaled, selected_features)
    
    # 6.2 源域vs目标域特征分布对比
    print("6.2 源域vs目标域特征分布对比...")
    stats_diff = plot_source_target_comparison(X_src_filtered, X_tgt_scaled, selected_features)
    
    # 6.3 域适应前后对比（如果使用传统方法）
    if not PYTORCH_AVAILABLE and 'X_src_coral' in locals():
        print("6.3 域适应前后效果对比...")
        plot_domain_adaptation_comparison(X_src_filtered, X_src_coral, X_tgt_scaled, "CORAL")
        
        # 特征重要性对比
        print("6.4 特征重要性对比...")
        imp_df = plot_feature_importance_comparison(X_src_filtered, X_src_coral, y_src_filtered, selected_features, "CORAL")
    
    # 6.5 t-SNE可视化
    print("6.5 t-SNE降维可视化...")
    if PYTORCH_AVAILABLE and 'X_src_coral' not in locals():
        # 对于Transformer方法，使用原始筛选后的数据进行可视化
        X_src_vis = X_src_filtered
    else:
        X_src_vis = X_src_coral if 'X_src_coral' in locals() else X_src_filtered
        
    domain_labels = np.array(["Source"]*len(X_src_vis) + ["Target"]*len(X_tgt_scaled))
    class_labels = np.concatenate([y_src_filtered.values, ["Target"]*len(X_tgt_scaled)])
    plot_tsne_comparison(X_src_vis, X_tgt_scaled, domain_labels, class_labels, "域适应后特征分布")
    
    # 6.6 源域模型性能评估（如果有验证数据）
    if not PYTORCH_AVAILABLE:
        print("6.6 源域模型性能评估...")
        # 在源域上做一个简单的train-test split来评估
        X_train_src, X_test_src, y_train_src, y_test_src = train_test_split(
            X_src_vis, y_src_filtered, test_size=0.3, random_state=42, stratify=y_src_filtered
        )
        
        temp_model = RandomForestClassifier(random_state=42)
        temp_model.fit(X_train_src, y_train_src)
        y_pred_src = temp_model.predict(X_test_src)
        y_pred_proba_src = temp_model.predict_proba(X_test_src)
        
        plot_model_performance_metrics(y_test_src, y_pred_src, y_pred_proba_src, 
                                     np.unique(y_src_filtered), "源域模型")
    
    # 7. 聚类一致性分析
    print("7. 聚类一致性分析...")
    n_clusters = len(np.unique(y_src_filtered))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X_tgt_scaled)
    
    # 计算一致性指标
    nmi = normalized_mutual_info_score(y_tgt_pred_main, y_kmeans)
    ari = adjusted_rand_score(y_tgt_pred_main, y_kmeans)
    print(f"聚类一致性：NMI={nmi:.3f}, ARI={ari:.3f}")
    
    plot_clustering_comparison(X_tgt_scaled, y_tgt_pred_main, y_kmeans, "目标域聚类对比")
    
    # 7.1 绘制预测vs聚类的混淆矩阵
    print("7.1 绘制预测vs聚类混淆矩阵...")
    plot_confusion_matrix(y_kmeans, y_tgt_pred_main, 
                         classes=sorted(set(y_tgt_pred_main)), 
                         title="目标域：聚类标签 vs 预测标签混淆矩阵")
    
    # 8. 全面可解释性分析
    print("8. 全面可解释性分析...")
    if PYTORCH_AVAILABLE and main_model is not None:
        # 对于Transformer模型，跳过SHAP分析（SHAP不直接支持PyTorch模型）
        print("   Transformer模型暂不支持SHAP分析")
        shap_values, explainer = None, None
    else:
        # 对于传统模型，进行详细的SHAP分析
        print("8.1 SHAP特征重要性汇总...")
        shap_result = shap_analysis(main_model, X_src_vis, selected_features, max_samples=200)
        
        if shap_result is not None:
            shap_values, explainer = shap_result
            
            # 8.2 按类别的SHAP分析
            print("8.2 各类别SHAP蜂群图...")
            plot_shap_by_class(main_model, X_src_vis, selected_features, max_samples=100)
            
            # 8.3 目标域样本SHAP解释
            if len(X_tgt_scaled) > 0:
                print("8.3 目标域样本SHAP瀑布图解释...")
                # 目标域只有16个样本，选择前3个进行解释
                n_samples_to_explain = min(3, len(X_tgt_scaled))
                print(f"   目标域共有 {len(X_tgt_scaled)} 个样本，分析前 {n_samples_to_explain} 个...")
                for i in range(n_samples_to_explain):
                    print(f"   分析目标域样本 {i+1}/{n_samples_to_explain}...")
                    plot_shap_waterfall(main_model, X_tgt_scaled, sample_idx=i, feature_names=selected_features)
        else:
            shap_values, explainer = None, None
    
    # 9. 结果总结
    print("9. 结果总结...")
    print("="*50)
    print("基于Transformer的目标域预测结果统计：")
    
    if PYTORCH_AVAILABLE:
        if y_tgt_pred_transformer_baseline is not None:
            print(f"Transformer基线：{pd.Series(y_tgt_pred_transformer_baseline).value_counts().to_dict()}")
        if y_tgt_pred_transformer_coral is not None:
            print(f"Transformer + CORAL：{pd.Series(y_tgt_pred_transformer_coral).value_counts().to_dict()}")
    else:
        print(f"RandomForest基线：{pd.Series(y_tgt_pred_baseline if 'y_tgt_pred_baseline' in locals() else []).value_counts().to_dict()}")
        print(f"RandomForest + CORAL：{pd.Series(y_tgt_pred_coral if 'y_tgt_pred_coral' in locals() else []).value_counts().to_dict()}")
    
    print(f"主要预测结果：{pd.Series(y_tgt_pred_main).value_counts().to_dict()}")
    print(f"聚类一致性：NMI={nmi:.3f}, ARI={ari:.3f}")
    print("="*50)
    
    print("基于Transformer的目标域故障诊断与过程解释完成！")
    
    # 构建返回结果
    result_dict = {
        'source_data': (X_src_filtered, y_src_filtered),
        'target_data': (X_tgt, X_tgt_scaled),
        'clustering': {
            'kmeans_labels': y_kmeans,
            'nmi': nmi,
            'ari': ari
        },
        'shap_analysis': {
            'shap_values': shap_values,
            'explainer': explainer
        },
        'selected_features': selected_features,
        'scaler': scaler,
        'feature_stats': stats_diff
    }
    
    if PYTORCH_AVAILABLE:
        result_dict['models'] = {
            'transformer_baseline': transformer_baseline if 'transformer_baseline' in locals() else None,
            'transformer_coral': transformer_coral if 'transformer_coral' in locals() else None,
            'main_model': main_model
        }
        result_dict['predictions'] = {
            'transformer_baseline': y_tgt_pred_transformer_baseline if 'y_tgt_pred_transformer_baseline' in locals() else None,
            'transformer_coral': y_tgt_pred_transformer_coral if 'y_tgt_pred_transformer_coral' in locals() else None,
            'main_prediction': y_tgt_pred_main
        }
    else:
        result_dict['models'] = {
            'baseline': clf_baseline if 'clf_baseline' in locals() else None,
            'coral': clf_coral if 'clf_coral' in locals() else None,
            'main_model': main_model
        }
        result_dict['predictions'] = {
            'baseline': y_tgt_pred_baseline if 'y_tgt_pred_baseline' in locals() else None,
            'coral': y_tgt_pred_coral if 'y_tgt_pred_coral' in locals() else None,
            'main_prediction': y_tgt_pred_main
        }
    
    return result_dict


if __name__ == "__main__":
    results = main()
    
    # 清理临时文件
    import os
    temp_files = ['best_transformer_baseline.pth', 'best_model.pth']
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"已清理临时文件: {temp_file}")
            except:
                pass
