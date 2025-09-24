"""
基于 Transformer + CDAN 迁移学习 + SVM 样本加权的目标域故障诊断
结合了 Conditional Domain Adversarial Network (CDAN) 和支持向量机 (SVM) 的源域样本加权方法
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============== 数据预处理和特征提取 ==============

def time_features(x):
    """时域统计特征"""
    from scipy.stats import kurtosis, skew
    x = np.asarray(x).ravel()
    if len(x) == 0:
        return {}
    rms = np.sqrt(np.mean(x**2))
    mean_abs = np.mean(np.abs(x))
    sqr_mean = np.mean(np.sqrt(np.abs(x)))
    peak = np.max(np.abs(x))

    feats = {
        "均值": np.mean(x),
        "标准差": np.std(x),
        "方差": np.var(x),
        "均方根": rms,
        "峰值": peak,
        "峰峰值": np.ptp(x),
        "平均绝对值": mean_abs,
        "波形指标": rms / (mean_abs + 1e-12),
        "峭度指标": peak / (rms + 1e-12),
        "脉冲指标": peak / (mean_abs + 1e-12),
        "裕度指标": peak / (sqr_mean**2 + 1e-12),
        "间隙指标": peak / (sqr_mean + 1e-12),
        "峭度": kurtosis(x),
        "偏度": skew(x),
    }
    return feats

def freq_features(x, fs, fr):
    """频域特征（转速归一化）"""
    from scipy.signal import welch
    from scipy.stats import kurtosis, skew
    
    f, Pxx = welch(x, fs=fs, nperseg=2048)
    Pxx = Pxx / np.sum(Pxx)  # 归一化功率谱

    # 转频归一化
    f_norm = f / fr  

    feats = {
        "谱质心": np.sum(f_norm * Pxx),
        "谱带宽": np.sqrt(np.sum(((f_norm - np.mean(f_norm))**2) * Pxx)),
        "谱偏度": skew(Pxx),
        "谱峭度": kurtosis(Pxx),
        "谱熵": -np.sum(Pxx * np.log(Pxx + 1e-12)),
    }

    # 带通能量（按 fr 倍频划分）
    bands = [(0.8,1.2),(1.8,2.2),(2.8,3.2),(4.5,5.5)]
    for i,(lo,hi) in enumerate(bands,1):
        mask = (f_norm>=lo)&(f_norm<=hi)
        feats[f"带通能量_{i}"] = np.sum(Pxx[mask])
    return feats

def tf_features(x, fs, fr):
    """时频域特征（STFT + 小波能量）"""
    from scipy.signal import stft
    import pywt
    
    # --- STFT ---
    f,t,Zxx = stft(x, fs=fs, nperseg=1024)
    power = np.abs(Zxx)**2
    power = power / (np.sum(power) + 1e-12)

    # 归一化频率
    f_norm = f / fr
    feats = {
        "时频熵": -np.sum(power * np.log(power + 1e-12)),
        "时频均值频率": np.sum(np.mean(power,axis=1)*f_norm),
    }

    # --- 小波分解 ---
    coeffs = pywt.wavedec(x, 'db4', level=4)
    energy = np.array([np.sum(c**2) for c in coeffs])
    energy_ratio = energy / (np.sum(energy)+1e-12)
    for i,e in enumerate(energy_ratio):
        feats[f"小波能量_{i}"] = e
    return feats

def cdan_alignment_loss(features_src, features_tgt, class_probs_src, class_probs_tgt):
    """
    CDAN 对齐损失函数
    通过最小化源域和目标域特征分布的差异来实现对齐
    """
    # 计算条件特征：特征与类别概率的外积
    cond_features_src = torch.bmm(features_src.unsqueeze(2), class_probs_src.unsqueeze(1))
    cond_features_tgt = torch.bmm(features_tgt.unsqueeze(2), class_probs_tgt.unsqueeze(1))
    
    # 展平条件特征
    cond_features_src = cond_features_src.view(features_src.size(0), -1)
    cond_features_tgt = cond_features_tgt.view(features_tgt.size(0), -1)
    
    # 计算 MMD (Maximum Mean Discrepancy) 损失
    def gaussian_kernel(x, y, sigma=1.0):
        """高斯核函数"""
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        return torch.exp(-dist / (2 * sigma ** 2))
    
    # 计算核矩阵
    K_ss = gaussian_kernel(cond_features_src, cond_features_src)
    K_tt = gaussian_kernel(cond_features_tgt, cond_features_tgt)
    K_st = gaussian_kernel(cond_features_src, cond_features_tgt)
    
    # MMD 损失
    mmd_loss = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
    
    return mmd_loss


# ============== Transformer 模型架构 ==============

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

class TransformerFeatureExtractor(nn.Module):
    """Transformer 特征提取器"""
    def __init__(self, input_dim, d_model=64, num_heads=4, num_layers=2, dropout=0.3):
        super(TransformerFeatureExtractor, self).__init__()
        
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
        
        # 特征池化
        self.feature_pooling = nn.AdaptiveAvgPool1d(1)
        self.d_model = d_model
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # 池化得到特征向量
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.feature_pooling(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)
        
        return x

class ClassifierHead(nn.Module):
    """分类器头"""
    def __init__(self, d_model, n_classes, dropout=0.3):
        super(ClassifierHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class DomainDiscriminator(nn.Module):
    """域判别器"""
    def __init__(self, d_model, n_classes, dropout=0.3):
        super(DomainDiscriminator, self).__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(d_model + n_classes, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 2)  # 2个域：源域和目标域
        )
    
    def forward(self, features, class_probs):
        # 将特征和类别概率连接
        combined = torch.cat([features, class_probs], dim=1)
        return self.domain_classifier(combined)

class CDANTransformerModel(nn.Module):
    """CDAN + Transformer 模型"""
    def __init__(self, input_dim, d_model=64, num_heads=4, num_layers=2, 
                 n_classes=4, dropout=0.3):
        super(CDANTransformerModel, self).__init__()
        
        self.feature_extractor = TransformerFeatureExtractor(
            input_dim, d_model, num_heads, num_layers, dropout
        )
        self.classifier = ClassifierHead(d_model, n_classes, dropout)
        self.domain_discriminator = DomainDiscriminator(d_model, n_classes, dropout)
        self.n_classes = n_classes
    
    def forward(self, x, alpha=1.0):
        # 特征提取
        features = self.feature_extractor(x)
        
        # 分类预测
        class_logits = self.classifier(features)
        class_probs = F.softmax(class_logits, dim=1)
        
        # 域分类（带梯度反转）
        reversed_features = grad_reverse(features, alpha)
        domain_logits = self.domain_discriminator(reversed_features, class_probs.detach())
        
        return class_logits, domain_logits, features

# ============== SVM 样本加权 ==============

def compute_svm_sample_weights(X_src, X_tgt):
    """
    使用 SVM 计算源域样本权重
    思路：训练一个二分类 SVM 来区分源域和目标域，
    源域样本被误分类为目标域的概率作为权重
    """
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    
    # 标签：源域=0，目标域=1
    y_src = np.zeros(len(X_src))
    y_tgt = np.ones(len(X_tgt))
    
    X_all = np.vstack([X_src, X_tgt])
    y_all = np.concatenate([y_src, y_tgt])
    
    # 使用 XGBoost 代替 SVM（更快且效果好）
    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, n_jobs=-1, random_state=42,
        eval_metric="logloss", tree_method="hist"
    )
    
    clf.fit(X_all, y_all)
    
    # 预测源域样本被分类为目标域的概率
    y_proba = clf.predict_proba(X_all)[:, 1]  # 目标域概率
    sample_weights = y_proba[:len(X_src)]  # 只取源域部分
    
    # 归一化权重
    sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
    
    return sample_weights, clf

# ============== 数据集类 ==============

class FaultDataset(Dataset):
    """故障诊断数据集"""
    def __init__(self, X, y=None, domain_labels=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
        self.domain_labels = torch.LongTensor(domain_labels) if domain_labels is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None and self.domain_labels is not None:
            return self.X[idx], self.y[idx], self.domain_labels[idx]
        elif self.y is not None:
            return self.X[idx], self.y[idx]
        elif self.domain_labels is not None:
            return self.X[idx], self.domain_labels[idx]
        else:
            return self.X[idx]

# ============== 训练函数 ==============

def train_cdan_transformer(X_src, y_src, X_tgt, sample_weights=None, 
                          d_model=64, num_heads=4, num_layers=2, dropout=0.3,
                          lr=1e-3, batch_size=32, epochs=100, alpha_schedule='progressive',
                          alignment_weight=0.1):
    """
    训练 CDAN + Transformer 模型，整合 CDAN 对齐损失
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 标签编码
    le = LabelEncoder()
    y_src_enc = le.fit_transform(y_src)
    n_classes = len(le.classes_)
    
    # 重塑数据为序列格式
    seq_len = min(20, X_src.shape[1])
    input_dim = X_src.shape[1] // seq_len
    
    X_src_seq = X_src[:, :seq_len*input_dim].reshape(-1, seq_len, input_dim)
    X_tgt_seq = X_tgt[:, :seq_len*input_dim].reshape(-1, seq_len, input_dim)
    
    # 域标签
    domain_src = np.zeros(len(X_src_seq))
    domain_tgt = np.ones(len(X_tgt_seq))
    
    # 创建数据集
    src_dataset = FaultDataset(X_src_seq, y_src_enc, domain_src)
    tgt_dataset = FaultDataset(X_tgt_seq, y=None, domain_labels=domain_tgt)
    
    # 创建数据加载器
    if sample_weights is not None:
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        src_loader = DataLoader(src_dataset, batch_size=batch_size, sampler=sampler)
    else:
        src_loader = DataLoader(src_dataset, batch_size=batch_size, shuffle=True)
    
    tgt_loader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = CDANTransformerModel(
        input_dim=input_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        n_classes=n_classes,
        dropout=dropout
    ).to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_class_loss = 0
        total_domain_loss = 0
        total_alignment_loss = 0
        total_loss = 0
        
        # 计算 alpha（梯度反转强度）
        if alpha_schedule == 'progressive':
            p = epoch / epochs
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
        else:
            alpha = 1.0
        
        # 训练一个 epoch
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        
        max_iter = max(len(src_loader), len(tgt_loader))
        
        for i in range(max_iter):
            # 源域数据
            try:
                src_data, src_labels, src_domain = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                src_data, src_labels, src_domain = next(src_iter)
            
            # 目标域数据
            try:
                tgt_data, tgt_domain = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_data, tgt_domain = next(tgt_iter)
            
            # 移到设备
            src_data = src_data.to(device)
            src_labels = src_labels.to(device)
            src_domain = src_domain.to(device)
            tgt_data = tgt_data.to(device)
            tgt_domain = tgt_domain.to(device)
            
            optimizer.zero_grad()
            
            # 源域前向传播
            src_class_logits, src_domain_logits, src_features = model(src_data, alpha)
            
            # 目标域前向传播
            tgt_class_logits, tgt_domain_logits, tgt_features = model(tgt_data, alpha)
            
            # 分类损失（只在源域）
            class_loss = class_criterion(src_class_logits, src_labels)
            
            # 域分类损失
            domain_logits = torch.cat([src_domain_logits, tgt_domain_logits], dim=0)
            domain_labels = torch.cat([src_domain.long(), tgt_domain.long()], dim=0)
            domain_loss = domain_criterion(domain_logits, domain_labels)
            
            # CDAN 对齐损失
            src_class_probs = F.softmax(src_class_logits, dim=1)
            tgt_class_probs = F.softmax(tgt_class_logits, dim=1)
            alignment_loss = cdan_alignment_loss(src_features, tgt_features, 
                                               src_class_probs, tgt_class_probs)
            
            # 总损失
            loss = class_loss + domain_loss + alignment_weight * alignment_loss
            
            loss.backward()
            optimizer.step()
            
            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()
            total_alignment_loss += alignment_loss.item()
            total_loss += loss.item()
        
        avg_class_loss = total_class_loss / max_iter
        avg_domain_loss = total_domain_loss / max_iter
        avg_alignment_loss = total_alignment_loss / max_iter
        avg_total_loss = total_loss / max_iter
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], '
                  f'Class Loss: {avg_class_loss:.4f}, '
                  f'Domain Loss: {avg_domain_loss:.4f}, '
                  f'Alignment Loss: {avg_alignment_loss:.4f}, '
                  f'Total Loss: {avg_total_loss:.4f}, '
                  f'Alpha: {alpha:.3f}')
        
        # 保存最佳模型
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            torch.save(model.state_dict(), 'best_transformer_cdan.pth')
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_transformer_cdan.pth'))
    
    return model, le

# ============== 预测和评估函数 ==============

def predict_target_domain(model, X_tgt, le):
    """在目标域上进行预测"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # 重塑数据
    seq_len = min(20, X_tgt.shape[1])
    input_dim = X_tgt.shape[1] // seq_len
    X_tgt_seq = X_tgt[:, :seq_len*input_dim].reshape(-1, seq_len, input_dim)
    
    # 创建数据加载器
    tgt_dataset = FaultDataset(X_tgt_seq)
    tgt_loader = DataLoader(tgt_dataset, batch_size=32, shuffle=False)
    
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for batch_X in tgt_loader:
            batch_X = batch_X.to(device)
            class_logits, _, _ = model(batch_X)
            probs = F.softmax(class_logits, dim=1)
            preds = torch.argmax(class_logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    # 转换回原始标签
    y_pred = le.inverse_transform(predictions)
    y_proba = np.array(probabilities)
    
    return y_pred, y_proba

def evaluate_clustering_consistency(y_pred, y_proba, n_clusters=None):
    """评估聚类一致性"""
    if n_clusters is None:
        n_clusters = len(np.unique(y_pred))
    
    # KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(y_proba)
    
    # 计算一致性指标
    ari = adjusted_rand_score(y_pred, clusters)
    nmi = normalized_mutual_info_score(y_pred, clusters)
    
    return ari, nmi, clusters

def plot_class_distribution(y_src, y_pred_tgt, clusters):
    """绘制类别分布图"""
    plt.figure(figsize=(18, 6))
    
    # 1. 源域类别分布
    plt.subplot(1, 4, 1)
    src_unique, src_counts = np.unique(y_src, return_counts=True)
    bars1 = plt.bar(src_unique, src_counts, color='lightblue', alpha=0.7, edgecolor='black')
    plt.title('源域类别分布', fontsize=14, fontweight='bold')
    plt.xlabel('故障类别', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for bar, count in zip(bars1, src_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # 2. 目标域预测类别分布
    plt.subplot(1, 4, 2)
    pred_unique, pred_counts = np.unique(y_pred_tgt, return_counts=True)
    bars2 = plt.bar(pred_unique, pred_counts, color='lightgreen', alpha=0.7, edgecolor='black')
    plt.title('目标域预测类别分布', fontsize=14, fontweight='bold')
    plt.xlabel('故障类别', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for bar, count in zip(bars2, pred_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # 3. 聚类分布
    plt.subplot(1, 4, 3)
    cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
    cluster_labels = [f'聚类{i}' for i in cluster_unique]
    bars3 = plt.bar(cluster_labels, cluster_counts, color='lightcoral', alpha=0.7, edgecolor='black')
    plt.title('目标域聚类分布', fontsize=14, fontweight='bold')
    plt.xlabel('聚类标签', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for bar, count in zip(bars3, cluster_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # 4. 预测类别 vs 聚类对比
    plt.subplot(1, 4, 4)
    # 创建交叉表
    cross_tab = pd.crosstab(y_pred_tgt, clusters, margins=True)
    
    # 绘制热图
    import seaborn as sns
    sns.heatmap(cross_tab.iloc[:-1, :-1], annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': '样本数量'})
    plt.title('预测类别 vs 聚类交叉表', fontsize=14, fontweight='bold')
    plt.xlabel('聚类标签', fontsize=12)
    plt.ylabel('预测类别', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细统计信息
    print("\n" + "="*60)
    print("详细类别分布统计")
    print("="*60)
    
    print("\n1. 源域类别分布:")
    for cls, count in zip(src_unique, src_counts):
        percentage = count / len(y_src) * 100
        print(f"   {cls}: {count} 个样本 ({percentage:.1f}%)")
    
    print(f"\n2. 目标域预测类别分布:")
    for cls, count in zip(pred_unique, pred_counts):
        percentage = count / len(y_pred_tgt) * 100
        print(f"   {cls}: {count} 个样本 ({percentage:.1f}%)")
    
    print(f"\n3. 目标域聚类分布:")
    for cluster, count in zip(cluster_unique, cluster_counts):
        percentage = count / len(clusters) * 100
        print(f"   聚类{cluster}: {count} 个样本 ({percentage:.1f}%)")
    
    print(f"\n4. 预测类别与聚类的交叉统计:")
    print(cross_tab)

def plot_prediction_confidence(y_proba_tgt, y_pred_tgt):
    """绘制预测置信度分布图"""
    plt.figure(figsize=(15, 5))
    
    # 1. 整体置信度分布
    plt.subplot(1, 3, 1)
    max_probs = np.max(y_proba_tgt, axis=1)
    plt.hist(max_probs, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    plt.title('预测置信度分布', fontsize=14, fontweight='bold')
    plt.xlabel('最大预测概率', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.axvline(np.mean(max_probs), color='red', linestyle='--', 
               label=f'平均置信度: {np.mean(max_probs):.3f}')
    plt.legend()
    
    # 2. 各类别的平均置信度
    plt.subplot(1, 3, 2)
    unique_classes = np.unique(y_pred_tgt)
    avg_confidences = []
    
    for cls in unique_classes:
        mask = (y_pred_tgt == cls)
        cls_probs = y_proba_tgt[mask]
        avg_conf = np.mean(np.max(cls_probs, axis=1))
        avg_confidences.append(avg_conf)
    
    bars = plt.bar(unique_classes, avg_confidences, color='lightgreen', alpha=0.7, edgecolor='black')
    plt.title('各类别平均预测置信度', fontsize=14, fontweight='bold')
    plt.xlabel('故障类别', fontsize=12)
    plt.ylabel('平均置信度', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # 添加数值标签
    for bar, conf in zip(bars, avg_confidences):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 预测概率矩阵热图
    plt.subplot(1, 3, 3)
    import seaborn as sns
    
    # 计算每个类别的平均预测概率
    prob_matrix = []
    class_names = []
    
    for i, cls in enumerate(unique_classes):
        mask = (y_pred_tgt == cls)
        if np.sum(mask) > 0:
            avg_probs = np.mean(y_proba_tgt[mask], axis=0)
            prob_matrix.append(avg_probs)
            class_names.append(cls)
    
    prob_matrix = np.array(prob_matrix)
    
    sns.heatmap(prob_matrix, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=[f'类别{i}' for i in range(prob_matrix.shape[1])],
                yticklabels=class_names,
                cbar_kws={'label': '平均预测概率'})
    plt.title('各类别预测概率热图', fontsize=14, fontweight='bold')
    plt.xlabel('预测类别索引', fontsize=12)
    plt.ylabel('真实预测类别', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # 打印置信度统计
    print(f"\n预测置信度统计:")
    print(f"   平均置信度: {np.mean(max_probs):.4f}")
    print(f"   置信度标准差: {np.std(max_probs):.4f}")
    print(f"   最高置信度: {np.max(max_probs):.4f}")
    print(f"   最低置信度: {np.min(max_probs):.4f}")
    
    # 低置信度样本统计
    low_conf_threshold = 0.5
    low_conf_mask = max_probs < low_conf_threshold
    if np.sum(low_conf_mask) > 0:
        print(f"\n低置信度样本 (< {low_conf_threshold}):")
        print(f"   数量: {np.sum(low_conf_mask)} 个样本")
        print(f"   占比: {np.sum(low_conf_mask)/len(max_probs)*100:.1f}%")
        
        low_conf_classes = y_pred_tgt[low_conf_mask]
        unique_low, counts_low = np.unique(low_conf_classes, return_counts=True)
        for cls, count in zip(unique_low, counts_low):
            print(f"   {cls}: {count} 个样本")

def visualize_results(model, X_src, X_tgt, y_src, y_pred_tgt, y_proba_tgt, clusters):
    """可视化结果，使用模型提取的特征"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # 重塑数据为序列格式
    seq_len = min(20, X_src.shape[1])
    input_dim = X_src.shape[1] // seq_len
    
    X_src_seq = X_src[:, :seq_len*input_dim].reshape(-1, seq_len, input_dim)
    X_tgt_seq = X_tgt[:, :seq_len*input_dim].reshape(-1, seq_len, input_dim)
    
    # 提取 CDAN 对齐后的特征
    with torch.no_grad():
        X_src_tensor = torch.FloatTensor(X_src_seq).to(device)
        X_tgt_tensor = torch.FloatTensor(X_tgt_seq).to(device)
        
        src_features = model.feature_extractor(X_src_tensor).cpu().numpy()
        tgt_features = model.feature_extractor(X_tgt_tensor).cpu().numpy()
    
    # PCA 降维
    pca = PCA(n_components=2, random_state=42)
    X_all_features = np.vstack([src_features, tgt_features])
    X_pca = pca.fit_transform(X_all_features)
    
    n_src = len(src_features)
    X_src_pca = X_pca[:n_src]
    X_tgt_pca = X_pca[n_src:]
    
    plt.figure(figsize=(15, 5))
    
    # 源域分布（CDAN 对齐后的特征空间）
    plt.subplot(1, 3, 1)
    le_vis = LabelEncoder()
    y_src_num = le_vis.fit_transform(y_src)
    scatter = plt.scatter(X_src_pca[:, 0], X_src_pca[:, 1], c=y_src_num, cmap='Set1', alpha=0.7)
    plt.title('源域数据分布 (CDAN 对齐特征)')
    plt.colorbar(scatter)
    
    # 目标域预测结果
    plt.subplot(1, 3, 2)
    y_pred_num = le_vis.transform(y_pred_tgt)
    scatter = plt.scatter(X_tgt_pca[:, 0], X_tgt_pca[:, 1], c=y_pred_num, cmap='Set1', alpha=0.7)
    plt.title('目标域预测结果 (CDAN 对齐特征)')
    plt.colorbar(scatter)
    
    # 聚类结果
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(X_tgt_pca[:, 0], X_tgt_pca[:, 1], c=clusters, cmap='Set2', alpha=0.7)
    plt.title('目标域聚类结果')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    # 绘制类别分布图
    plot_class_distribution(y_src, y_pred_tgt, clusters)
    
    # 绘制预测置信度分析
    plot_prediction_confidence(y_proba_tgt, y_pred_tgt)

def create_classification_table_visualization(classification_table):
    """创建分类表格的可视化图表"""
    plt.figure(figsize=(20, 12))
    
    # 1. 样本分类结果条形图
    plt.subplot(2, 3, 1)
    samples = classification_table['目标域样本']
    predictions = classification_table['预测类别']
    colors = ['red' if pred == 'B' else 'green' if pred == 'IR' else 'blue' if pred == 'N' else 'orange' 
              for pred in predictions]
    
    bars = plt.bar(samples, [1]*len(samples), color=colors, alpha=0.7, edgecolor='black')
    plt.title('目标域样本分类结果', fontsize=14, fontweight='bold')
    plt.xlabel('目标域样本', fontsize=12)
    plt.ylabel('分类结果', fontsize=12)
    plt.xticks(rotation=45)
    
    # 添加图例
    unique_classes = classification_table['预测类别'].unique()
    legend_colors = {'B': 'red', 'IR': 'green', 'N': 'blue', 'OR': 'orange'}
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=legend_colors.get(cls, 'gray'), 
                                   alpha=0.7, label=cls) for cls in unique_classes]
    plt.legend(handles=legend_elements, title='故障类别')
    
    # 在每个条形上添加类别标签
    for bar, pred in zip(bars, predictions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, pred,
                ha='center', va='center', fontweight='bold', fontsize=10)
    
    # 2. 置信度热图
    plt.subplot(2, 3, 2)
    confidence_matrix = classification_table[['预测置信度']].values.reshape(1, -1)
    im = plt.imshow(confidence_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    plt.title('预测置信度热图', fontsize=14, fontweight='bold')
    plt.xlabel('目标域样本', fontsize=12)
    plt.xticks(range(len(samples)), samples, rotation=45)
    plt.yticks([0], ['置信度'])
    plt.colorbar(im, label='置信度')
    
    # 在每个格子中添加置信度数值
    for i, conf in enumerate(classification_table['预测置信度']):
        plt.text(i, 0, f'{conf:.3f}', ha='center', va='center', 
                fontweight='bold', fontsize=8)
    
    # 3. 各类别概率堆叠图
    plt.subplot(2, 3, 3)
    prob_cols = [col for col in classification_table.columns if col.endswith('_概率')]
    bottom = np.zeros(len(samples))
    colors_prob = ['red', 'green', 'blue', 'orange']
    
    for i, col in enumerate(prob_cols):
        class_name = col.replace('_概率', '')
        plt.bar(samples, classification_table[col], bottom=bottom, 
               label=class_name, color=colors_prob[i % len(colors_prob)], alpha=0.7)
        bottom += classification_table[col]
    
    plt.title('各样本类别概率分布', fontsize=14, fontweight='bold')
    plt.xlabel('目标域样本', fontsize=12)
    plt.ylabel('概率', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='故障类别')
    plt.ylim(0, 1)
    
    # 4. 置信度等级饼图
    plt.subplot(2, 3, 4)
    conf_level_counts = classification_table['置信度等级'].value_counts()
    colors_pie = ['green', 'yellow', 'red']
    plt.pie(conf_level_counts.values, labels=conf_level_counts.index, autopct='%1.1f%%',
           colors=colors_pie[:len(conf_level_counts)], startangle=90)
    plt.title('置信度等级分布', fontsize=14, fontweight='bold')
    
    # 5. 聚类 vs 预测对比
    plt.subplot(2, 3, 5)
    scatter_colors = [colors[i] for i in range(len(samples))]
    plt.scatter(classification_table['聚类标签'], classification_table['预测类别'], 
               c=scatter_colors, s=100, alpha=0.7, edgecolor='black')
    
    for i, sample in enumerate(samples):
        plt.annotate(sample, (classification_table.iloc[i]['聚类标签'], 
                             list(unique_classes).index(classification_table.iloc[i]['预测类别'])),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('聚类标签 vs 预测类别', fontsize=14, fontweight='bold')
    plt.xlabel('聚类标签', fontsize=12)
    plt.ylabel('预测类别索引', fontsize=12)
    
    # 6. 置信度分布直方图
    plt.subplot(2, 3, 6)
    plt.hist(classification_table['预测置信度'], bins=10, color='skyblue', 
            alpha=0.7, edgecolor='black')
    plt.axvline(classification_table['预测置信度'].mean(), color='red', 
               linestyle='--', label=f'平均值: {classification_table["预测置信度"].mean():.3f}')
    plt.title('预测置信度分布', fontsize=14, fontweight='bold')
    plt.xlabel('置信度', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('目标域A-P分类结果可视化.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   分类结果可视化图表已保存为 '目标域A-P分类结果可视化.png'")

def print_academic_table(classification_table):
    """打印学术风格的分类结果表格"""
    print("\n" + "="*120)
    print("表 1: 目标域样本 A-P 故障诊断分类结果")
    print("="*120)
    print("Table 1: Fault Diagnosis Classification Results for Target Domain Samples A-P")
    print("="*120)
    
    # 表头
    header = f"{'样本':<8} {'预测类别':<10} {'置信度':<10} {'置信度等级':<12} {'聚类标签':<10}"
    prob_headers = []
    for col in classification_table.columns:
        if col.endswith('_概率'):
            class_name = col.replace('_概率', '')
            header += f" {class_name+'概率':<10}"
            prob_headers.append(col)
    
    print(header)
    print("-" * len(header))
    
    # 数据行
    for _, row in classification_table.iterrows():
        line = f"{row['目标域样本']:<8} {row['预测类别']:<10} {row['预测置信度']:<10.4f} {row['置信度等级']:<12} {row['聚类标签']:<10}"
        for prob_col in prob_headers:
            line += f" {row[prob_col]:<10.4f}"
        print(line)
    
    print("-" * len(header))
    
    # 统计摘要
    print(f"\n统计摘要 (Statistical Summary):")
    print(f"  总样本数 (Total Samples): {len(classification_table)}")
    print(f"  平均置信度 (Average Confidence): {classification_table['预测置信度'].mean():.4f}")
    print(f"  置信度标准差 (Confidence Std): {classification_table['预测置信度'].std():.4f}")
    
    # 各类别统计
    class_stats = classification_table.groupby('预测类别').agg({
        '目标域样本': 'count',
        '预测置信度': ['mean', 'std']
    }).round(4)
    
    print(f"\n各类别统计 (Classification Statistics):")
    for pred_class in class_stats.index:
        count = class_stats.loc[pred_class, ('目标域样本', 'count')]
        mean_conf = class_stats.loc[pred_class, ('预测置信度', 'mean')]
        std_conf = class_stats.loc[pred_class, ('预测置信度', 'std')]
        percentage = count / len(classification_table) * 100
        print(f"  {pred_class}: {count} 个样本 ({percentage:.1f}%), 平均置信度: {mean_conf:.4f} ± {std_conf:.4f}")
    
    print("="*120)

# ============== 主函数 ==============

def main():
    """主程序入口"""
    print("开始基于 Transformer + CDAN + SVM 的目标域故障诊断...")
    
    # 1. 加载数据
    print("1. 加载源域和目标域数据...")
    try:
        # 源域数据
        df_src = pd.read_excel("./data/merged_data.xlsx")
        X_src = df_src.drop(columns=["label"]).fillna(method='ffill').values
        y_src = df_src["label"].values
        
        # 目标域数据
        X_tgt = pd.read_excel("./data/X_tgt_full.xlsx").fillna(method='ffill').values
        
        print(f"   源域数据形状: {X_src.shape}")
        print(f"   目标域数据形状: {X_tgt.shape}")
        print(f"   源域类别分布: {np.unique(y_src, return_counts=True)}")
        
    except FileNotFoundError as e:
        print(f"   错误：文件未找到 {e}")
        return None
    
    # 2. 数据预处理
    print("2. 数据预处理...")
    scaler = StandardScaler()
    X_src_scaled = scaler.fit_transform(X_src)
    X_tgt_scaled = scaler.transform(X_tgt)
    
    # 3. SVM 样本加权（基于原始标准化数据）
    print("3. 计算 SVM 样本权重...")
    sample_weights, domain_clf = compute_svm_sample_weights(X_src_scaled, X_tgt_scaled)
    print(f"   样本权重范围: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
    print(f"   平均权重: {sample_weights.mean():.4f}")
    
    # 4. 训练 CDAN Transformer 模型（包含 CDAN 对齐）
    print("4. 训练 CDAN Transformer 模型（包含 CDAN 对齐）...")
    model, label_encoder = train_cdan_transformer(
        X_src_scaled, y_src, X_tgt_scaled, 
        sample_weights=sample_weights,
        d_model=64, num_heads=4, num_layers=2, dropout=0.3,
        lr=1e-3, batch_size=32, epochs=100, alignment_weight=0.1
    )
    print("   模型训练完成（已集成 CDAN 对齐）")
    
    # 6. 目标域预测
    print("6. 目标域预测...")
    y_pred_tgt, y_proba_tgt = predict_target_domain(model, X_tgt_scaled, label_encoder)
    
    # 打印预测结果分布
    unique, counts = np.unique(y_pred_tgt, return_counts=True)
    print("   目标域预测类别分布：")
    for u, c in zip(unique, counts):
        print(f"   类别 {u}: {c} 条样本")
    
    # 7. 聚类一致性评估
    print("7. 聚类一致性评估...")
    ari, nmi, clusters = evaluate_clustering_consistency(y_pred_tgt, y_proba_tgt)
    print(f"   Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"   Normalized Mutual Information (NMI): {nmi:.4f}")
    
    # 8. 可视化结果
    print("8. 可视化结果...")
    visualize_results(model, X_src_scaled, X_tgt_scaled, y_src, y_pred_tgt, y_proba_tgt, clusters)
    
    # 9. 生成目标域 A-P 分类表格
    print("9. 生成目标域 A-P 分类表格...")
    target_samples = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    
    # 创建详细分类表格
    classification_table = pd.DataFrame({
        '目标域样本': target_samples[:len(y_pred_tgt)],  # 只取实际样本数量
        '预测类别': y_pred_tgt,
        '聚类标签': clusters,
        '预测置信度': np.max(y_proba_tgt, axis=1)
    })
    
    # 添加各类别的预测概率
    for i, class_name in enumerate(label_encoder.classes_):
        classification_table[f'{class_name}_概率'] = y_proba_tgt[:, i]
    
    # 添加置信度等级
    def get_confidence_level(prob):
        if prob >= 0.8:
            return '高'
        elif prob >= 0.6:
            return '中'
        else:
            return '低'
    
    classification_table['置信度等级'] = classification_table['预测置信度'].apply(get_confidence_level)
    
    # 显示分类表格
    print("\n" + "="*80)
    print("目标域样本 A-P 分类结果表")
    print("="*80)
    print(classification_table.round(4))
    
    # 按类别统计
    print("\n" + "="*50)
    print("按预测类别统计:")
    print("="*50)
    class_summary = classification_table.groupby('预测类别').agg({
        '目标域样本': 'count',
        '预测置信度': ['mean', 'std', 'min', 'max']
    }).round(4)
    class_summary.columns = ['样本数量', '平均置信度', '置信度标准差', '最低置信度', '最高置信度']
    print(class_summary)
    
    # 按置信度等级统计
    print("\n" + "="*50)
    print("按置信度等级统计:")
    print("="*50)
    confidence_summary = classification_table.groupby('置信度等级').agg({
        '目标域样本': 'count',
        '预测置信度': 'mean'
    }).round(4)
    confidence_summary.columns = ['样本数量', '平均置信度']
    print(confidence_summary)
    
    # 详细样本信息表
    print("\n" + "="*100)
    print("详细样本信息表 (按预测置信度降序排列)")
    print("="*100)
    detailed_table = classification_table.sort_values('预测置信度', ascending=False)
    
    # 格式化显示
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(detailed_table)
    
    # 保存完整结果
    classification_table.to_csv('目标域A-P分类结果表_CDAN_Transformer.csv', index=False, encoding='utf-8-sig')
    detailed_table.to_csv('目标域A-P详细分类结果_CDAN_Transformer.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n分类结果已保存到:")
    print(f"   - 目标域A-P分类结果表_CDAN_Transformer.csv")
    print(f"   - 目标域A-P详细分类结果_CDAN_Transformer.csv")
    
    # 生成分类结果摘要
    print("\n" + "="*60)
    print("分类结果摘要")
    print("="*60)
    
    for i, (sample, pred_class, confidence, conf_level) in enumerate(zip(
        classification_table['目标域样本'], 
        classification_table['预测类别'],
        classification_table['预测置信度'],
        classification_table['置信度等级']
    )):
        print(f"样本 {sample}: {pred_class} (置信度: {confidence:.3f} - {conf_level})")
    
    # 异常样本识别
    low_confidence_samples = classification_table[classification_table['预测置信度'] < 0.5]
    if len(low_confidence_samples) > 0:
        print(f"\n⚠️  低置信度样本 (< 0.5):")
        for _, row in low_confidence_samples.iterrows():
            print(f"   样本 {row['目标域样本']}: {row['预测类别']} (置信度: {row['预测置信度']:.3f})")
    else:
        print(f"\n✅ 所有样本预测置信度均 >= 0.5")
    
    # 生成分类表格可视化
    print("\n10. 生成分类表格可视化...")
    create_classification_table_visualization(classification_table)
    
    # 打印学术风格表格
    print_academic_table(classification_table)
    
    print("基于 Transformer + CDAN + SVM 的目标域故障诊断完成！")
    
    return {
        'model': model,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'predictions': y_pred_tgt,
        'probabilities': y_proba_tgt,
        'clusters': clusters,
        'metrics': {'ARI': ari, 'NMI': nmi}
    }

if __name__ == "__main__":
    results = main()
