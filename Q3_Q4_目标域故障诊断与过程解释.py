"""
目标域故障诊断与过程解释脚本 - 完整版
功能：完整实现域适应、目标域预测、聚类分析和SHAP解释（按原版Jupyter笔记本复刻）
作者：华为杯研赛E题代码助攻
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import random
from scipy.signal import hilbert, welch, stft
from scipy.stats import kurtosis, skew
import pywt
from numpy.linalg import eigh

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report, 
                           normalized_mutual_info_score, adjusted_rand_score, f1_score)
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

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


def time_features(x):
    """时域统计特征"""
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
        rpm = group["RPM"].iloc[0]
        fr = (rpm / 60.0) if (pd.notna(rpm) and rpm > 0) else 1.0
        status = group["status"].iloc[0]

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

        rpm = group["RPM"].iloc[0]
        fr = (rpm/60.0) if rpm and rpm>0 else 1.0

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


def plot_tsne_comparison(X_src, X_tgt, domain_labels, class_labels=None, title="t-SNE 特征分布"):
    """绘制t-SNE特征分布对比"""
    X_all = np.vstack([X_src, X_tgt])
    
    # 调整perplexity
    n_samples = X_all.shape[0]
    perplexity = max(5, min(30, n_samples//3))
    
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
    perplexity = max(1, min(30, n_samples // 3))

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


def main():
    """主程序入口"""
    print("开始目标域故障诊断与过程解释（完整版）...")
    
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
    
    # 5. 域适应方法对比
    print("5. 域适应方法对比...")
    
    # 5.1 无适应基线
    print("5.1 训练无适应基线模型...")
    clf_baseline = RandomForestClassifier(random_state=42)
    clf_baseline.fit(X_src_filtered, y_src_filtered)
    y_tgt_pred_baseline = clf_baseline.predict(X_tgt_scaled)
    plot_prediction_distribution(y_tgt_pred_baseline, "无适应基线 - 目标域预测分布")
    
    # 5.2 CORAL适应
    print("5.2 CORAL域适应...")
    X_src_coral = coral_alignment(X_src_filtered, X_tgt_scaled)
    clf_coral = RandomForestClassifier(random_state=42)
    clf_coral.fit(X_src_coral, y_src_filtered)
    y_tgt_pred_coral = clf_coral.predict(X_tgt_scaled)
    plot_prediction_distribution(y_tgt_pred_coral, "CORAL适应 - 目标域预测分布")
    
    # 5.3 TCA适应
    print("5.3 TCA域适应...")
    try:
        Zs, Zt = TCA(X_src_filtered, X_tgt_scaled, dim=min(20, X_src_filtered.shape[1]), mu=1.0)
        clf_tca = RandomForestClassifier(random_state=42)
        clf_tca.fit(Zs, y_src_filtered)
        y_tgt_pred_tca = clf_tca.predict(Zt)
        plot_prediction_distribution(y_tgt_pred_tca, "TCA适应 - 目标域预测分布")
        
        # TCA聚类一致性
        kmeans_tca = KMeans(n_clusters=len(np.unique(y_src_filtered)), random_state=42)
        y_kmeans_tca = kmeans_tca.fit_predict(Zt)
        nmi_tca = normalized_mutual_info_score(y_tgt_pred_tca, y_kmeans_tca)
        ari_tca = adjusted_rand_score(y_tgt_pred_tca, y_kmeans_tca)
        print(f"TCA 目标域聚类一致性：NMI={nmi_tca:.3f}  ARI={ari_tca:.3f}")
        
    except Exception as e:
        print(f"TCA适应失败：{e}")
        y_tgt_pred_tca = y_tgt_pred_coral  # 使用CORAL结果作为备选
        clf_tca = None
    
    # 6. 可视化分析
    print("6. 可视化分析...")
    
    # 6.1 特征分布对比
    print("6.1 特征分布对比...")
    stats_diff = analyze_feature_distribution(X_src_filtered, X_tgt_scaled, selected_features)
    
    # 6.2 t-SNE可视化
    print("6.2 t-SNE可视化...")
    domain_labels = np.array(["Source"]*len(X_src_coral) + ["Target"]*len(X_tgt_scaled))
    class_labels = np.concatenate([y_src_filtered.values, ["Target"]*len(X_tgt_scaled)])
    plot_tsne_comparison(X_src_coral, X_tgt_scaled, domain_labels, class_labels, "CORAL适应后特征分布")
    
    # 7. 聚类一致性分析
    print("7. 聚类一致性分析...")
    n_clusters = len(np.unique(y_src_filtered))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X_tgt_scaled)
    
    # 计算一致性指标
    nmi = normalized_mutual_info_score(y_tgt_pred_coral, y_kmeans)
    ari = adjusted_rand_score(y_tgt_pred_coral, y_kmeans)
    print(f"聚类一致性：NMI={nmi:.3f}, ARI={ari:.3f}")
    
    plot_clustering_comparison(X_tgt_scaled, y_tgt_pred_coral, y_kmeans, "目标域聚类对比")
    
    # 8. 可解释性分析
    print("8. 可解释性分析...")
    shap_result = shap_analysis(clf_coral, X_src_coral, selected_features)
    if shap_result is not None:
        shap_values, explainer = shap_result
    else:
        shap_values, explainer = None, None
    
    # 9. 结果总结
    print("9. 结果总结...")
    print("="*50)
    print("目标域预测结果统计：")
    print(f"无适应基线：{pd.Series(y_tgt_pred_baseline).value_counts().to_dict()}")
    print(f"CORAL适应：{pd.Series(y_tgt_pred_coral).value_counts().to_dict()}")
    try:
        print(f"TCA适应：{pd.Series(y_tgt_pred_tca).value_counts().to_dict()}")
    except:
        pass
    print(f"聚类一致性：NMI={nmi:.3f}, ARI={ari:.3f}")
    print("="*50)
    
    print("目标域故障诊断与过程解释完成！")
    
    return {
        'source_data': (X_src_filtered, y_src_filtered),
        'target_data': (X_tgt, X_tgt_scaled),
        'models': {
            'baseline': clf_baseline,
            'coral': clf_coral,
            'tca': clf_tca if 'clf_tca' in locals() and clf_tca is not None else None
        },
        'predictions': {
            'baseline': y_tgt_pred_baseline,
            'coral': y_tgt_pred_coral,
            'tca': y_tgt_pred_tca if 'y_tgt_pred_tca' in locals() else None
        },
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


if __name__ == "__main__":
    results = main()
