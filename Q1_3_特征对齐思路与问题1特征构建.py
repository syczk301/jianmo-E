"""
特征对齐思路与问题1特征构建脚本
功能：实现源域和目标域的特征提取、对齐和可视化分析
作者：华为杯研赛E题代码助攻

方案说明：
方案 A（最稳健）：做通道无关的特征对齐
- 不假设目标域属于 DE/FE/BA 的哪一个，把源域三通道"揉"成一种通道鲁棒特征
- 统一采样率与长度：全部重采样到 12 kHz、长度 96k（8 秒）
- 按转速归一化频轴：以转频 fr 为基准，把频率轴/特征都除以 fr
- 提取通道鲁棒特征：时域统计、频域、时频域特征
- 跨通道聚合（源域）：对每个源域样本的三路分别提特征，然后在特征层做聚合
- 域对齐：用 CORAL/MMD/DANN 等在特征层做对齐
"""

import numpy as np
import pandas as pd
from scipy.signal import hilbert, welch, stft
from scipy.stats import kurtosis, skew
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings


def setup_plotting():
    """设置中文字体和忽略警告"""
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
        "mean": np.mean(x),
        "std": np.std(x),
        "var": np.var(x),
        "rms": rms,
        "peak": peak,
        "p2p": np.ptp(x),
        "mean_abs": mean_abs,
        "shape_factor": rms / (mean_abs + 1e-12),
        "crest_factor": peak / (rms + 1e-12),
        "impulse_factor": peak / (mean_abs + 1e-12),
        "margin_factor": peak / (sqr_mean**2 + 1e-12),
        "clearance_factor": peak / (sqr_mean + 1e-12),
        "kurtosis": kurtosis(x),
        "skewness": skew(x),
    }
    return feats


def freq_features(x, fs, fr):
    """频域特征（转速归一化）"""
    f, Pxx = welch(x, fs=fs, nperseg=2048)
    Pxx = Pxx / np.sum(Pxx)  # 归一化功率谱

    # 转频归一化
    f_norm = f / fr  

    feats = {
        "spec_centroid": np.sum(f_norm * Pxx),
        "spec_bandwidth": np.sqrt(np.sum(((f_norm - np.mean(f_norm))**2) * Pxx)),
        "spec_skewness": skew(Pxx),
        "spec_kurtosis": kurtosis(Pxx),
        "spec_entropy": -np.sum(Pxx * np.log(Pxx + 1e-12)),
    }

    # 带通能量（按 fr 倍频划分）
    bands = [(0.8,1.2),(1.8,2.2),(2.8,3.2),(4.5,5.5)]
    for i,(lo,hi) in enumerate(bands,1):
        mask = (f_norm>=lo)&(f_norm<=hi)
        feats[f"band_energy_{i}"] = np.sum(Pxx[mask])
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
        "tf_entropy": -np.sum(power * np.log(power + 1e-12)),
        "tf_mean_freq": np.sum(np.mean(power,axis=1)*f_norm),
    }

    # --- 小波分解 ---
    coeffs = pywt.wavedec(x, 'db4', level=4)
    energy = np.array([np.sum(c**2) for c in coeffs])
    energy_ratio = energy / (np.sum(energy)+1e-12)
    for i,e in enumerate(energy_ratio):
        feats[f"wavelet_energy_{i}"] = e
    return feats


def extract_features_from_long(long_table, signal_col="DE_time", fs=32000):
    """
    从长格式数据表中提取特征
    输入: long_table (含 file/status/.../signal_col/RPM)
    输出: X (特征矩阵), y (标签)
    """
    feature_rows = []
    labels = []

    for fid, group in long_table.groupby("file"):
        x = group[signal_col].dropna().values
        if len(x) == 0: 
            continue

        # 获取转速 fr (Hz)
        rpm = group["RPM"].iloc[0]
        fr = (rpm/60.0) if rpm and rpm>0 else 1.0

        feats = {}
        feats.update(time_features(x))
        feats.update(freq_features(x, fs, fr))
        feats.update(tf_features(x, fs, fr))

        feats["file"] = fid
        feature_rows.append(feats)
        labels.append(group["status"].iloc[0])

    X = pd.DataFrame(feature_rows).set_index("file")
    y = pd.Series(labels, index=X.index, name="status")
    return X, y


def extract_features_from_target_long(tgt_long, signal_col="Xtime", fs=32000):
    """
    从目标域长格式数据中提取特征
    输入: tgt_long (含 file/load/RPM/Xtime)
    输出: X (特征矩阵 DataFrame, index=file)
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


def load_source_data(source_csv="源域数据32khz整理.csv"):
    """加载源域数据"""
    try:
        long_table = pd.read_csv(source_csv)
        print(f"源域数据加载成功，维度：{long_table.shape}")
        return long_table
    except FileNotFoundError:
        print(f"警告：未找到文件 {source_csv}")
        print("请确保先运行 Q1_1_源域数据整合.py 生成源域数据")
        return None


def load_target_data(target_csv="目标域数据整理.csv"):
    """加载目标域数据"""
    try:
        tgt_long = pd.read_csv(target_csv)
        print(f"目标域数据加载成功，维度：{tgt_long.shape}")
        return tgt_long
    except FileNotFoundError:
        print(f"警告：未找到文件 {target_csv}")
        print("请确保先运行 Q1_2_目标域数据整理.py 生成目标域数据")
        return None


def extract_multi_channel_features(long_table, fs=32000):
    """提取源域多通道特征"""
    print("提取源域多通道特征...")
    
    # 提取三路特征
    X_src_DE, y_src = extract_features_from_long(long_table, signal_col="DE_time", fs=fs)
    X_src_FE, _     = extract_features_from_long(long_table, signal_col="FE_time", fs=fs)
    X_src_BA, _     = extract_features_from_long(long_table, signal_col="BA_time", fs=fs)
    
    print(f"DE通道特征维度：{X_src_DE.shape}")
    print(f"FE通道特征维度：{X_src_FE.shape}")
    print(f"BA通道特征维度：{X_src_BA.shape}")
    
    return X_src_DE, X_src_FE, X_src_BA, y_src


def normalize_features(X_src_DE, X_src_FE, X_src_BA, X_tgt):
    """特征标准化"""
    print("进行特征标准化...")
    
    # 拼接所有数据，保证用同一套缩放参数
    all_data = pd.concat([X_src_DE, X_src_FE, X_src_BA, X_tgt], axis=0)

    scaler = StandardScaler()
    scaler.fit(all_data.values)

    # 分别归一化
    X_src_DE_norm = pd.DataFrame(scaler.transform(X_src_DE), 
                                 index=X_src_DE.index, columns=X_src_DE.columns)
    X_src_FE_norm = pd.DataFrame(scaler.transform(X_src_FE), 
                                 index=X_src_FE.index, columns=X_src_FE.columns)
    X_src_BA_norm = pd.DataFrame(scaler.transform(X_src_BA), 
                                 index=X_src_BA.index, columns=X_src_BA.columns)
    X_tgt_norm    = pd.DataFrame(scaler.transform(X_tgt), 
                                 index=X_tgt.index, columns=X_tgt.columns)
    
    return X_src_DE_norm, X_src_FE_norm, X_src_BA_norm, X_tgt_norm, scaler


def plot_feature_comparison_scatter(X_src_DE, X_src_FE, X_src_BA, X_tgt, common_features, save_plots=False):
    """绘制特征散点图比较"""
    print("绘制特征散点图比较...")
    
    for feat in common_features:
        plt.figure(figsize=(8,5))

        # 源域三通道
        plt.scatter([0]*len(X_src_DE), X_src_DE[feat], alpha=0.6, label="Source-DE")
        plt.scatter([1]*len(X_src_FE), X_src_FE[feat], alpha=0.6, label="Source-FE")
        plt.scatter([2]*len(X_src_BA), X_src_BA[feat], alpha=0.6, label="Source-BA")

        # 目标域
        plt.scatter([3]*len(X_tgt), X_tgt[feat], alpha=0.6, label="Target-Xtime", color="red")

        plt.xticks([0,1,2,3], ["DE_time","FE_time","BA_time","Xtime"])
        plt.ylabel(feat)
        plt.title(f"Feature Comparison: {feat}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'scatter_{feat}.png', dpi=150, bbox_inches='tight')
        plt.show()


def plot_feature_comparison_boxplot(X_src_DE, X_src_FE, X_src_BA, X_tgt, common_features, save_plots=False):
    """绘制特征箱线图比较"""
    print("绘制特征箱线图比较...")
    
    for feat in common_features:
        plt.figure(figsize=(8,5))
        
        data = [
            X_src_DE[feat].dropna().values,
            X_src_FE[feat].dropna().values,
            X_src_BA[feat].dropna().values,
            X_tgt[feat].dropna().values
        ]
        labels = ["DE_time","FE_time","BA_time","Xtime"]
        
        plt.boxplot(data, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor="lightblue", color="blue"),
                    medianprops=dict(color="red", linewidth=2),
                    whiskerprops=dict(color="blue"),
                    capprops=dict(color="blue"))
        
        plt.ylabel(feat)
        plt.title(f"Feature Distribution Comparison: {feat}")
        plt.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'boxplot_{feat}.png', dpi=150, bbox_inches='tight')
        plt.show()


def plot_normalized_feature_comparison(X_src_DE_norm, X_src_FE_norm, X_src_BA_norm, X_tgt_norm, common_features, save_plots=False):
    """绘制标准化后的特征比较（箱线图+散点）"""
    print("绘制标准化后的特征比较...")
    
    for feat in common_features:
        plt.figure(figsize=(8,5))

        data = [
            X_src_DE_norm[feat].dropna().values,
            X_src_FE_norm[feat].dropna().values,
            X_src_BA_norm[feat].dropna().values,
            X_tgt_norm[feat].dropna().values
        ]
        labels = ["DE_time","FE_time","BA_time","Xtime"]

        # --- 箱线图 ---
        plt.boxplot(
            data, labels=labels, patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="blue", alpha=0.5),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(color="blue"),
            capprops=dict(color="blue"),
            flierprops=dict(marker='')  # 禁用箱线图自带异常值点
        )

        # --- 散点 (添加抖动避免重叠) ---
        for i, vals in enumerate(data, start=1):
            x_jitter = np.random.normal(i, 0.05, size=len(vals))  # 在 x 轴添加小扰动
            plt.scatter(x_jitter, vals, alpha=0.5, s=15, edgecolor='k')

        plt.ylabel(feat)
        plt.title(f"Feature Distribution Comparison (Normalized): {feat}")
        plt.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'normalized_{feat}.png', dpi=150, bbox_inches='tight')
        plt.show()


def analyze_feature_alignment(X_src_DE, X_src_FE, X_src_BA, X_tgt):
    """分析特征对齐情况"""
    print("=== 特征对齐分析 ===")
    
    # 确保特征名对齐
    common_features = sorted(set(X_src_DE.columns) & set(X_tgt.columns))
    print(f"共有 {len(common_features)} 个特征可比较")
    
    # 计算每个特征在不同通道间的相关性
    correlations = {}
    for feat in common_features:
        src_de = X_src_DE[feat].values
        src_fe = X_src_FE[feat].values  
        src_ba = X_src_BA[feat].values
        tgt = X_tgt[feat].values
        
        # 计算目标域与各源域通道的相关性
        corr_de = np.corrcoef(tgt, src_de[:len(tgt)])[0,1] if len(tgt) <= len(src_de) else np.nan
        corr_fe = np.corrcoef(tgt, src_fe[:len(tgt)])[0,1] if len(tgt) <= len(src_fe) else np.nan  
        corr_ba = np.corrcoef(tgt, src_ba[:len(tgt)])[0,1] if len(tgt) <= len(src_ba) else np.nan
        
        correlations[feat] = {
            'DE': corr_de,
            'FE': corr_fe, 
            'BA': corr_ba
        }
    
    # 转换为DataFrame便于分析
    corr_df = pd.DataFrame(correlations).T
    print("\n各特征与源域通道的相关性：")
    print(corr_df.round(3))
    
    return common_features, corr_df


def main():
    """主程序入口"""
    print("开始特征对齐思路与问题1特征构建...")
    
    # 设置绘图
    setup_plotting()
    
    # 1. 加载数据
    print("1. 加载数据...")
    long_table = load_source_data()
    tgt_long = load_target_data()
    
    if long_table is None or tgt_long is None:
        print("数据加载失败，请检查文件路径")
        return None
    
    # 2. 提取源域多通道特征
    print("2. 提取源域多通道特征...")
    X_src_DE, X_src_FE, X_src_BA, y_src = extract_multi_channel_features(long_table)
    
    # 3. 提取目标域特征
    print("3. 提取目标域特征...")
    X_tgt = extract_features_from_target_long(tgt_long, signal_col="Xtime", fs=32000)
    print(f"目标域特征维度：{X_tgt.shape}")
    
    # 4. 特征对齐分析
    print("4. 特征对齐分析...")
    common_features, corr_df = analyze_feature_alignment(X_src_DE, X_src_FE, X_src_BA, X_tgt)
    
    # 5. 特征标准化
    print("5. 特征标准化...")
    X_src_DE_norm, X_src_FE_norm, X_src_BA_norm, X_tgt_norm, scaler = normalize_features(
        X_src_DE, X_src_FE, X_src_BA, X_tgt)
    
    # 6. 可视化分析
    print("6. 可视化分析...")
    plot_choice = input("选择可视化类型 (1:散点图, 2:箱线图, 3:标准化比较, 4:全部, 0:跳过): ")
    
    if plot_choice == '0':
        print("跳过可视化分析")
    else:
        save_plots = input("是否保存图片？(y/n): ").lower() == 'y'
        
        if plot_choice in ['1', '4']:
            plot_feature_comparison_scatter(X_src_DE, X_src_FE, X_src_BA, X_tgt, common_features, save_plots)
        
        if plot_choice in ['2', '4']:
            plot_feature_comparison_boxplot(X_src_DE, X_src_FE, X_src_BA, X_tgt, common_features, save_plots)
        
        if plot_choice in ['3', '4']:
            plot_normalized_feature_comparison(X_src_DE_norm, X_src_FE_norm, X_src_BA_norm, X_tgt_norm, common_features, save_plots)
    
    print("特征对齐分析完成！")
    
    return {
        'source_features': {'DE': X_src_DE, 'FE': X_src_FE, 'BA': X_src_BA},
        'source_labels': y_src,
        'target_features': X_tgt,
        'normalized_features': {
            'DE': X_src_DE_norm, 'FE': X_src_FE_norm, 
            'BA': X_src_BA_norm, 'target': X_tgt_norm
        },
        'common_features': common_features,
        'correlations': corr_df,
        'scaler': scaler
    }


if __name__ == "__main__":
    results = main()
