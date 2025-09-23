# -*- coding: utf-8 -*-
"""
XGBoost模型 - 轴承故障诊断
基于Q2.ipynb的特征工程和数据处理方法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert, welch, stft
from scipy.stats import kurtosis, skew
import pywt
import random
import math
import time

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:
    print("请先安装xgboost: pip install xgboost")
    exit()

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------------------ 特征提取工具函数 ------------------ #

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
    基于原始 long 表进行采样与样本均衡：
    1) 先按文件提取原始特征；
    2) 对少数类用"随机二值掩码混合同类两条原始信号"的方式生成新信号，再提特征；
    3) 输出均衡后的特征矩阵 X 与标签 y（中文字段名）。
    """
    rng = np.random.default_rng(random_state)
    random.seed(random_state)

    # -------- Step 1：先按文件提取"原始样本"的特征，并缓存原始信号用于后续合成 --------
    feature_rows = []
    labels = []
    # 为后续合成准备一个缓存：每个状态 -> [(file_id, x_array, rpm), ...]
    raw_pool = {}

    for fid, group in long_table.groupby("file"):
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

    # -------- Step 2：统计每个类别当前数量，决定需要合成多少 --------
    counts = y.value_counts()
    # 只对少数类进行增广
    need_augment = {
        cls: max(0, target_per_class - cnt)
        for cls, cnt in counts.items()
    }

    # -------- Step 3：对少数类进行"随机掩码混合"合成新信号并提特征 --------
    aug_feature_rows = []
    aug_labels = []

    for cls, n_needed in need_augment.items():
        if n_needed <= 0:
            continue  # 该类已达到或超过目标，无需增广

        # 该类别的原始池
        pool = raw_pool.get(cls, [])
        if len(pool) == 0:
            print(f"警告：类别 {cls} 在原始数据中不存在，无法增广。")
            continue
        if len(pool) == 1:
            print(f"提示：类别 {cls} 只有 1 个原始样本，增广将以同一条信号自混合方式进行。")

        for k in range(n_needed):
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

    # -------- Step 4：合并原始与增广后的特征，并返回 --------
    if len(aug_feature_rows) > 0:
        X_aug = pd.DataFrame(aug_feature_rows).set_index("文件名")
        y_aug = pd.Series(aug_labels, index=X_aug.index, name="状态")
        X_balanced = pd.concat([X, X_aug], axis=0)
        y_balanced = pd.concat([y, y_aug], axis=0)
    else:
        X_balanced, y_balanced = X, y

    # 可选：检查均衡结果
    print("均衡后各类数量：")
    print(y_balanced.value_counts())

    return X_balanced, y_balanced

def load_data():
    """加载数据 - 支持从merged_data.xlsx直接加载"""
    import os
    # 尝试多个可能的路径
    possible_paths = [
        "../data/merged_data.xlsx",
        "./data/merged_data.xlsx", 
        "data/merged_data.xlsx",
        os.path.join("..", "data", "merged_data.xlsx")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"找到数据文件: {path}")
            df_loaded = pd.read_excel(path)
            X = df_loaded.drop(columns=["label"])
            y = df_loaded["label"]
            
            # 处理缺失值和无穷值
            print(f"原始数据形状: {X.shape}")
            print(f"缺失值数量: {X.isnull().sum().sum()}")
            
            if X.isnull().sum().sum() > 0:
                print("发现缺失值，进行处理...")
                X = X.fillna(X.mean())
                print("缺失值处理完成")
            
            # 处理无穷值
            X = X.replace([np.inf, -np.inf], np.nan)
            if X.isnull().sum().sum() > 0:
                X = X.fillna(X.mean())
            
            return X, y
    
    raise FileNotFoundError(f"未找到数据文件，尝试了以下路径: {possible_paths}")
    return None, None

def feature_selection_rfe(X, y, n_features=20):
    """使用RFE进行特征选择"""
    print(f"开始RFE特征选择，保留 {n_features} 个特征...")
    
    # 基学习器，这里用随机森林
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 构造RFE
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    
    # 拟合RFE
    selector = selector.fit(X, y)
    
    # 被选择的特征
    selected_features = X.columns[selector.support_]
    
    print("选择的特征：")
    print(selected_features.tolist())
    
    # 构造新的特征矩阵
    X_selected = X[selected_features]
    
    print(f"原始特征维度：{X.shape[1]}")
    print(f"筛选后特征维度：{X_selected.shape[1]}")
    
    return X_selected, selected_features

def train_xgboost(X_train, X_test, y_train, y_test):
    """训练XGBoost模型"""
    
    # 对标签进行编码（XGBoost要求数值标签）
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # 优化后的参数网格搜索（减少参数组合避免电脑卡死）
    param_grid = {
        'n_estimators': [100, 200],  # 减少参数
        'max_depth': [3, 5],         # 减少参数
        'learning_rate': [0.1, 0.2], # 减少参数
        'subsample': [0.8, 1.0],     # 减少参数
        'colsample_bytree': [0.8, 1.0], # 减少参数
        'reg_alpha': [0, 0.1],       # 减少参数
        'reg_lambda': [1, 1.5]       # 减少参数
    }
    
    # 创建XGBoost分类器（添加早停和资源限制）
    xgb_clf = XGBClassifier(
        random_state=42,
        n_jobs=4,  # 限制使用4个CPU核心，避免系统卡死
        eval_metric='mlogloss',  # 多分类对数损失
        early_stopping_rounds=10,  # 早停机制
        verbosity=0  # 减少输出
    )
    
    # 网格搜索（减少CV折数和并行数）
    print("开始网格搜索最优参数...")
    print(f"参数组合总数: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree']) * len(param_grid['reg_alpha']) * len(param_grid['reg_lambda'])}")
    
    grid_search = GridSearchCV(
        xgb_clf, param_grid, 
        cv=3,  # 减少交叉验证折数从5到3
        scoring='accuracy', 
        n_jobs=2,  # 限制并行数
        verbose=1
    )
    
    # 添加验证集用于早停
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train_encoded, test_size=0.2, random_state=42
    )
    
    grid_search.fit(
        X_train, y_train_encoded,
        eval_set=[(X_val_split, y_val_split)],
        verbose=False
    )
    
    print(f"最优参数: {grid_search.best_params_}")
    print(f"最优交叉验证分数: {grid_search.best_score_:.4f}")
    
    # 使用最优参数训练模型
    best_xgb = grid_search.best_estimator_
    
    # 预测
    y_pred_encoded = best_xgb.predict(X_test)
    y_pred_proba = best_xgb.predict_proba(X_test)
    
    # 将预测结果转换回原始标签
    y_pred = le.inverse_transform(y_pred_encoded)
    
    return best_xgb, y_pred, y_pred_proba, le

def evaluate_model(y_test, y_pred, y_pred_proba, model_name="XGBoost"):
    """评估模型性能"""
    
    # 基本指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n{model_name}模型性能评估:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 分类报告
    print(f"\n详细分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), 
                yticklabels=np.unique(y_test))
    plt.title(f'{model_name}模型混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC曲线（多分类）
    classes = np.unique(y_test)
    if len(classes) > 2:
        # 多分类ROC
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = y_test_bin.shape[1]
        
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title(f'{model_name}模型ROC曲线')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{model_name}_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_feature_importance(model, feature_names, model_name="XGBoost"):
    """绘制特征重要性"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'{model_name}模型特征重要性')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('重要性')
        plt.tight_layout()
        plt.savefig(f'{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印前10个最重要特征
        print(f"\n{model_name}模型前10个最重要特征:")
        for i in range(min(10, len(importances))):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def plot_xgboost_trees(model, feature_names, model_name="XGBoost"):
    """绘制XGBoost树结构（前几棵树）"""
    try:
        # 绘制前3棵树
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        for i in range(3):
            xgb.plot_tree(model, num_trees=i, ax=axes[i], rankdir='TB')
            axes[i].set_title(f'{model_name} 第{i+1}棵树')
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_trees.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"绘制树结构时出错: {e}")

def plot_training_history(model, X_train, y_train, X_test, y_test, le, model_name="XGBoost"):
    """绘制训练历史"""
    try:
        # 重新训练以获取评估历史
        y_train_encoded = le.transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        eval_set = [(X_train, y_train_encoded), (X_test, y_test_encoded)]
        
        # 创建新的模型实例并训练
        new_model = XGBClassifier(**model.get_params())
        new_model.fit(X_train, y_train_encoded, 
                     eval_set=eval_set, 
                     eval_metric='mlogloss',
                     verbose=False)
        
        # 获取评估结果
        results = new_model.evals_result()
        epochs = len(results['validation_0']['mlogloss'])
        x_axis = range(0, epochs)
        
        # 绘制训练和验证损失
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_axis, results['validation_0']['mlogloss'], label='训练集')
        ax.plot(x_axis, results['validation_1']['mlogloss'], label='验证集')
        ax.legend()
        ax.set_ylabel('对数损失')
        ax.set_xlabel('轮数')
        ax.set_title(f'{model_name}模型训练历史')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"绘制训练历史时出错: {e}")

def main():
    """主函数"""
    print("=" * 50)
    print("XGBoost模型 - 轴承故障诊断")
    print("=" * 50)
    
    # 加载数据
    print("加载数据...")
    X, y = load_data()
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"类别分布:\n{y.value_counts()}")
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 训练模型
    model, y_pred, y_pred_proba, le = train_xgboost(X_train, X_test, y_train, y_test)
    
    # 评估模型
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, "XGBoost")
    
    # 特征重要性
    plot_feature_importance(model, X.columns, "XGBoost")
    
    # XGBoost特有的可视化
    plot_xgboost_trees(model, X.columns, "XGBoost")
    plot_training_history(model, X_train, y_train, X_test, y_test, le, "XGBoost")
    
    # 保存模型性能结果
    results_df = pd.DataFrame([metrics])
    results_df.to_csv('XGBoost_performance_results.csv', index=False)
    print("\n模型性能结果已保存到 'XGBoost_performance_results.csv'")
    
    print("\nXGBoost模型分析完成！")

if __name__ == "__main__":
    main()
