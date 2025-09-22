"""
源域故障诊断脚本
功能：基于源域数据进行故障诊断模型训练和评估
包含：数据平衡、特征选择、PSO优化、深度学习模型等
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_curve, auc, confusion_matrix, classification_report, 
                           f1_score, normalized_mutual_info_score, adjusted_rand_score)
from sklearn.preprocessing import label_binarize

# 深度学习相关
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("警告：TensorFlow未安装，LSTM相关功能将不可用")
    TENSORFLOW_AVAILABLE = False


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


def extract_features_from_long(long_table, signal_col="DE_time", fs=32000):
    """
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

        feats["文件名"] = fid
        feature_rows.append(feats)
        labels.append(group["status"].iloc[0])

    X = pd.DataFrame(feature_rows).set_index("文件名")
    y = pd.Series(labels, index=X.index, name="状态")
    return X, y


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

    # Step 1：先按文件提取"原始样本"的特征，并缓存原始信号用于后续合成
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

    # Step 2：统计每个类别当前数量，决定需要合成多少
    counts = y.value_counts()
    need_augment = {
        cls: max(0, target_per_class - cnt)
        for cls, cnt in counts.items()
    }

    # Step 3：对少数类进行"随机掩码混合"合成新信号并提特征
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

            # 也可加一点极小高斯噪声，打破完全重复（可选）
            # noise = rng.normal(0, 1e-6*np.std(x_new) if np.std(x_new)>0 else 1e-6, size=L)
            # x_new = x_new + noise

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


def plot_label_distribution(y, title="标签分布统计"):
    """绘制标签分布图"""
    dist = y.value_counts()
    ratio = y.value_counts(normalize=True)

    print("标签分布：")
    print(pd.DataFrame({"数量": dist, "比例": ratio}))

    # 柱状图展示 + 标签
    plt.figure(figsize=(6,4))
    ax = dist.plot(kind="bar")
    plt.xlabel("状态")
    plt.ylabel("数量")
    plt.title(title)
    plt.xticks(rotation=45)

    # 在柱子上加标签
    for i, v in enumerate(dist):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def feature_selection_rfe(X, y, n_features=20):
    """使用RFE进行特征选择"""
    # 基学习器，这里用随机森林
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)

    # 构造RFE，选择保留的特征数
    selector = RFE(estimator, n_features_to_select=n_features, step=1)

    # 拟合RFE
    selector = selector.fit(X, y)

    # 被选择的特征
    selected_features = X.columns[selector.support_]

    print("选择的特征：")
    print(selected_features)

    # 构造新的特征矩阵
    X_selected = X[selected_features]

    print("原始特征维度：", X.shape[1])
    print("筛选后特征维度：", X_selected.shape[1])

    return X_selected, selected_features


def evaluate_model_performance(X_test, y_test, y_pred, y_pred_proba, classes, title="模型评估"):
    """评估模型性能并可视化"""
    # ROC曲线
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]

    plt.figure(figsize=(6,5))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{classes[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("假阳率 (FPR)")
    plt.ylabel("真正率 (TPR)")
    plt.title(f"{title} - ROC曲线")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title(f"{title} - 混淆矩阵")
    plt.tight_layout()
    plt.show()

    # F1 报告
    print(f"{title} - F1 分类报告：")
    print(classification_report(y_test, y_pred, target_names=classes))


class PSO_RandomForest:
    """PSO优化随机森林超参数"""
    
    def __init__(self, space, num_particles=20, num_iters=25, w=0.72, c1=1.49, c2=1.49):
        self.space = space
        self.num_particles = num_particles
        self.num_iters = num_iters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.default_rng(2025)
        
        # 初始化边界
        keys = list(space.keys())
        self.dim = len(keys)
        self.lows = np.array([space[k][0] for k in keys])
        self.highs = np.array([space[k][1] for k in keys])
        
    def decode_particle(self, vec):
        """将粒子实数向量映射为随机森林超参数字典"""
        params = {
            "n_estimators": int(round(np.clip(vec[0], self.lows[0], self.highs[0]))),
            "max_depth": int(round(np.clip(vec[1], self.lows[1], self.highs[1]))),
            "min_samples_split": int(round(np.clip(vec[2], self.lows[2], self.highs[2]))),
            "min_samples_leaf": int(round(np.clip(vec[3], self.lows[3], self.highs[3]))),
            "random_state": 42,
            "n_jobs": -1,
        }
        # 边界条件修正
        params["min_samples_split"] = max(params["min_samples_split"], params["min_samples_leaf"])
        return params

    def evaluate_particle(self, vec, X_tr, y_tr, X_val, y_val):
        """以验证集 F1_macro 为目标函数（越大越好）"""
        params = self.decode_particle(vec)
        clf_pso = RandomForestClassifier(**params)
        clf_pso.fit(X_tr, y_tr)
        pred_val = clf_pso.predict(X_val)
        f1 = f1_score(y_val, pred_val, average="macro")
        return f1

    def optimize(self, X_train, y_train, X_test, y_test):
        """PSO主过程"""
        # 初始化粒子
        positions = self.rng.uniform(self.lows, self.highs, size=(self.num_particles, self.dim))
        velocities = self.rng.uniform(-np.abs(self.highs-self.lows), 
                                     np.abs(self.highs-self.lows), 
                                     size=(self.num_particles, self.dim))*0.1

        # 评估初值
        pbest_pos = positions.copy()
        pbest_val = np.array([self.evaluate_particle(p, X_train, y_train, X_test, y_test) 
                             for p in positions])

        gbest_idx = int(np.argmax(pbest_val))
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_val = float(pbest_val[gbest_idx])

        print(f"[PSO] 初始最优 F1_macro = {gbest_val:.4f}，超参数 = {self.decode_particle(gbest_pos)}")

        t0 = time.time()
        for it in range(1, self.num_iters+1):
            # 更新速度与位置
            r1 = self.rng.random((self.num_particles, self.dim))
            r2 = self.rng.random((self.num_particles, self.dim))
            velocities = (
                self.w*velocities
                + self.c1*r1*(pbest_pos - positions)
                + self.c2*r2*(gbest_pos - positions)
            )
            positions = positions + velocities
            # 位置边界裁剪
            positions = np.minimum(np.maximum(positions, self.lows), self.highs)

            # 评估
            vals = np.array([self.evaluate_particle(p, X_train, y_train, X_test, y_test) 
                           for p in positions])

            # 更新个体最优
            improved = vals > pbest_val
            pbest_pos[improved] = positions[improved]
            pbest_val[improved] = vals[improved]

            # 更新全局最优
            if pbest_val.max() > gbest_val:
                gbest_idx = int(np.argmax(pbest_val))
                gbest_pos = pbest_pos[gbest_idx].copy()
                gbest_val = float(pbest_val[gbest_idx])

            if it % 5 == 0 or it == self.num_iters:
                print(f"[PSO] 迭代 {it:02d}/{self.num_iters}，当前最优 F1_macro = {gbest_val:.4f}")

        t1 = time.time()
        print(f"[PSO] 完成。耗时 {t1 - t0:.1f}s")
        best_params = self.decode_particle(gbest_pos)
        print("[PSO] 最优超参数：", best_params)

        return best_params, gbest_val


def build_lstm_model(input_timesteps, n_classes, units_lstm=64, dropout=0.3, dense_units=64, lr=1e-3):
    """构建LSTM模型"""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow未安装，无法使用LSTM模型")
        
    model = Sequential([
        LSTM(int(units_lstm), input_shape=(input_timesteps, 1), return_sequences=False),
        Dropout(dropout),
        Dense(int(dense_units), activation="relu"),
        Dropout(dropout),
        Dense(n_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_lstm_model(X_selected, y_bal, test_size=0.3, random_state=42):
    """训练LSTM模型"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow未安装，跳过LSTM训练")
        return None, None, None
        
    # 数据准备
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_bal, test_size=test_size, random_state=random_state, stratify=y_bal
    )

    # 标签编码
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    classes = le.classes_
    n_classes = len(classes)

    # One-hot，用于多分类交叉熵
    y_train_oh = to_categorical(y_train_enc, num_classes=n_classes)
    y_test_oh = to_categorical(y_test_enc, num_classes=n_classes)

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    # LSTM 输入需要 3D： (样本数, 时间步, 每步特征维度)
    timesteps = X_train_scaled.shape[1]
    X_train_lstm = X_train_scaled.reshape(-1, timesteps, 1)
    X_test_lstm = X_test_scaled.reshape(-1, timesteps, 1)

    # 固定随机种子
    np.random.seed(42)
    tf.random.set_seed(42)

    # 构建模型
    model = build_lstm_model(timesteps, n_classes)

    # 早停
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True
    )

    # 训练
    history = model.fit(
        X_train_lstm, y_train_oh,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # 预测
    y_proba = model.predict(X_test_lstm)
    y_pred_enc = np.argmax(y_proba, axis=1)
    y_pred = le.inverse_transform(y_pred_enc)

    return model, (X_test, y_test, y_pred, y_proba, classes), (scaler, le)


def main():
    """主程序入口"""
    print("开始源域故障诊断...")
    
    # 设置环境
    setup_environment()
    
    # 1. 加载数据
    print("1. 加载源域数据...")
    try:
        long_table = pd.read_csv('源域数据32khz整理.csv')
        print(f"   数据加载成功，维度：{long_table.shape}")
    except FileNotFoundError:
        print("   错误：未找到'源域数据32khz整理.csv'文件")
        print("   请先运行Q1_1_源域数据整合.py生成数据文件")
        return None
    
    # 2. 数据平衡和特征提取
    print("2. 数据平衡和特征提取...")
    X_bal, y_bal = extract_features_balanced_from_long(
        long_table,
        signal_col="DE_time",
        fs=32000,
        target_per_class=77,
        mix_ratio=0.5,
        random_state=42
    )

    # 验证特征矩阵
    print(f"   均衡后特征矩阵形状：{X_bal.shape}")
    print(f"   特征列：{list(X_bal.columns)}")

    # 绘制标签分布
    plot_label_distribution(y_bal, "均衡后标签分布")
    
    # 3. 特征选择
    print("3. 特征选择...")
    X_selected, selected_features = feature_selection_rfe(X_bal, y_bal, n_features=20)

    # 验证选择的特征
    print(f"   选择的特征：{list(selected_features)}")
    
    # 4. 随机森林基线模型
    print("4. 训练随机森林基线模型...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )
    
    clf_baseline = RandomForestClassifier(random_state=42)
    clf_baseline.fit(X_train, y_train)
    y_pred_baseline = clf_baseline.predict(X_test)
    y_pred_proba_baseline = clf_baseline.predict_proba(X_test)
    
    classes = np.unique(y_bal)
    evaluate_model_performance(X_test, y_test, y_pred_baseline, y_pred_proba_baseline, 
                              classes, "随机森林基线模型")
    
    # 5. PSO优化随机森林
    print("5. PSO优化随机森林...")
    space = {
        "n_estimators": (50, 1000),
        "max_depth": (3, 50),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 20)
    }
    
    pso = PSO_RandomForest(space, num_particles=20, num_iters=25)
    best_params, best_score = pso.optimize(X_train, y_train, X_test, y_test)
    
    # 用最优参数训练模型
    clf_best = RandomForestClassifier(**best_params)
    clf_best.fit(X_train, y_train)
    y_pred_best = clf_best.predict(X_test)
    y_pred_proba_best = clf_best.predict_proba(X_test)
    
    evaluate_model_performance(X_test, y_test, y_pred_best, y_pred_proba_best, 
                              classes, "PSO优化随机森林")
    
    # 6. LSTM深度学习模型（可选）
    if TENSORFLOW_AVAILABLE:
        print("6. 训练LSTM深度学习模型...")
        lstm_model, lstm_results, lstm_utils = train_lstm_model(X_selected, y_bal)
        
        if lstm_results is not None:
            X_test_lstm, y_test_lstm, y_pred_lstm, y_proba_lstm, classes_lstm = lstm_results
            evaluate_model_performance(X_test_lstm, y_test_lstm, y_pred_lstm, y_proba_lstm, 
                                      classes_lstm, "LSTM深度学习模型")
    else:
        print("6. 跳过LSTM模型训练（TensorFlow未安装）")
        lstm_model, lstm_results, lstm_utils = None, None, None
    
    print("源域故障诊断完成！")
    
    return {
        'balanced_data': (X_bal, y_bal),
        'selected_features': (X_selected, selected_features),
        'baseline_model': clf_baseline,
        'optimized_model': (clf_best, best_params),
        'lstm_model': lstm_model,
        'test_results': {
            'baseline': (X_test, y_test, y_pred_baseline, y_pred_proba_baseline),
            'optimized': (X_test, y_test, y_pred_best, y_pred_proba_best),
            'lstm': lstm_results
        }
    }


if __name__ == "__main__":
    results = main()
