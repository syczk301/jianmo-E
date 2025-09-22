"""
基于CNN的源域故障诊断脚本
在Q2的流程基础上，将深度学习模型由LSTM改为一维卷积神经网络
"""

import importlib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from Q2_源域故障诊断 import (
    setup_environment,
    extract_features_balanced_from_long,
    plot_label_distribution,
    feature_selection_rfe,
    evaluate_model_performance,
    PSO_RandomForest,
)

_tf_spec = importlib.util.find_spec("tensorflow")
if _tf_spec is not None:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv1D,
        Dense,
        Dropout,
        GlobalAveragePooling1D,
        MaxPooling1D,
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical

    TENSORFLOW_AVAILABLE = True
else:
    tf = None
    EarlyStopping = None
    BatchNormalization = None
    Conv1D = None
    Dense = None
    Dropout = None
    GlobalAveragePooling1D = None
    MaxPooling1D = None
    Sequential = None
    Adam = None
    to_categorical = None

    TENSORFLOW_AVAILABLE = False


def build_cnn_model(
    input_timesteps: int,
    n_classes: int,
    filters=(64, 128),
    kernel_size: int = 3,
    dropout: float = 0.3,
    dense_units: int = 64,
    lr: float = 1e-3,
):
    """构建一维卷积神经网络模型"""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow未安装，无法使用CNN模型")

    conv_filters = filters if isinstance(filters, (list, tuple)) else (filters,)
    model_layers = []

    for idx, filt in enumerate(conv_filters):
        conv_kwargs = {
            "filters": int(filt),
            "kernel_size": int(kernel_size),
            "activation": "relu",
            "padding": "same",
        }
        if idx == 0:
            conv_kwargs["input_shape"] = (input_timesteps, 1)
        model_layers.append(Conv1D(**conv_kwargs))
        model_layers.append(BatchNormalization())
        model_layers.append(MaxPooling1D(pool_size=2))
        if dropout > 0:
            model_layers.append(Dropout(dropout))

    model_layers.append(GlobalAveragePooling1D())
    model_layers.append(Dense(int(dense_units), activation="relu"))
    if dropout > 0:
        model_layers.append(Dropout(dropout))
    model_layers.append(Dense(n_classes, activation="softmax"))

    model = Sequential(model_layers)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn_model(
    X_selected: pd.DataFrame,
    y_bal: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
    filters=(64, 128),
    kernel_size: int = 3,
    dropout: float = 0.3,
    dense_units: int = 64,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 100,
):
    """训练CNN模型"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow未安装，跳过CNN训练")
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y_bal,
        test_size=test_size,
        random_state=random_state,
        stratify=y_bal,
    )

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    classes = le.classes_
    n_classes = len(classes)

    y_train_oh = to_categorical(y_train_enc, num_classes=n_classes)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    timesteps = X_train_scaled.shape[1]
    X_train_cnn = X_train_scaled.reshape(-1, timesteps, 1)
    X_test_cnn = X_test_scaled.reshape(-1, timesteps, 1)

    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    model = build_cnn_model(
        timesteps,
        n_classes,
        filters=filters,
        kernel_size=kernel_size,
        dropout=dropout,
        dense_units=dense_units,
        lr=lr,
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
    )

    model.fit(
        X_train_cnn,
        y_train_oh,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    y_proba = model.predict(X_test_cnn)
    y_pred_enc = np.argmax(y_proba, axis=1)
    y_pred = le.inverse_transform(y_pred_enc)

    return model, (X_test, y_test, y_pred, y_proba, classes), (scaler, le)


def main():
    """主程序入口"""
    print("开始基于CNN的源域故障诊断...")

    setup_environment()

    print("1. 加载源域数据...")
    try:
        long_table = pd.read_csv("源域数据32khz整理.csv")
        print(f"   数据加载成功，维度：{long_table.shape}")
    except FileNotFoundError:
        print("   错误：未找到'源域数据32khz整理.csv'文件")
        print("   请先运行Q1_1_源域数据整合.py生成数据文件")
        return None

    print("2. 数据平衡和特征提取...")
    X_bal, y_bal = extract_features_balanced_from_long(
        long_table,
        signal_col="DE_time",
        fs=32000,
        target_per_class=77,
        mix_ratio=0.5,
        random_state=42,
    )

    print(f"   均衡后特征矩阵形状：{X_bal.shape}")
    print(f"   特征列：{list(X_bal.columns)}")

    plot_label_distribution(y_bal, "均衡后标签分布")

    print("3. 特征选择...")
    X_selected, selected_features = feature_selection_rfe(X_bal, y_bal, n_features=20)
    print(f"   选择的特征：{list(selected_features)}")

    print("4. 训练随机森林基线模型...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y_bal,
        test_size=0.3,
        random_state=42,
        stratify=y_bal,
    )

    clf_baseline = RandomForestClassifier(random_state=42)
    clf_baseline.fit(X_train, y_train)
    y_pred_baseline = clf_baseline.predict(X_test)
    y_pred_proba_baseline = clf_baseline.predict_proba(X_test)

    classes = np.unique(y_bal)
    evaluate_model_performance(
        X_test,
        y_test,
        y_pred_baseline,
        y_pred_proba_baseline,
        classes,
        "随机森林基线模型",
    )

    print("5. PSO优化随机森林...")
    space = {
        "n_estimators": (50, 1000),
        "max_depth": (3, 50),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 20),
    }

    pso = PSO_RandomForest(space, num_particles=20, num_iters=25)
    best_params, best_score = pso.optimize(X_train, y_train, X_test, y_test)

    clf_best = RandomForestClassifier(**best_params)
    clf_best.fit(X_train, y_train)
    y_pred_best = clf_best.predict(X_test)
    y_pred_proba_best = clf_best.predict_proba(X_test)

    evaluate_model_performance(
        X_test,
        y_test,
        y_pred_best,
        y_pred_proba_best,
        classes,
        "PSO优化随机森林",
    )

    if TENSORFLOW_AVAILABLE:
        print("6. 训练CNN深度学习模型...")
        cnn_model, cnn_results, cnn_utils = train_cnn_model(X_selected, y_bal)
        if cnn_results is not None:
            X_test_cnn, y_test_cnn, y_pred_cnn, y_proba_cnn, classes_cnn = cnn_results
            evaluate_model_performance(
                X_test_cnn,
                y_test_cnn,
                y_pred_cnn,
                y_proba_cnn,
                classes_cnn,
                "CNN深度学习模型",
            )
    else:
        print("6. 跳过CNN模型训练（TensorFlow未安装）")
        cnn_model, cnn_results, cnn_utils = None, None, None

    print("基于CNN的源域故障诊断完成！")

    return {
        "balanced_data": (X_bal, y_bal),
        "selected_features": (X_selected, selected_features),
        "baseline_model": clf_baseline,
        "optimized_model": (clf_best, best_params),
        "cnn_model": cnn_model,
        "test_results": {
            "baseline": (X_test, y_test, y_pred_baseline, y_pred_proba_baseline),
            "optimized": (X_test, y_test, y_pred_best, y_pred_proba_best),
            "cnn": cnn_results,
        },
    }


if __name__ == "__main__":
    results = main()
