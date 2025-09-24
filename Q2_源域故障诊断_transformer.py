"""
基于PyTorch Transformer的源域故障诊断脚本
使用PyTorch实现Transformer架构进行故障诊断
"""

import importlib
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from Q2_源域故障诊断 import (
    setup_environment,
    extract_features_balanced_from_long,
    plot_label_distribution,
    feature_selection_rfe,
)

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
    """PyTorch Transformer模型"""
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


class FaultDataset(Dataset):
    """故障诊断数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_transformer_model(
    X_selected: pd.DataFrame,
    y_bal: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.3,
    dense_units: int = 64,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 100,
):
    """训练PyTorch Transformer模型"""
    if not PYTORCH_AVAILABLE:
        print("PyTorch未安装，跳过Transformer训练")
        return None, None, None

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y_bal,
        test_size=test_size,
        random_state=random_state,
        stratify=y_bal,
    )

    # 标签编码
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    classes = le.classes_
    n_classes = len(classes)

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    # 重塑数据为序列格式 (batch_size, seq_len, input_dim)
    # 将特征重塑为时间序列
    seq_len = min(20, X_train_scaled.shape[1])  # 使用前20个特征作为序列长度
    input_dim = X_train_scaled.shape[1] // seq_len
    
    X_train_seq = X_train_scaled[:, :seq_len*input_dim].reshape(-1, seq_len, input_dim)
    X_test_seq = X_test_scaled[:, :seq_len*input_dim].reshape(-1, seq_len, input_dim)

    # 创建数据集和数据加载器
    train_dataset = FaultDataset(X_train_seq, y_train_enc)
    test_dataset = FaultDataset(X_test_seq, y_test_enc)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 设置设备
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
    model.train()
    best_accuracy = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
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
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {train_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # 早停
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停在epoch {epoch}")
                break
        
        model.train()

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # 预测
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    y_pred = le.inverse_transform(all_predictions)
    y_proba = np.array(all_probabilities)

    return model, (X_test, y_test, y_pred, y_proba, classes), (scaler, le)


def plot_confusion_matrix(y_true, y_pred, classes, model_name, figsize=(10, 8)):
    """绘制混淆矩阵"""
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 使用seaborn绘制热图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': '样本数量'})
    
    plt.title(f'{model_name} 混淆矩阵', fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    filename = f'{model_name}_confusion_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存为: {filename}")
    
    # 显示图片
    plt.show()
    
    return cm


def evaluate_model_performance_simple(X_test, y_test, y_pred, y_proba, classes, model_name):
    """简单的模型性能评估"""
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} 性能评估:")
    print(f"准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_test, y_pred, classes, model_name)


def main():
    """主程序入口"""
    print("开始基于PyTorch Transformer的源域故障诊断...")

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
    # 根据实际数据分布，OR类有77个样本，是最多的类别
    # 将其他类别也平衡到77个样本，保持数据集平衡
    X_bal, y_bal = extract_features_balanced_from_long(
        long_table,
        signal_col="DE_time",
        fs=32000,
        target_per_class=77,  # 基于最大类别数量设置
        mix_ratio=0.5,
        random_state=42,
    )

    print(f"   均衡后特征矩阵形状：{X_bal.shape}")
    print(f"   特征列：{list(X_bal.columns)}")

    plot_label_distribution(y_bal, "均衡后标签分布")

    print("3. 特征选择...")
    X_selected, selected_features = feature_selection_rfe(X_bal, y_bal, n_features=20)
    print(f"   选择的特征：{list(selected_features)}")

    if PYTORCH_AVAILABLE:
        print("4. 训练Transformer深度学习模型...")
        transformer_model, transformer_results, transformer_utils = train_transformer_model(X_selected, y_bal)
        if transformer_results is not None:
            X_test_tf, y_test_tf, y_pred_tf, y_proba_tf, classes_tf = transformer_results
            evaluate_model_performance_simple(
                X_test_tf,
                y_test_tf,
                y_pred_tf,
                y_proba_tf,
                classes_tf,
                "PyTorch Transformer模型",
            )
    else:
        print("4. 跳过Transformer模型训练（PyTorch未安装）")
        transformer_model, transformer_results, transformer_utils = None, None, None

    print("基于PyTorch Transformer的源域故障诊断完成！")

    return {
        "balanced_data": (X_bal, y_bal),
        "selected_features": (X_selected, selected_features),
        "transformer_model": transformer_model,
        "test_results": {
            "transformer": transformer_results,
        },
    }


if __name__ == "__main__":
    results = main()
