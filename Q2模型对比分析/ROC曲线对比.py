# -*- coding: utf-8 -*-
"""
所有模型ROC曲线对比 - 轴承故障诊断
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost未安装，将跳过XGBoost模型")

import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据"""
    import os
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
    
    print(f"数据文件未找到，尝试了以下路径: {possible_paths}")
    return None, None

def get_models():
    """获取所有要对比的模型"""
    models = {
        '决策树': DecisionTreeClassifier(
            max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42
        ),
        '随机森林': RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5, 
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100, learning_rate=1.0, random_state=42
        ),
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, 
            random_state=42, n_jobs=2, eval_metric='mlogloss',
            verbosity=0
        )
    
    return models

def plot_individual_roc_curves(models, X_test, y_test, y_test_bin, model_predictions):
    """为每个模型单独绘制ROC曲线"""
    classes = ['B', 'IR', 'N', 'OR']
    n_classes = len(classes)
    
    # 为每个模型创建单独的ROC图
    for model_name, y_pred_proba in model_predictions.items():
        plt.figure(figsize=(10, 8))
        
        # 为每个类别绘制ROC曲线
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2,
                    label=f'{classes[i]} (AUC = {roc_auc:.3f})')
        
        # 绘制随机分类器线
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='随机分类器 (AUC = 0.50)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (FPR)', fontsize=12)
        plt.ylabel('真阳性率 (TPR)', fontsize=12)
        plt.title(f'{model_name}模型 - 多分类ROC曲线', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(f'{model_name}_ROC曲线.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ {model_name}模型ROC曲线已保存")

def plot_combined_roc_curves(models, X_test, y_test, y_test_bin, model_predictions):
    """绘制所有模型的ROC曲线对比图（按类别分组）"""
    classes = ['B', 'IR', 'N', 'OR']
    n_classes = len(classes)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # 为每个类别创建一个子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i in range(n_classes):
        ax = axes[i]
        
        # 为每个模型在当前类别上绘制ROC曲线
        for j, (model_name, y_pred_proba) in enumerate(model_predictions.items()):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[j % len(colors)], linewidth=2,
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # 绘制随机分类器线
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8, label='随机分类器')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('假阳性率 (FPR)')
        ax.set_ylabel('真阳性率 (TPR)')
        ax.set_title(f'类别 {classes[i]} 的ROC曲线对比', fontweight='bold')
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('所有模型ROC曲线对比 - 按类别分组', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('所有模型ROC曲线对比_按类别.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ 所有模型ROC曲线对比图（按类别分组）已保存")

def plot_average_roc_curves(models, X_test, y_test, y_test_bin, model_predictions):
    """绘制所有模型的平均ROC曲线对比"""
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    plt.figure(figsize=(10, 8))
    
    for i, (model_name, y_pred_proba) in enumerate(model_predictions.items()):
        # 计算每个类别的ROC曲线
        fpr_list = []
        tpr_list = []
        auc_list = []
        
        for j in range(y_test_bin.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_bin[:, j], y_pred_proba[:, j])
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(auc(fpr, tpr))
        
        # 计算平均AUC
        mean_auc = np.mean(auc_list)
        
        # 使用插值计算平均ROC曲线
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)
        
        for fpr, tpr in zip(fpr_list, tpr_list):
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
        
        mean_tpr /= len(fpr_list)
        mean_tpr[0] = 0.0  # 确保起点为(0,0)
        
        plt.plot(mean_fpr, mean_tpr, color=colors[i % len(colors)], linewidth=3,
                label=f'{model_name} (平均AUC = {mean_auc:.3f})')
    
    # 绘制随机分类器线
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.8, label='随机分类器 (AUC = 0.50)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR)', fontsize=12)
    plt.title('所有模型平均ROC曲线对比', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('所有模型平均ROC曲线对比.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ 所有模型平均ROC曲线对比图已保存")

def main():
    """主函数"""
    print("=" * 60)
    print("所有模型ROC曲线对比分析")
    print("=" * 60)
    
    # 加载数据
    print("加载数据...")
    X, y = load_data()
    if X is None or y is None:
        return
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"类别分布:\n{y.value_counts()}")
    
    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 获取模型
    models = get_models()
    print(f"将分析以下 {len(models)} 个模型: {list(models.keys())}")
    
    # 准备标签编码
    classes = ['B', 'IR', 'N', 'OR']
    y_test_bin = label_binarize(y_test, classes=classes)
    
    # 为XGBoost准备数值标签
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # 训练所有模型并获取预测概率
    model_predictions = {}
    
    print("\n开始训练所有模型...")
    for name, model in models.items():
        print(f"正在训练 {name} 模型...")
        
        try:
            if name == 'XGBoost':
                # XGBoost使用数值标签
                model.fit(X_train, y_train_encoded)
                y_pred_proba = model.predict_proba(X_test)
            else:
                # 其他模型使用原始标签
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)
            
            model_predictions[name] = y_pred_proba
            print(f"✓ {name} 训练完成")
            
        except Exception as e:
            print(f"✗ {name} 训练失败: {e}")
    
    print(f"\n成功训练 {len(model_predictions)} 个模型")
    
    # 绘制ROC曲线
    print("\n开始绘制ROC曲线...")
    
    # 1. 为每个模型单独绘制ROC曲线
    plot_individual_roc_curves(models, X_test, y_test, y_test_bin, model_predictions)
    
    # 2. 绘制按类别分组的对比图
    plot_combined_roc_curves(models, X_test, y_test, y_test_bin, model_predictions)
    
    # 3. 绘制平均ROC曲线对比
    plot_average_roc_curves(models, X_test, y_test, y_test_bin, model_predictions)
    
    print("\n🎉 所有ROC曲线绘制完成！")
    print("生成的图片文件:")
    print("- 各模型单独ROC曲线: [模型名]_ROC曲线.png")
    print("- 按类别分组对比: 所有模型ROC曲线对比_按类别.png")
    print("- 平均ROC曲线对比: 所有模型平均ROC曲线对比.png")

if __name__ == "__main__":
    main()
