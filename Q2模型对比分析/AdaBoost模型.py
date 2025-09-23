# -*- coding: utf-8 -*-
"""
AdaBoost模型 - 轴承故障诊断
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据"""
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

def train_adaboost(X_train, X_test, y_train, y_test):
    """训练AdaBoost模型"""
    
    # 参数网格搜索
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.5, 1.0, 1.5],
        'algorithm': ['SAMME', 'SAMME.R'],
        'base_estimator__max_depth': [1, 2, 3, 4, 5],
        'base_estimator__min_samples_split': [2, 5, 10],
        'base_estimator__min_samples_leaf': [1, 2, 4]
    }
    
    # 创建AdaBoost分类器（使用决策树作为基学习器）
    base_estimator = DecisionTreeClassifier(random_state=42)
    ada = AdaBoostClassifier(
        base_estimator=base_estimator,
        random_state=42
    )
    
    # 网格搜索
    print("开始网格搜索最优参数...")
    grid_search = GridSearchCV(
        ada, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"最优参数: {grid_search.best_params_}")
    print(f"最优交叉验证分数: {grid_search.best_score_:.4f}")
    
    # 使用最优参数训练模型
    best_ada = grid_search.best_estimator_
    
    # 预测
    y_pred = best_ada.predict(X_test)
    y_pred_proba = best_ada.predict_proba(X_test)
    
    return best_ada, y_pred, y_pred_proba

def evaluate_model(y_test, y_pred, y_pred_proba, model_name="AdaBoost"):
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

def plot_feature_importance(model, feature_names, model_name="AdaBoost"):
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

def plot_boosting_error(model, X_test, y_test, model_name="AdaBoost"):
    """绘制AdaBoost训练过程中的错误率变化"""
    if hasattr(model, 'staged_predict'):
        # 计算每个阶段的错误率
        staged_predictions = list(model.staged_predict(X_test))
        errors = [1 - accuracy_score(y_test, pred) for pred in staged_predictions]
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(errors) + 1), errors, 'b-', label='测试错误率')
        plt.xlabel('提升轮数')
        plt.ylabel('错误率')
        plt.title(f'{model_name}模型错误率变化')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{model_name}_boosting_error.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_estimator_weights(model, model_name="AdaBoost"):
    """绘制基学习器权重分布"""
    if hasattr(model, 'estimator_weights_'):
        weights = model.estimator_weights_
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(weights) + 1), weights, 'ro-')
        plt.xlabel('基学习器索引')
        plt.ylabel('权重')
        plt.title(f'{model_name}模型基学习器权重分布')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{model_name}_estimator_weights.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n{model_name}基学习器权重统计:")
        print(f"平均权重: {np.mean(weights):.4f}")
        print(f"权重标准差: {np.std(weights):.4f}")
        print(f"最大权重: {np.max(weights):.4f}")
        print(f"最小权重: {np.min(weights):.4f}")

def main():
    """主函数"""
    print("=" * 50)
    print("AdaBoost模型 - 轴承故障诊断")
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
    model, y_pred, y_pred_proba = train_adaboost(X_train, X_test, y_train, y_test)
    
    # 评估模型
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, "AdaBoost")
    
    # 特征重要性
    plot_feature_importance(model, X.columns, "AdaBoost")
    
    # AdaBoost特有的可视化
    plot_boosting_error(model, X_test, y_test, "AdaBoost")
    plot_estimator_weights(model, "AdaBoost")
    
    # 保存模型性能结果
    results_df = pd.DataFrame([metrics])
    results_df.to_csv('AdaBoost_performance_results.csv', index=False)
    print("\n模型性能结果已保存到 'AdaBoost_performance_results.csv'")
    
    print("\nAdaBoost模型分析完成！")

if __name__ == "__main__":
    main()
