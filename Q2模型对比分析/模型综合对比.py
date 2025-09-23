# -*- coding: utf-8 -*-
"""
模型综合对比分析 - 轴承故障诊断
整合所有模型的性能对比和可视化分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

# 导入所有模型
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost未安装，将跳过XGBoost模型")

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
            
            # 处理缺失值
            print(f"原始数据形状: {X.shape}")
            print(f"缺失值数量: {X.isnull().sum().sum()}")
            
            if X.isnull().sum().sum() > 0:
                print("发现缺失值，进行处理...")
                # 用均值填充数值型特征的缺失值
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
            verbosity=0  # 减少输出，优化性能
        )
    
    return models

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """评估所有模型"""
    results = {}
    
    # 为XGBoost准备标签编码
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    print("开始训练和评估所有模型...")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"正在训练 {name} 模型...")
        
        # 记录训练时间
        start_time = time.time()
        
        # 根据模型类型选择合适的标签格式
        if name == 'XGBoost':
            # XGBoost使用数值标签
            model.fit(X_train, y_train_encoded)
            y_pred_encoded = model.predict(X_test)
            y_pred = le.inverse_transform(y_pred_encoded)
        else:
            # 其他模型使用原始标签
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        training_time = time.time() - start_time
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 交叉验证分数
        if name == 'XGBoost':
            cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=5, scoring='accuracy')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'training_time': training_time,
            'model': model,
            'y_pred': y_pred
        }
        
        print(f"{name} - 准确率: {accuracy:.4f}, F1分数: {f1:.4f}, 训练时间: {training_time:.2f}s")
    
    print("=" * 60)
    return results

def plot_performance_comparison(results):
    """绘制性能对比图"""
    # 准备数据
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数']
    
    # 创建性能对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = [results[name][metric] for name in model_names]
        
        bars = axes[i].bar(model_names, values, alpha=0.8, 
                          color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(model_names)])
        axes[i].set_title(f'{metric_name}对比', fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric_name)
        axes[i].set_ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        axes[i].grid(axis='y', alpha=0.3)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('模型性能对比.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_time_comparison(results):
    """绘制训练时间对比"""
    model_names = list(results.keys())
    training_times = [results[name]['training_time'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, training_times, alpha=0.8, 
                   color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(model_names)])
    
    plt.title('模型训练时间对比', fontsize=14, fontweight='bold')
    plt.ylabel('训练时间 (秒)')
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for bar, time_val in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('模型训练时间对比.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_cv_scores_comparison(results):
    """绘制交叉验证分数对比"""
    model_names = list(results.keys())
    cv_means = [results[name]['cv_mean'] for name in model_names]
    cv_stds = [results[name]['cv_std'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, cv_means, yerr=cv_stds, alpha=0.8, capsize=5,
                   color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(model_names)])
    
    plt.title('模型交叉验证分数对比', fontsize=14, fontweight='bold')
    plt.ylabel('交叉验证准确率')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # 添加数值标签
    for bar, mean_val, std_val in zip(bars, cv_means, cv_stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.01,
                f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('模型交叉验证对比.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(results, y_test):
    """绘制所有模型的混淆矩阵对比"""
    n_models = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    classes = np.unique(y_test)
    
    for i, (name, result) in enumerate(results.items()):
        if i >= len(axes):
            break
            
        cm = confusion_matrix(y_test, result['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=classes, yticklabels=classes)
        axes[i].set_title(f'{name}模型混淆矩阵', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('预测标签')
        axes[i].set_ylabel('真实标签')
    
    # 隐藏多余的子图
    for i in range(len(results), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('所有模型混淆矩阵对比.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(results):
    """创建性能汇总表"""
    summary_data = []
    
    for name, result in results.items():
        summary_data.append({
            '模型': name,
            '准确率': f"{result['accuracy']:.4f}",
            '精确率': f"{result['precision']:.4f}",
            '召回率': f"{result['recall']:.4f}",
            'F1分数': f"{result['f1']:.4f}",
            '交叉验证': f"{result['cv_mean']:.4f}±{result['cv_std']:.4f}",
            '训练时间(s)': f"{result['training_time']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # 保存到CSV
    summary_df.to_csv('模型性能汇总表.csv', index=False, encoding='utf-8-sig')
    
    # 显示表格
    print("\n模型性能汇总表:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    
    return summary_df

def plot_radar_chart(results):
    """绘制雷达图对比"""
    try:
        from math import pi
        
        # 准备数据
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['准确率', '精确率', '召回率', 'F1分数']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 设置角度
        angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
        angles += angles[:1]  # 闭合
        
        colors = ['b', 'r', 'g', 'orange', 'purple']
        
        for i, (name, result) in enumerate(results.items()):
            values = [result[metric] for metric in metrics]
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.set_title('模型性能雷达图对比', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('模型性能雷达图.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"绘制雷达图时出错: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("模型综合对比分析 - 轴承故障诊断")
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
    print(f"将对比以下 {len(models)} 个模型: {list(models.keys())}")
    
    # 评估所有模型
    results = evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # 生成各种对比图表
    print("\n生成对比图表...")
    
    # 1. 性能对比图
    plot_performance_comparison(results)
    
    # 2. 训练时间对比
    plot_training_time_comparison(results)
    
    # 3. 交叉验证对比
    plot_cv_scores_comparison(results)
    
    # 4. 混淆矩阵对比
    plot_confusion_matrices(results, y_test)
    
    # 5. 雷达图对比
    plot_radar_chart(results)
    
    # 6. 创建汇总表
    summary_df = create_summary_table(results)
    
    # 找出最佳模型
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
    print(f"\n最佳模型（基于F1分数）: {best_model_name}")
    print(f"F1分数: {results[best_model_name]['f1']:.4f}")
    
    print("\n模型综合对比分析完成！")
    print("所有图表和结果已保存到当前目录。")

if __name__ == "__main__":
    main()
