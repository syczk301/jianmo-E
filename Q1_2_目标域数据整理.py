"""
目标域数据整理脚本
功能：读取目标域.mat文件，整理为统一格式并保存为CSV
作者：华为杯研赛E题代码助攻
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path


def explore_target_files(current_dir=None):
    """探索目标域文件结构"""
    if current_dir is None:
        current_dir = Path.cwd()
    
    print("=== 探索目标域文件结构 ===")
    records = []

    for file in current_dir.rglob("*.mat"):
        rel_path = file.relative_to(current_dir)

        # 只处理目标域数据集
        if "源域数据集" in str(rel_path):
            continue

        try:
            mat_data = sio.loadmat(file)
            variables = [k for k in mat_data.keys() if not k.startswith("__")]

            for var in variables:
                data = mat_data[var]
                if isinstance(data, np.ndarray):
                    rows = data.shape[0]
                    cols = data.shape[1] if data.ndim > 1 else 1
                    records.append([str(rel_path), var, rows, cols])
                else:
                    records.append([str(rel_path), var, None, None])
        except Exception as e:
            records.append([str(rel_path), "读取失败", None, None])

    # 转换为 DataFrame
    df = pd.DataFrame(records, columns=["file", "variable", "samples", "columns"])
    return df


def pick_signal_var(mat_dict, stem_upper):
    """从 loadmat 的字典中挑选信号变量名"""
    keys = [k for k in mat_dict.keys() if not k.startswith("__")]
    if not keys:
        return None
    # 1) 优先文件名同名变量（不区分大小写）
    for k in keys:
        if k.upper() == stem_upper:
            return k
    # 2) 其次第一个 ndarray 变量
    for k in keys:
        if isinstance(mat_dict[k], np.ndarray):
            return k
    # 3) 实在不行就返回第一个
    return keys[0]


def ensure_2d_col(a):
    """将数组转换为二维列向量 (n,1)；若为标量则 (1,1)"""
    arr = np.array(a)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        # 若是 (1,n) 也转为 (n,1)
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            return arr.reshape(-1, 1)
        return arr
    return arr.reshape(-1, 1)


def create_target_wide_dataframe(target_dir="目标域数据集"):
    """创建目标域宽格式DataFrame"""
    TARGET_DIR = Path(target_dir)
    records = []

    for f in sorted(TARGET_DIR.glob("*.mat")):
        try:
            mat = sio.loadmat(f)
        except Exception as e:
            print(f"[WARN] 读取失败：{f} | {e}")
            continue

        var_name = pick_signal_var(mat, f.stem.upper())
        if var_name is None:
            print(f"[WARN] 未找到变量：{f}")
            continue

        x = ensure_2d_col(mat[var_name])  # 统一为 (n,1)
        rpm = np.array([[600.0]])         # 统一 RPM（shape 1x1）

        records.append({
            "file": str(f.resolve()),
            "status": None,               # 未知
            "fault_size_inch": np.nan,    # 未知
            "position": None,             # 未知
            "load": np.nan,               # 未知
            "Xtime": x,
            "RPM": rpm
        })

    tgt_df = pd.DataFrame(records, columns=[
        "file", "status", "fault_size_inch", "position", "load", "Xtime", "RPM"
    ])

    return tgt_df


def load_vector(file_path):
    """读取 .mat 文件，返回一维 np.array"""
    mat = sio.loadmat(file_path)
    # 提取变量名：优先与文件名一致，否则取第一个
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if not keys:
        return None
    var_name = next((k for k in keys if k.upper() == file_path.stem.upper()), keys[0])
    arr = np.array(mat[var_name]).reshape(-1)  # 直接扁平化
    return arr


def create_target_long_dataframe(target_dir="目标域数据集"):
    """创建目标域长格式DataFrame"""
    TARGET_DIR = Path(target_dir)
    frames = []
    
    for f in sorted(TARGET_DIR.glob("*.mat")):
        try:
            x = load_vector(f)
        except Exception as e:
            print(f"[WARN] 读取失败：{f} | {e}")
            continue
        if x is None:
            continue

        n = len(x)
        print(f"处理文件 {f.name}，数据长度：{n}")

        df = pd.DataFrame({
            "file": str(f.resolve()),
            "status": None,
            "fault_size_inch": np.nan,
            "position": None,
            "load": np.nan,
            "Xtime": x,
            "RPM": 600.0
        })

        frames.append(df)

    # 合并
    tgt_long = pd.concat(frames, ignore_index=True)
    return tgt_long


def create_target_summary(target_dir="目标域数据集"):
    """创建目标域轻量级摘要表"""
    TARGET_DIR = Path(target_dir)
    summary_rows = []
    
    for f in sorted(TARGET_DIR.glob("*.mat")):
        try:
            x = load_vector(f)
        except Exception as e:
            print(f"[WARN] 读取失败：{f} | {e}")
            continue
        if x is None:
            continue

        # 计算基本统计信息
        summary_info = {
            "file": str(f.resolve()),
            "filename": f.name,
            "status": None,
            "fault_size_inch": np.nan,
            "position": None,
            "load": np.nan,
            "RPM": 600.0,
            "signal_length": len(x),
            "signal_mean": np.mean(x),
            "signal_std": np.std(x),
            "signal_min": np.min(x),
            "signal_max": np.max(x),
            "signal_rms": np.sqrt(np.mean(x**2))
        }
        summary_rows.append(summary_info)
    
    return pd.DataFrame(summary_rows)


def main():
    """主程序入口"""
    print("开始目标域数据整理...")
    
    # 1. 探索文件结构
    print("1. 探索文件结构...")
    df_explore = explore_target_files()
    print(f"   发现 {len(df_explore)} 个变量")
    
    # 2. 创建宽格式DataFrame
    print("2. 创建宽格式DataFrame...")
    tgt_wide_df = create_target_wide_dataframe()
    print(f"   宽格式数据维度：{tgt_wide_df.shape}")
    
    # 3. 创建轻量级摘要表（推荐）
    print("3. 创建数据摘要表...")
    summary_df = create_target_summary()
    summary_df.to_csv('目标域数据摘要.csv', index=None)
    print(f"   摘要表已保存，维度：{summary_df.shape}")
    
    # 4. 可选：创建长格式DataFrame（内存密集）
    create_long = input("是否创建长格式数据表？（内存密集，输入y确认）: ").lower() == 'y'
    if create_long:
        print("4. 创建长格式DataFrame（可能需要较长时间）...")
        tgt_long_df = create_target_long_dataframe()
        tgt_long_df.to_csv('目标域数据整理.csv', index=None)
        print(f"   长格式数据已保存，维度：{tgt_long_df.shape}")
        return df_explore, tgt_wide_df, summary_df, tgt_long_df
    else:
        print("4. 跳过长格式数据表创建")
        return df_explore, tgt_wide_df, summary_df, None
    
    print("目标域数据整理完成！")


if __name__ == "__main__":
    results = main()
