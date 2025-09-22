"""
源域数据整合脚本
功能：读取.mat文件，解析轴承故障数据，提取特征并保存为CSV格式
作者：华为杯研赛E题代码助攻
"""

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import resample, hilbert, welch, stft
from scipy.stats import kurtosis, skew
import pywt


def explore_mat_files(current_dir=None):
    """探索当前目录下所有.mat文件的结构"""
    if current_dir is None:
        current_dir = Path.cwd()
    
    print("=== 探索.mat文件结构 ===")
    for file in current_dir.rglob("*.mat"):
        rel_path = file.relative_to(current_dir)
        print(f"\n文件: {rel_path}")

        try:
            mat_data = sio.loadmat(file)
            # 筛掉 Matlab 内置的 __ 开头变量
            variables = [k for k in mat_data.keys() if not k.startswith("__")]

            for var in variables:
                data = mat_data[var]
                if isinstance(data, np.ndarray):
                    rows, cols = data.shape[0], data.shape[1] if data.ndim > 1 else 1
                    print(f"  变量: {var}, 样本数: {rows}, 列数: {cols}")
                else:
                    print(f"  变量: {var}, 类型: {type(data)} (非 ndarray)")
        except Exception as e:
            print(f"   读取失败: {e}")


def create_file_summary(current_dir=None):
    """创建文件摘要DataFrame"""
    if current_dir is None:
        current_dir = Path.cwd()
    
    records = []
    
    for file in current_dir.rglob("*.mat"):
        rel_path = file.relative_to(current_dir)

        # 跳过"目标域数据集"目录下的文件
        if "目标域数据集" in str(rel_path):
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
    
    # 拆分 variable，提取数据编号和特征类型
    df['data_id'] = df['variable'].str.extract(r'(X\d+)')   # 数据编号，如 X118
    df['feature'] = df['variable'].str.replace(r'X\d+', '', regex=True)  # 剔除编号，保留特征
    
    return df


def parse_status_size_load_position_from_path(path_str: str):
    """
    从文件路径/文件名中解析：
      - status: B / IR / OR / N
      - fault_size_inch: 0.007 等（英寸）；正常样本返回 0 或 None
      - position: 仅 OR 有，3->Orthogonal, 6->Centered, 12->Opposite
      - load: 0/1/2/3
    """
    p = Path(path_str)
    name = p.stem  # 文件名(无扩展)
    up = name.upper()

    parts_upper = [s.upper() for s in p.parts]

    # 1) status
    status = None
    # 先看文件名前缀
    if up.startswith("IR"):
        status = "IR"
    elif up.startswith("OR"):
        status = "OR"
    elif up.startswith("B"):
        status = "B"
    elif up.startswith("N") or "NORMAL" in up:
        status = "N"
    # 再从目录兜底
    if status is None:
        if any(seg in {"IR", "INNER", "INNER_RACE"} for seg in parts_upper):
            status = "IR"
        elif any(seg in {"OR", "OUTER", "OUTER_RACE"} for seg in parts_upper):
            status = "OR"
        elif any(seg in {"B", "BALL"} for seg in parts_upper):
            status = "B"
        elif any("NORMAL" in seg or seg == "N" for seg in parts_upper):
            status = "N"

    # 2) fault size（仅 B/IR/OR 有）
    fault_size_inch = None
    m_size = re.search(r'(?:B|IR|OR)(\d{3})', up)
    if m_size:
        fault_size_inch = int(m_size.group(1)) / 1000.0
    elif status == "N":
        fault_size_inch = 0.0

    # 3) position（仅 OR 有且可能包含 3/6/12）
    position = None
    if status == "OR":
        m_pos = re.search(r'OR\d{3}[@_\-]?(\d{1,2})', up)
        pos_val = None
        if m_pos:
            pos_val = m_pos.group(1)
        else:
            # 目录兜底：若路径中包含 3/6/12 且离 OR 最近的数字
            candidates = {seg for seg in parts_upper if seg in {"3", "6", "12"}}
            if "12" in candidates:
                pos_val = "12"
            elif "6" in candidates:
                pos_val = "6"
            elif "3" in candidates:
                pos_val = "3"

        if pos_val in {"3", "6", "12"}:
            if pos_val == "3":
                position = "Orthogonal"  # 3 点钟
            elif pos_val == "6":
                position = "Centered"    # 6 点钟
            elif pos_val == "12":
                position = "Opposite"    # 12 点钟

    # 4) load（优先文件名下划线后的尾数；兜底从目录判断）
    load = None
    m_load = re.search(r'_(\d)(?:\.MAT)?$', up)
    if m_load:
        load = int(m_load.group(1))
    else:
        # 目录兜底：常见路径含 0/1/2/3 目录段
        for seg in reversed(parts_upper):
            if seg in {"0", "1", "2", "3"}:
                load = int(seg)
                break

    return status, fault_size_inch, position, load


def infer_rpm_from_load(load: int, fs_hint: str = "") -> int:
    """
    根据 12kHz CWRU 常见映射推断 RPM。
    载荷 0/1/2/3 -> 1797 / 1772 / 1750 / 1730
    """
    rpm_map = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
    if load in rpm_map:
        return rpm_map[load]
    return None


def extract_data_id(var_names):
    """
    从变量名列表中提取数据编号（如 'X118'）。
    """
    for k in var_names:
        m = re.search(r'(X\d+)', k)
        if m:
            return m.group(1)
    return None


def load_mat_signals_to_features(mat_path: str):
    """
    读取 .mat 文件并返回统一特征 dict：
      {'DE_time': ndarray or None, 'FE_time': ndarray or None,
       'BA_time': ndarray or None, 'RPM': ndarray or None}, 以及 data_id
    """
    mat = sio.loadmat(mat_path)
    keys = [k for k in mat.keys() if not k.startswith("__")]

    features = {"DE_time": None, "FE_time": None, "BA_time": None, "RPM": None}
    data_id = extract_data_id(keys)

    for k in keys:
        upk = k.upper()
        # 匹配 *_DE_time
        if re.search(r'_DE_TIME$', upk):
            features["DE_time"] = mat[k]
        elif re.search(r'_FE_TIME$', upk):
            features["FE_time"] = mat[k]
        elif re.search(r'_BA_TIME$', upk):
            features["BA_time"] = mat[k]
        elif re.search(r'RPM$', upk):
            # RPM 统一为 shape (1,1)
            v = mat[k]
            v = np.array(v)
            if v.size == 1:
                v = v.reshape(1, 1)
            features["RPM"] = v

    return features, data_id


def create_wide_dataframe(df):
    """创建宽格式DataFrame，每个文件一行"""
    rows = []
    unique_files = df["file"].unique()

    for fpath in unique_files:
        # 解析元数据
        status, fault_size_inch, position, load = parse_status_size_load_position_from_path(fpath)

        # 读取特征
        try:
            feats, data_id = load_mat_signals_to_features(fpath)
        except Exception as e:
            # 若文件读取失败，构造空行以便后续排查
            feats = {"DE_time": None, "FE_time": None, "BA_time": None, "RPM": None}
            data_id = None
            print(f"[WARN] 读取失败: {fpath} | {e}")

        # 若 RPM 缺失且能根据载荷推断，则补齐（shape 1x1）
        if feats["RPM"] is None and load is not None:
            rpm_val = infer_rpm_from_load(load)
            if rpm_val is not None:
                feats["RPM"] = np.array([[rpm_val]])

        # 组装一行
        row = {
            "file": fpath,
            "status": status,
            "fault_size_inch": fault_size_inch,
            "position": position,   # OR 才有；其他状态为 None
            "load": load,
            "data_id": data_id,
            "DE_time": feats["DE_time"],
            "FE_time": feats["FE_time"],
            "BA_time": feats["BA_time"],
            "RPM": feats["RPM"],
        }
        rows.append(row)

    # 合并为大宽表
    wide_df = pd.DataFrame(rows, columns=[
        "file", "status", "fault_size_inch", "position", "load", "data_id",
        "DE_time", "FE_time", "BA_time", "RPM"
    ])

    return wide_df


def resample_to_8s_32k(x, fs_in, target_fs=32000, target_dur=8.0):
    """重采样信号到目标采样率和时长"""
    if x is None:
        return None
    x = np.array(x).reshape(-1)

    target_len = int(target_fs * target_dur)
    n_samples = int(len(x) * target_fs / fs_in)

    # 重采样
    x_resampled = resample(x, n_samples)

    # 对齐长度
    if len(x_resampled) > target_len:
        return x_resampled[:target_len]
    elif len(x_resampled) < target_len:
        return np.pad(x_resampled, (0, target_len - len(x_resampled)))
    else:
        return x_resampled


def infer_fs(row):
    """推断采样率"""
    path = str(row["file"]).lower()
    if "12khz" in path:
        return 12000
    if "48khz" in path:
        return 48000

    # fallback: 用长度推断
    for sig in ["DE_time", "FE_time", "BA_time"]:
        arr = row.get(sig, None)
        if arr is not None:
            n = len(arr)
            if 90000 <= n <= 110000:
                return 12000
            if 350000 <= n <= 410000:
                return 48000
    return None


def process_source_wide(wide_df):
    """处理源域数据，重采样到32kHz"""
    processed = []
    for _, row in wide_df.iterrows():
        fs_in = infer_fs(row)
        if fs_in is None:
            print(f"[WARN] 仍无法推断采样率：{row['file']}")
            continue

        de_new = resample_to_8s_32k(row.get("DE_time"), fs_in)
        fe_new = resample_to_8s_32k(row.get("FE_time"), fs_in)
        ba_new = resample_to_8s_32k(row.get("BA_time"), fs_in)

        processed.append({
            "file": row["file"],
            "status": row["status"],
            "fault_size_inch": row["fault_size_inch"],
            "position": row["position"],
            "load": row["load"],
            "data_id": row["data_id"],
            "DE_time": de_new,
            "FE_time": fe_new,
            "BA_time": ba_new,
            "RPM": row["RPM"],
        })

    return pd.DataFrame(processed, columns=wide_df.columns)


def _to_1d(arr):
    """把 wide_df 的 cell 转为 1D numpy 数组"""
    if arr is None:
        return None
    a = np.array(arr)
    if a.size == 0:
        return None
    if a.ndim == 1:
        return a
    if a.ndim == 2 and 1 in a.shape:
        return a.reshape(-1)
    return a.reshape(-1)


def _rpm_from_cell(cell):
    """提取 RPM 单值"""
    if cell is None:
        return None
    a = np.array(cell).astype("float64", copy=False)
    if a.size == 0:
        return None
    return float(a.reshape(-1)[0])


def expand_wide_df(wide_df):
    """将宽格式DataFrame展开为长格式（内存密集型操作）"""
    all_parts = []
    for _, row in wide_df.iterrows():
        meta = {
            "file": row["file"],
            "status": row["status"],
            "fault_size_inch": row["fault_size_inch"],
            "position": row["position"],
            "load": row["load"],
            "data_id": row["data_id"],
        }

        # 提取三路信号
        de = _to_1d(row["DE_time"])
        fe = _to_1d(row["FE_time"])
        ba = _to_1d(row["BA_time"])
        rpm_val = _rpm_from_cell(row["RPM"])

        max_len = max(len(x) if x is not None else 0 for x in [de, fe, ba])
        if max_len == 0:
            continue

        def align(arr):
            if arr is None:
                return np.full(max_len, np.nan)
            if len(arr) < max_len:
                return np.concatenate([arr, np.full(max_len - len(arr), np.nan)])
            return arr[:max_len]

        part = pd.DataFrame({
            **meta,
            "DE_time": align(de),
            "FE_time": align(fe),
            "BA_time": align(ba),
            "RPM": np.full(max_len, rpm_val if rpm_val is not None else np.nan)
        })
        all_parts.append(part)

    long_table = pd.concat(all_parts, ignore_index=True)
    return long_table


def create_summary_table(wide_df):
    """创建轻量级的数据摘要表，避免内存问题"""
    summary_rows = []
    
    for _, row in wide_df.iterrows():
        # 基本元数据
        base_info = {
            "file": row["file"],
            "status": row["status"],
            "fault_size_inch": row["fault_size_inch"],
            "position": row["position"],
            "load": row["load"],
            "data_id": row["data_id"],
            "RPM": _rpm_from_cell(row["RPM"])
        }
        
        # 对每个信号计算基本统计信息
        for signal_name in ["DE_time", "FE_time", "BA_time"]:
            signal_data = _to_1d(row[signal_name])
            if signal_data is not None and len(signal_data) > 0:
                signal_info = base_info.copy()
                signal_info.update({
                    "signal_type": signal_name,
                    "signal_length": len(signal_data),
                    "signal_mean": np.mean(signal_data),
                    "signal_std": np.std(signal_data),
                    "signal_min": np.min(signal_data),
                    "signal_max": np.max(signal_data),
                    "signal_rms": np.sqrt(np.mean(signal_data**2))
                })
                summary_rows.append(signal_info)
    
    return pd.DataFrame(summary_rows)


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


def extract_features_from_wide(wide_df, signal_col="DE_time", fs=32000):
    """
    直接从宽格式DataFrame提取特征，避免内存问题
    输入: wide_df (含 file/status/.../signal_col/RPM)
    输出: X (特征矩阵), y (标签)
    """
    feature_rows = []
    labels = []

    for _, row in wide_df.iterrows():
        # 获取信号数据
        signal_data = row[signal_col]
        if signal_data is None:
            continue
            
        x = _to_1d(signal_data)
        if x is None or len(x) == 0:
            continue

        # 获取转速 fr (Hz)
        rpm_val = _rpm_from_cell(row["RPM"])
        fr = (rpm_val/60.0) if rpm_val and rpm_val > 0 else 1.0

        feats = {}
        feats.update(time_features(x))
        feats.update(freq_features(x, fs, fr))
        feats.update(tf_features(x, fs, fr))

        feats["file"] = row["file"]
        feature_rows.append(feats)
        labels.append(row["status"])

    X = pd.DataFrame(feature_rows).set_index("file")
    y = pd.Series(labels, index=X.index, name="status")
    return X, y


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

        feats["file"] = fid
        feature_rows.append(feats)
        labels.append(group["status"].iloc[0])

    X = pd.DataFrame(feature_rows).set_index("file")
    y = pd.Series(labels, index=X.index, name="status")
    return X, y


def main():
    """主程序入口"""
    print("开始源域数据整合...")
    
    # 1. 创建文件摘要
    print("1. 创建文件摘要...")
    df = create_file_summary()
    df.to_csv('df.csv', index=None)
    print(f"   文件摘要已保存，共{len(df)}条记录")
    
    # 2. 创建宽格式DataFrame
    print("2. 创建宽格式DataFrame...")
    wide_df = create_wide_dataframe(df)
    wide_df.to_csv('data.csv', index=None)
    wide_df.to_pickle("wide_df.pkl")
    print(f"   宽格式数据已保存，维度：{wide_df.shape}")
    
    # 3. 重采样到32kHz
    print("3. 重采样到32kHz...")
    new_wide_df = process_source_wide(wide_df)
    new_wide_df.to_csv('源域数据汇总整理_共享.csv', index=None)
    print(f"   重采样后数据维度：{new_wide_df.shape}")
    
    # 4. 创建轻量级数据摘要表
    print("4. 创建数据摘要表...")
    summary_table = create_summary_table(new_wide_df)
    summary_table.to_csv('源域数据32khz摘要.csv', index=None)
    print(f"   数据摘要表已保存，维度：{summary_table.shape}")
    
    # 5. 直接从宽格式提取特征（避免内存问题）
    print("5. 提取特征...")
    X, y = extract_features_from_wide(new_wide_df, signal_col="DE_time", fs=32000)
    print(f"   特征矩阵维度：{X.shape}，标签维度：{y.shape}")
    
    # 可选：如果需要完整长格式数据且内存充足，可以取消下面的注释
    # print("6. 展开为长格式（可选，需要大量内存）...")
    # long_table = expand_wide_df(new_wide_df)
    # long_table.to_csv('源域数据32khz整理.csv', index=None)
    # print(f"   长格式数据已保存，维度：{long_table.shape}")
    
    print("源域数据整合完成！")
    
    return df, wide_df, new_wide_df, summary_table, X, y


if __name__ == "__main__":
    df, wide_df, new_wide_df, summary_table, X, y = main()
