import os
import pandas as pd
from glob import glob
import re
from typing import List, Tuple
import numpy as np

def parse_log_file(log_path: str) -> Tuple[float, float]:
    """解析日志文件，提取时间信息"""
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # 提取所有rank的平均时间
    times = []
    for line in lines:
        if "Average time:" in line:
            time_str = line.split("Average time:")[-1].strip()
            time_str = time_str.replace("ms", "").strip()
            times.append(float(time_str))
    
    if not times:
        return 0.0, 0.0
        
    # 返回所有rank中的最大平均时间和最大时间
    return np.max(times), np.max(times)

def analyze_experiments(log_dir: str) -> pd.DataFrame:
    """分析实验结果并生成DataFrame"""
    results = []
    
    # 获取所有日志文件
    log_files = glob(os.path.join(log_dir, "*.log"))
    
    for log_file in log_files:
        # 从文件名解析实验参数
        basename = os.path.basename(log_file)
        name_parts = basename.replace(".log", "").split("_")
        
        mode = name_parts[0]
        features = int(name_parts[1].replace("f", ""))
        batch_size = int(name_parts[2].replace("b", ""))
        gpus = int(name_parts[3].replace("g", ""))
        seq_len = int(name_parts[4].replace("s", ""))
        
        # 解析日志文件获取时间信息
        avg_time, max_time = parse_log_file(log_file)
        
        results.append({
            "Mode": mode,
            "Input Features": features,
            "Batch Size": batch_size,
            "GPU Number": gpus,
            "Sequence Length": seq_len,
            "Average Time (ms)": avg_time,
            "Max Time (ms)": max_time
        })
    
    # 创建DataFrame并排序
    df = pd.DataFrame(results)
    df = df.sort_values(by=["Mode", "Input Features", "Batch Size", "GPU Number", "Sequence Length"])
    
    return df

def main():
    # 获取最新的日志目录
    log_dirs = glob("logs/*/")
    latest_log_dir = max(log_dirs, key=os.path.getctime)
#    latest_log_dir = './logs/20250227_141848'
    
    # 分析实验结果
    results_df = analyze_experiments(latest_log_dir)
    
    # 保存到CSV
    output_path = os.path.join(latest_log_dir, "summary_python.csv")
    results_df.to_csv(output_path, index=False, float_format="%.6f")
    
    # 打印摘要统计
    print("\nExperiment Summary:")
    print(f"Total experiments: {len(results_df)}")
    print("\nAverage times by mode:")
    print(results_df.groupby("Mode")["Average Time (ms)"].mean())
    
    # 保存详细的统计报告
    report_path = os.path.join(latest_log_dir, "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("Detailed Analysis Report\n")
        f.write("======================\n\n")
        
        # 按模式分组的统计
        for mode in results_df["Mode"].unique():
            mode_data = results_df[results_df["Mode"] == mode]
            f.write(f"\nMode: {mode}\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average time: {mode_data['Average Time (ms)'].mean():.6f} ms\n")
            f.write(f"Max time: {mode_data['Max Time (ms)'].max():.6f} ms\n")
            f.write(f"Min time: {mode_data['Average Time (ms)'].min():.6f} ms\n")
            f.write(f"Std dev: {mode_data['Average Time (ms)'].std():.6f} ms\n")

if __name__ == "__main__":
    main()
