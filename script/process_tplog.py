import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import seaborn as sns
import json

def parse_log_file(log_path: str) -> tuple[list[float], list[float]]:
    """解析日志文件，提取每个rank的时间信息"""
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    times = []
    for line in lines:
        if "Average time:" in line:
            time_str = line.split("Average time:")[-1].strip()
            time_str = time_str.replace("ms", "").strip()
            times.append(float(time_str))
    
    if not times:
        return [], []
    
    return times, times

def analyze_experiments(log_dir: str) -> tuple[pd.DataFrame, dict]:
    """分析实验结果并生成DataFrame和统计字典"""
    results = []
    stats_dict = {}

    log_files = glob(os.path.join(log_dir, "*.log"))

    for log_file in log_files:
        basename = os.path.basename(log_file)
        parts = basename.split('_')
        mode = parts[0]
        batch_size = int(parts[1][1:])
        num_gpus = int(parts[2][1:])
        seq_len = int(parts[3][1:].replace('.log', ''))

        # 只保留 num_gpus > 1 的情况
        if num_gpus <= 1:
            continue  

        times, max_times = parse_log_file(log_file)

        if times:
            avg_time = np.mean(times)
            max_time = np.max(max_times)

            results.append({
                "mode": mode,
                "batch_size": batch_size,
                "num_gpus": num_gpus,
                "seq_len": seq_len,
                "avg_time": avg_time,
                "max_time": max_time
            })

            key = (mode, batch_size, num_gpus, seq_len)
            stats_dict[key] = {
                "times": times,
                "avg_time": avg_time,
                "max_time": max_time
            }

    df = pd.DataFrame(results)
    return df, stats_dict


def plot_performance_graphs(df: pd.DataFrame, log_dir: str):
    """绘制性能图表"""
    # 设置颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 基础颜色
    
    # 创建两个图表：平均时间和最大时间
    for metric in ['avg_time', 'max_time']:
        # 获取所有唯一的seq_len值
        seq_lens = sorted(df['seq_len'].unique())
        
        # 创建子图
        fig, axes = plt.subplots(1, len(seq_lens), figsize=(5*len(seq_lens), 6))
        fig.suptitle(f'{"Average" if metric == "avg_time" else "Maximum"} Time by Sequence Length',
                    fontsize=16, y=1.05)
        
        # 如果只有一个seq_len，确保axes是一个列表
        if len(seq_lens) == 1:
            axes = [axes]
        
        # 为每个seq_len创建一个子图
        for ax, seq_len in zip(axes, seq_lens):
            # 筛选当前seq_len的数据
            data = df[df['seq_len'] == seq_len]
            
            # 获取所有唯一的mode和num_gpus
            modes = sorted(data['mode'].unique())
            gpu_nums = sorted(data['num_gpus'].unique())
            
            # 设置柱状图的位置
            x = np.arange(len(gpu_nums))
            width = 0.8 / len(modes)  # 调整柱子宽度
            
            # 为每个mode画一组柱状图
            for i, mode in enumerate(modes):
                mode_data = data[data['mode'] == mode]
                heights = []
                for gpu in gpu_nums:
                    val = mode_data[mode_data['num_gpus'] == gpu][metric].values
                    heights.append(val[0] if len(val) > 0 else 0)
                
                ax.bar(x + i*width - width*len(modes)/2 + width/2, 
                      heights, width, label=mode, color=colors[i % len(colors)])
            
            # 设置子图的标题和标签
            ax.set_title(f'Sequence Length = {seq_len}')
            ax.set_xlabel('Number of GPUs')
            ax.set_ylabel('Time (ms)')
            ax.set_xticks(x)
            ax.set_xticklabels(gpu_nums)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # 添加图例
            ax.legend()
            
            # 添加数值标签
            for i, mode in enumerate(modes):
                mode_data = data[data['mode'] == mode]
                for j, gpu in enumerate(gpu_nums):
                    val = mode_data[mode_data['num_gpus'] == gpu][metric].values
                    if len(val) > 0:
                        ax.text(j + i*width - width*len(modes)/2 + width/2, 
                               val[0], f'{val[0]:.1f}', 
                               ha='center', va='bottom')
        
        # 调整布局并保存图片
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'{metric}_performance.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()

def main():
    # 获取最新的日志目录
    log_dirs = glob("logs/*/")
    # latest_log_dir = max(log_dirs, key=os.path.getctime)
    latest_log_dir = '/logs/20250318_0315/'
    # 分析实验结果
    df, stats_dict = analyze_experiments(latest_log_dir)
    
    # 保存统计字典
    stats_file = os.path.join(latest_log_dir, "stats.json")
    # 将统计字典的键转换为字符串
    serializable_stats = {f"{mode}_{batch}_{gpus}_{seq}": values 
                         for (mode, batch, gpus, seq), values in stats_dict.items()}
    with open(stats_file, 'w') as f:
        json.dump(serializable_stats, f, indent=4)
    
    # 保存DataFrame到CSV
    csv_file = os.path.join(latest_log_dir, "summary.csv")
    df.to_csv(csv_file, index=False)
    
    # 绘制性能图表
    plot_performance_graphs(df, latest_log_dir)
    
    print(f"Analysis complete. Results saved in {latest_log_dir}")
    print(f"- Summary CSV: {csv_file}")
    print(f"- Statistics JSON: {stats_file}")
    print(f"- Performance graphs: {latest_log_dir}/*_performance.png")

if __name__ == "__main__":
    main()
