import os
import time

import torch
from config import MoeConfig, DistConfig
from optimized_distributed_Moe import DistShareExpertMOE
from distributed_Moe import DistShareExpertMOE as DistShareExpertMOEBaseline
import torch.distributed as dist

def init_dist(dist_config: DistConfig):
    dist.init_process_group(
        backend=dist_config.backend, world_size=dist_config.world_size
    )
    dist_config.device = torch.device(f"cuda:{dist.get_rank()}")
    # print(f"rank {dist.get_rank()} device {dist_config.device}")

def run_distributed_share_expert_moe(warmup: int, runs: int):
    # x = torch.rand(1, 1, 16)
    x = torch.arange(8 * 512 * 2048, dtype=torch.float64).reshape(8, 512, 2048)
    config = MoeConfig(2048, 3, 2)
    dist_config = DistConfig()
    dist_config.world_size = 3
    init_dist(dist_config)
    share_expert_moe = DistShareExpertMOE(config, dist_config)
    share_expert_moe_baseline = DistShareExpertMOEBaseline(config, dist_config)
    share_expert_moe = share_expert_moe.to(device=dist_config.device, dtype=torch.float64)
    share_expert_moe_baseline = share_expert_moe_baseline.to(device=dist_config.device, dtype=torch.float64)
    x = x.to(device=dist_config.device, dtype=torch.float64)
    
    # 设置为评估模式
    share_expert_moe_baseline.eval()
    share_expert_moe.eval()
    
    # Warmup阶段（不计时）
    with torch.no_grad():
        for _ in range(warmup):
            _ = share_expert_moe(x)
            _ = share_expert_moe_baseline(x)

    # 记录运行时间
    total_time = 0
    total_time_baseline = 0

    # 用来比较两个模型的结果
    results_are_equal = True

    # 运行并计时
    with torch.no_grad():
        for _ in range(runs):
            # 测试 ShareExpertMOE
            start_time = time.time()
            out_moe = share_expert_moe(x)
            total_time += (time.time() - start_time) * 1000  # 转换为毫秒

            # 测试 Baseline ShareExpertMOE
            start_time = time.time()
            out_baseline = share_expert_moe_baseline(x)
            total_time_baseline += (time.time() - start_time) * 1000  # 转换为毫秒
            
            if (dist.get_rank() == 0):
                print(f"out_moe: {out_moe[0:20]}", f"shape: {out_moe.shape}")
                print(f"out_baseline: {out_baseline[0:20]}", f"shape: {out_baseline.shape}")

            # 比较两个模型的输出
            if not torch.allclose(out_moe, out_baseline, atol=1e-6):
                results_are_equal = False

    # 打印结果
    avg_time = total_time / runs
    avg_time_baseline = total_time_baseline / runs
    if (dist.get_rank() == 0):
        print(f"out_baseline前3个数据:{out_baseline[0:1][0:3][0:10]}")

    print(f"ShareExpertMOE average time over {runs} runs: {avg_time:.2f} ms")
    print(f"Baseline ShareExpertMOE average time over {runs} runs: {avg_time_baseline:.2f} ms")

    # 输出比较结果
    if results_are_equal:
        print("The results from ShareExpertMOE and Baseline ShareExpertMOE are the same.")
    else:
        print("The results from ShareExpertMOE and Baseline ShareExpertMOE are different.")

# 运行示例
run_distributed_share_expert_moe(20, 10)
