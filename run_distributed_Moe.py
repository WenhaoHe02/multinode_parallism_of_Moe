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
    dist_config.device = torch.device(f"cuda:{dist.get_rank()+4}")
    # print(f"rank {dist.get_rank()} device {dist_config.device}")

import torch
import time

# def run_distributed_share_expert_moe(warmup: int, runs: int):
#     # x = torch.rand(3, 4, 16)
#     x = torch.arange(2 * 128 * 8192, dtype=torch.float64).reshape(2, 128, 8192)
#     # x = torch.arange(1 * 1 * 8192, dtype=torch.float64).reshape(1, 1, 8192)
#     # config = MoeConfig(16, 3, 1)
#     config = MoeConfig(8192, 3, 1)
#     dist_config = DistConfig()
#     dist_config.world_size = 3
#     init_dist(dist_config)
#     share_expert_moe = DistShareExpertMOE(config, dist_config)
#     share_expert_moe_baseline = DistShareExpertMOEBaseline(config, dist_config)
#     share_expert_moe = share_expert_moe.to(device=dist_config.device, dtype=torch.float64)
#     share_expert_moe_baseline = share_expert_moe_baseline.to(device=dist_config.device, dtype=torch.float64)
#     x = x.to(device=dist_config.device, dtype=torch.float64)
    
#     # Set models to evaluation mode
#     share_expert_moe_baseline.eval()
#     share_expert_moe.eval()
    
#     # Warmup phase (no timing)
#     with torch.no_grad():
#         for _ in range(warmup):
#             _ = share_expert_moe(x)
#             _ = share_expert_moe_baseline(x)

#     # Initialize GPU events for time measurement
#     total_time = 0
#     total_time_baseline = 0

#     # Comparison flag
#     results_are_equal = True

#     # Run and measure GPU time
#     with torch.no_grad():
#         for _ in range(runs):
#             # Record start time for Baseline ShareExpertMOE
#             start_event_baseline = torch.cuda.Event(enable_timing=True)
#             end_event_baseline = torch.cuda.Event(enable_timing=True)
            
#             start_event_baseline.record()
#             out_baseline = share_expert_moe_baseline(x)
#             end_event_baseline.record()

#             # Synchronize to make sure the events are completed
#             torch.cuda.synchronize()
#             total_time_baseline += start_event_baseline.elapsed_time(end_event_baseline)  # Time in milliseconds

#             # Record start time for ShareExpertMOE
#             start_event_moe = torch.cuda.Event(enable_timing=True)
#             end_event_moe = torch.cuda.Event(enable_timing=True)

#             start_event_moe.record()
#             out_moe = share_expert_moe(x)
#             end_event_moe.record()

#             # Synchronize to make sure the events are completed
#             torch.cuda.synchronize()
#             total_time += start_event_moe.elapsed_time(end_event_moe)  # Time in milliseconds

#             if dist.get_rank() == 0:
#                 print(f"out_moe: {out_moe[0:1][0:3][0:3]}", f"shape: {out_moe.shape}")
#                 print(f"out_baseline: {out_baseline[0:1][0:3][0:3]}", f"shape: {out_baseline.shape}")

#             # Compare outputs from both models
#             if not torch.allclose(out_moe, out_baseline, atol=1e-6):
#                 results_are_equal = False

#     # Print results
#     avg_time = total_time / runs
#     avg_time_baseline = total_time_baseline / runs

#     print(f"ShareExpertMOE average GPU time over {runs} runs: {avg_time:.2f} ms")
#     print(f"Baseline ShareExpertMOE average GPU time over {runs} runs: {avg_time_baseline:.2f} ms")

#     # Output comparison result
#     if results_are_equal:
#         print("The results from ShareExpertMOE and Baseline ShareExpertMOE are the same.")
#     else:
#         print("The results from ShareExpertMOE and Baseline ShareExpertMOE are different.")

def run_distributed_share_expert_moe(warmup: int, runs: int):
    # x = torch.rand(3, 4, 16)
    x = torch.rand(2, 128, 8192, dtype=torch.float64) 
    # x = torch.arange(1 * 1 * 8192, dtype=torch.float64).reshape(1, 1, 8192)
    # config = MoeConfig(16, 3, 1)
    config = MoeConfig(8192, 3, 1)
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
        for run in range(runs):
           

            # 测试 Baseline ShareExpertMOE
            start_time = time.time()
            out_baseline = share_expert_moe_baseline(x)
            torch.cuda.synchronize()
            total_time_baseline += (time.time() - start_time) * 1000  # 转换为毫秒

             # 测试 ShareExpertMOE
            start_time = time.time()
            out_moe = share_expert_moe(x)
            torch.cuda.synchronize()
            total_time += (time.time() - start_time) * 1000  # 转换为毫秒
        

    # 打印结果
    avg_time = total_time / runs
    avg_time_baseline = total_time_baseline / runs

    print(f"rank:{torch.distributed.get_rank()}, ShareExpertMOE average time over {runs} runs: {avg_time:.2f} ms")
    print(f"rank:{torch.distributed.get_rank()}, Baseline ShareExpertMOE average time over {runs} runs: {avg_time_baseline:.2f} ms")

    # 输出比较结果
    if results_are_equal:
        print("The results from ShareExpertMOE and Baseline ShareExpertMOE are the same.")
    else:
        print("The results from ShareExpertMOE and Baseline ShareExpertMOE are different.")

# 运行示例
run_distributed_share_expert_moe(10, 10)
