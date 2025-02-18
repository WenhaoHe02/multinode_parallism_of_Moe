import os
import time

import torch
from llama2_and_deepseek_Moe import MoeConfig, ShareExpertMOE
import torch.distributed as dist

class DistConfig:
    world_size: int = 1
    backend: str = 'nccl'
    device = None
    capacity_factor: 0.5


def init_dist(dist_config: DistConfig):
    dist.init_process_group(backend=dist_config.backend, world_size=dist_config.world_size)
    dist_config.device = torch.device(f'cuda:{dist.get_rank()}')


def run_distributed_share_expert_moe(warmup: int, runs: int):
    x = torch.rand(2, 4, 16)
    config = MoeConfig(16, 2, 2)
    dist_config = DistConfig()
    dist_config.world_size = 4
    init_dist(dist_config)
    share_expert_moe = ShareExpertMOE(config)
    share_expert_moe.to(device=dist_config.device, dtype=torch.bfloat16)
    x.to(device=dist_config.device, dtype=torch.bfloat16)
    share_expert_moe.eval()
    with torch.no_grad():
        for _ in range(warmup):
            out = share_expert_moe(x)

    total_time = 0
    with torch.no_grad():
        for _ in range(runs):
            start_time = time.time
            out = share_expert_moe(x)
            total_time += (time.time() - start_time) * 1000  # 转换为毫秒

    print(out[0].shape, out[1].shape)


run_distributed_share_expert_moe()