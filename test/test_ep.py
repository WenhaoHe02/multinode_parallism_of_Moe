import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp

from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")


class Expert(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)  # 保持输入输出维度相同
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class ExpertParallel(nn.Module):
    def __init__(self, expert, expert_degree):
        super().__init__()
        self.expert = expert
        self.expert_degree = expert_degree

    def forward(self, x: torch.Tensor):
        expert_degree = self.expert_degree
        local_batch_size, seq_len, d_model = x.shape

        # 第一次all_to_all通信：分发输入到各个专家
        x_split = list(x.chunk(expert_degree, dim=1))
        recv_x = [torch.zeros_like(x_split[0]) for _ in range(expert_degree)]
        dist.all_to_all(recv_x, x_split)

        # 合并接收数据并处理
        expert_input = torch.cat(recv_x, dim=1)
        expert_output = self.expert(expert_input)

        # 第二次all_to_all通信：收集处理结果
        output_split = list(expert_output.chunk(expert_degree, dim=1))
        final_output = [torch.zeros_like(output_split[0]) for _ in range(expert_degree)]
        dist.all_to_all(final_output, output_split)

        return torch.cat(final_output, dim=0)


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    logger.info(f"[rank {rank}] {device} initialized")

    return rank, world_size, device


def run():
    rank, world_size, device = setup()

    # 模型参数
    seq_len = 128
    d_model = 512
    batch_size = 8  # 总batch_size需要能被world_size整除

    # 初始化模型
    expert = Expert(d_model, 4 * d_model).to(device=device)
    model = ExpertParallel(expert, world_size).to(device=device)

    # 示例数据
    x = torch.randn(batch_size // world_size, seq_len, d_model, device=device)

    # 前向传播
    output = model(x)
    print(f"Rank {rank} output shape: {output.shape}")


if __name__ == "__main__":
    run()
