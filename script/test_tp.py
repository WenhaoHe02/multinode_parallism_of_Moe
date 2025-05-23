import torch
import torch.distributed as dist
from loguru import logger
import sys
import nnx
from dataclasses import dataclass
import argparse
import sys
import os
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
sys.path.append('/home/zbw/multinode_parallism_of_Moe')

logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")

@dataclass
class Config:
    input_features: int
    output_features: int
    batch_size: int
    seq_len: int
    d_model: int
    mode: str
    warmup: int = 10
    runs: int = 20

def main(config):
    try:
        if not dist.is_initialized(): dist.init_process_group(backend='nccl')
        rank, device = dist.get_rank(), torch.device(f"cuda:{dist.get_rank()}")

        model = nnx.ParallelLinear(
            in_features=config.input_features,
            out_features=config.output_features,
            mode=config.mode,
        ).to(device)

        relu = nn.ReLU()

        model.load_state_dict
        
        input = torch.randn(config.batch_size, config.seq_len, config.d_model).to(device)

        log_dir = f"logs/traces"
        os.makedirs(log_dir, exist_ok=True)
        trace_file = f"{log_dir}/trace_f{config.d_model}_b{config.batch_size}_g{dist.get_world_size()}_s{config.seq_len}_rank{rank}.json"
        
        # Warmup
        for _ in range(config.warmup):
            _ = relu(model(input))

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(config.runs):
                with record_function("model_inference"):
                    _ = relu(model(input))
                prof.step()

        # 每个 rank 都保存自己的 trace 文件
        prof.export_chrome_trace(trace_file)
        logger.info(f"[rank {rank}] Trace file saved to: {trace_file}")

        # 原有的时间测量代码
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        elapsed_times = []

        for _ in range(config.runs):
            torch.cuda.synchronize()
            start_event.record()
            _ = model(input)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            elapsed_times.append(elapsed_time)

        avg_time = sum(elapsed_times) / len(elapsed_times)
        logger.info(f"[rank {rank}] Average time: {avg_time} ms")
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description='Parallel Linear Layer Benchmark')
    parser.add_argument('--input-features', type=int, default=8192)
    parser.add_argument('--output-features', type=int, default=8192)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--d-model', type=int, default=8192)
    parser.add_argument('--mode', type=str, choices=['row', 'column', 'combined'], default='combined')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--runs', type=int, default=20)
    args = parser.parse_args()
    return Config(**vars(args))

if __name__ == "__main__":
    config = parse_args()
    main(config)
