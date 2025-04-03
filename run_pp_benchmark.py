import os
import time
import torch
import torch.distributed as dist
from llama2_pp_demo import Llama, LlamaConfig, LlamaPipeStage
from torch.distributed.pipelining import ScheduleGPipe, PipelineStage

def init_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    return rank, device

def run_pp_benchmark(warmup: int, runs: int):
    # Initialize distributed environment
    rank, device = init_dist()
    
    # Create config and model
    config = LlamaConfig()
    config.batch_size = 12
    full_model = Llama(config).to(device)
    
    # Prepare input data
    batch_size = config.batch_size
    max_seq = config.max_seq
    random_input = torch.randint(0, config.vocab_size, (batch_size, max_seq)).to(device)
    random_target = torch.randint(0, config.vocab_size, (batch_size, max_seq)).to(device)
    
    # Setup pipeline stages
    chunks = config.batch_size // 4
    world_size = dist.get_world_size()
    blocks_per_stage = config.n_layer // world_size
    
    if rank == 0:
        module = LlamaPipeStage(full_model, (0, blocks_per_stage))
    elif rank == world_size - 1:
        module = LlamaPipeStage(full_model, (rank * blocks_per_stage, config.n_layer))
    else:
        module = LlamaPipeStage(full_model, (rank * blocks_per_stage, (rank + 1) * blocks_per_stage))
    
    stage = PipelineStage(module, rank, world_size, device)
    schedule = ScheduleGPipe(stage, chunks)
    
    # Set to eval mode
    full_model.eval()
    
    # Warmup phase
    with torch.no_grad():
        for _ in range(warmup):
            if rank == 0:
                schedule.step(random_input)
            elif rank == world_size - 1:
                _ = schedule.step()
            else:
                schedule.step()
            # Synchronize all processes after each warmup run
            dist.barrier()
    
    # Benchmark PP model
    total_time_pp = 0
    with torch.no_grad():
        for _ in range(runs):
            # Synchronize all processes before starting the timer
            dist.barrier()
            if rank == 0:
                start_time = time.time()
            
            if rank == 0:
                schedule.step(random_input)
            elif rank == world_size - 1:
                logits = schedule.step()
            else:
                schedule.step()
            
            # Synchronize all processes before ending the timer
            dist.barrier()
            if rank == 0:
                end_time = time.time()
                total_time_pp += (end_time - start_time) * 1000  # Convert to milliseconds
    
    # Benchmark original model (only on rank 0)
    total_time_original = 0
    if rank == 0:
        with torch.no_grad():
            for _ in range(runs):
                start_time = time.time()
                logits, _ = full_model(random_input)
                #torch.cuda.synchronize()
                end_time = time.time()
                total_time_original += (end_time - start_time) * 1000
    
    # Calculate and print results
    if rank == 0:
        avg_time_pp = total_time_pp / runs
        avg_time_original = total_time_original / runs
        print(f"\nBenchmark Results:")
        print(f"Pipeline Parallel (PP) average time over {runs} runs: {avg_time_pp:.2f} ms")
        print(f"Original model average time over {runs} runs: {avg_time_original:.2f} ms")
        print(f"Speedup: {avg_time_original/avg_time_pp:.2f}x")

if __name__ == "__main__":
    # Example usage:
    # torchrun --nproc-per-node 4 run_pp_benchmark.py
    warmup = 3
    runs = 5
    run_pp_benchmark(warmup, runs) 
