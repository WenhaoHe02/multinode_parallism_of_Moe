import torch
import torch.distributed as dist
import os
import time

# 全局通信组对象
group_1 = None
group_2 = None

def setup(rank, world_size):
    global group_1, group_2
   
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  
    
    # 定义 group 1 用于 all-reduce
    if rank == 0 or rank == 2:
        group_1 = dist.new_group([0, 2])
    else:
        group_1 = dist.new_group([1, 3])

    # 定义 group 2 用于 all-gather
    if rank == 0 or rank == 1:
        group_2 = dist.new_group([0, 1])
    else:
        group_2 = dist.new_group([2, 3])


def cleanup():
    dist.destroy_process_group()


def allreduce_and_gather_fusion(tensor, rank, world_size):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group_1)


    dist.barrier()


    gathered_tensors = [torch.zeros_like(tensor) for _ in range(2)]
    
    dist.all_gather(gathered_tensors, tensor, group=group_2)

    dist.barrier()

    return gathered_tensors


def allreduce_and_gather_no_fusion(tensor, rank, world_size):
    # 执行全局 all-reduce 操作
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    # 执行全局 all-gather 操作
    dist.all_gather(gathered_tensors, tensor)

    return gathered_tensors


def benchmark(rank, world_size, fusion=True):
    setup(rank, world_size)
    
    # 创建一个输入张量
    tensor = torch.ones(10).to(device=f"cuda:{rank}")

    # 记录开始时间
    start_time = time.time()
    if fusion:
        allreduce_and_gather_fusion(tensor, rank, world_size)
    else:
        allreduce_and_gather_no_fusion(tensor, rank, world_size)
    # 记录结束时间
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Rank {rank} elapsed time: {elapsed_time:.6f} seconds")

    cleanup()


if __name__ == "__main__":
    world_size = 4  # 假设有 4 个进程
    rank = int(os.environ['RANK'])  # 获取当前进程的 rank

    print("Running benchmark with fusion strategy...")
    benchmark(rank, world_size, fusion=True)

    # print("\nRunning benchmark without fusion strategy...")
    # benchmark(rank, world_size, fusion=False)
