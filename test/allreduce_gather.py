import torch
import torch.distributed as dist
import os
import time


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def allreduce_and_gather_fusion(tensor, rank, world_size):
    if rank == 0 or rank == 2:
        group_1 = dist.new_group([0, 2])
    else:
        group_1 = dist.new_group([1, 3])

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group_1)

    if rank == 0 or rank == 1:
        group_2 = dist.new_group([0, 1])
    else:
        group_2 = dist.new_group([2, 3])

    gathered_tensors = [torch.zeros_like(tensor) for _ in range(2)]
    dist.all_gather(gathered_tensors, tensor, group=group_2)

    return gathered_tensors


def allreduce_and_gather_no_fusion(tensor, rank, world_size):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)

    return gathered_tensors


def benchmark(rank, world_size, fusion=True):
    setup(rank, world_size)
    tensor = torch.ones(10).cuda(rank)

    start_time = time.time()
    if fusion:
        allreduce_and_gather_fusion(tensor, rank, world_size)
    else:
        allreduce_and_gather_no_fusion(tensor, rank, world_size)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Rank {rank} elapsed time: {elapsed_time:.6f} seconds")

    cleanup()


if __name__ == "__main__":
    world_size = 4
    rank = int(os.environ['RANK'])

    print("Running benchmark with fusion strategy...")
    benchmark(rank, world_size, fusion=True)

    print("\nRunning benchmark without fusion strategy...")
    benchmark(rank, world_size, fusion=False)
