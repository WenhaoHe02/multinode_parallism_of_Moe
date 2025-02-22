import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run_all2all():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # 创建一个一维张量
    tensor = torch.tensor(
        [rank * 10 + i for i in range(world_size)], dtype=torch.float32, device=device
    )

    # 创建一个输出张量，准备存储接收到的数据
    output_tensor = torch.zeros(world_size, dtype=torch.float32, device=device)

    # 使用 all_to_all 进行张量交换
    dist.all_to_all([output_tensor], [tensor])

    # 打印当前进程的输出张量
    print(f"Rank {rank} received tensor: {output_tensor}")


if __name__ == "__main__":
    run_all2all()
