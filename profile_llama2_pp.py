import os
import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
from llama2_pp_demo import Llama, LlamaConfig, LlamaPipeStage, PipelineStage, ScheduleGPipe
import torch.nn.functional as F

torch.manual_seed(11)

def main():
    # Initialize distributed environment
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(rank=rank, world_size=world_size)

    # Set up device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")

    # Create model and pipeline stages
    config = LlamaConfig()
    full_model: Llama = Llama(config).to(device)
    full_model.eval()  # Set model to evaluation mode
    
    batch_size = config.batch_size
    max_seq = config.max_seq
    chunks = 3

    blocks_per_stage = config.n_layer // world_size

    if rank == 0:
        module = LlamaPipeStage(full_model, (0, blocks_per_stage))
    elif rank == world_size - 1:
        module = LlamaPipeStage(full_model, (rank * blocks_per_stage, config.n_layer))
    else:
        module = LlamaPipeStage(full_model, (rank * blocks_per_stage, (rank + 1) * blocks_per_stage))
    
    stage = PipelineStage(module, rank, world_size, device)
    schedule = ScheduleGPipe(stage, chunks)

    # Prepare input data
    random_input = torch.randint(0, config.vocab_size, (batch_size, max_seq))
    random_target = torch.randint(0, config.vocab_size, (batch_size, max_seq))

    # Set up profiler and tensorboard
    if rank == 0:
        writer = SummaryWriter(f'runs/llama2_pp_world_size_{world_size}')
        
        with torch.no_grad():
            with profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=3,
                    repeat=2
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log_dir/llama2_pp_ws{world_size}'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                # Run the pipeline
                for step in range(6):  # 6 steps to match profiler schedule
                    if rank == 0:
                        schedule.step(random_input)
                    elif rank == world_size - 1:
                        logits = schedule.step()
                    else:
                        schedule.step()
                    
                    if rank == world_size - 1:
                        target = random_target
                        batch_size, seq_len, vocab_size = logits.size()
                        logits = logits.view(batch_size * seq_len, vocab_size)
                        target = target.view(batch_size * seq_len)
                        loss = F.cross_entropy(logits, target)
                        
                        print("logits shape:", logits.shape)
                        print("loss:", loss.item())
                
                    prof.step()

        # Print some statistics
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # Export chrome trace
        prof.export_chrome_trace(f"trace_llama2_pp_world_size_{world_size}.json")
        
        # Export stack trace
        prof.export_stacks(f"stacks_llama2_pp_world_size_{world_size}.txt", "self_cuda_time_total")
    else:
        for step in range(6):
            if rank == 0:
                schedule.step(random_input)
            elif rank == world_size - 1:
                logits = schedule.step()
            else:
                schedule.step()

            if rank == world_size - 1:
                target = random_target
                batch_size, seq_len, vocab_size = logits.size()
                logits = logits.view(batch_size * seq_len, vocab_size)
                target = target.view(batch_size * seq_len)
                loss = F.cross_entropy(logits, target)
                
                print("logits shape:", logits.shape)
                print("loss:", loss.item())

if __name__ == "__main__":
    main() 