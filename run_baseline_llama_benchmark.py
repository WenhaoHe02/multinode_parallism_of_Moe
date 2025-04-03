import time
import torch
from llama2_pp_demo import Llama, LlamaConfig

def run_baseline_benchmark(warmup: int, runs: int):
    # Create config and model
    config = LlamaConfig()
    model = Llama(config).cuda()
    
    # Prepare input data
    batch_size = config.batch_size
    max_seq = config.max_seq
    random_input = torch.randint(0, config.vocab_size, (batch_size, max_seq)).cuda()
    random_target = torch.randint(0, config.vocab_size, (batch_size, max_seq)).cuda()
    
    # Set to eval mode
    model.eval()
    
    # Warmup phase
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(random_input)
    
    # Benchmark original model
    total_time = 0
    with torch.no_grad():
        for _ in range(runs):
            start_time = time.time()
            logits, _ = model(random_input)
            torch.cuda.synchronize()
            end_time = time.time()
            total_time += (end_time - start_time) * 1000  # Convert to milliseconds
    
    # Calculate and print results
    avg_time = total_time / runs
    print(f"\nBaseline Benchmark Results:")
    print(f"Average time over {runs} runs: {avg_time:.2f} ms")

if __name__ == "__main__":
    warmup = 5
    runs = 10
    run_baseline_benchmark(warmup, runs) 

