import math
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import *

from config import LlamaConfig

class SingleHeadAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.key = nn.Linear(config.hidden_dim, config.head_size)
        self.value = nn.Linear(config.hidden_dim, config.head_size)
        self.query = nn.Linear(config.hidden_dim, config.head_size)
        self.register_buffer(
            "attention_mask", torch.tril(torch.ones(config.max_seq, config.max_seq))
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head_size = config.head_size

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, hidden_dim = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weight = q @ k.transpose(-2, -1)
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0, float("-inf")
        )
        weight = F.softmax(weight, dim=-1) // math.sqrt(self.head_size)
        self = self.dropout(weight)
        output = weight @ v
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.heads = nn.ModuleList(
            [SingleHeadAttention(config) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.proj(x)
        output = self.dropout(x)
        return output


class MLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.rn1 = nn.LayerNorm(config.hidden_dim)
        self.rn2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, x):
        x = x + self.att(self.rn1(x))
        x = x + self.mlp(self.rn2(x))
        return x


class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.token_embd_table = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embd_table = nn.Embedding(config.max_seq, config.hidden_dim)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.n_layer = config.n_layer
        self.rn_final = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        # 使用tie weight减少嵌入参数
        # linear(4->8)，实际上shape是8,4
        self.token_embd_table.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, target=None):
        batch, seq_len = idx.size()
        token_embd = self.token_embd_table(idx)  # [batch_size, seq_len, hidden_dim]
        postion_embd = self.position_embd_table(
            torch.arange(seq_len, device=idx.device)
        )
        x = token_embd + postion_embd
        x = self.blocks(x)
        x = self.rn_final(x)
        logits = self.lm_head(x)
        if target is None:
            loss = None
        else:
            batch_size, seq_len, vocab_size = logits.size()
            logits = logits.view(batch_size * seq_len, vocab_size)
            target = target.view(batch_size * seq_len)
            loss = F.cross_entropy(logits, target)
        return logits, loss
    
class LlamaPipeStage(nn.Module):
    def __init__(self, from_model: Llama, block_range: Tuple[int, int]):
        super().__init__()
        self.is_first = block_range[0] == 0
        self.is_last = block_range[1] == from_model.n_layer - 1
        self.blocks = from_model.blocks[block_range[0]:block_range[1]]
        if self.is_first:
            self.token_embd_table = from_model.token_embd_table
            self.position_embd_table = from_model.position_embd_table
        elif self.is_last:
            self.rn_final = from_model.rn_final
            self.lm_head = from_model.lm_head

    def forward(self, x: torch.Tensor):
        if self.is_first:
            batch, seq_len = x.size()
            token_embd = self.token_embd_table(x)  # [batch_size, seq_len, hidden_dim]
            postion_embd = self.position_embd_table(
                torch.arange(seq_len, device=x.device)
            )
            x = token_embd + postion_embd

        for block in self.blocks:
            x = block(x)

        if self.is_last:
            x = self.rn_final(x)
            logits = self.lm_head(x)
            return logits
        else:
            return x

# To run a distributed inference job, we must launch the script in multiple
# different processes. We are using `torchrun` to do so in this example.
# `torchrun` defines two environment variables: `RANK` and `WORLD_SIZE`,
# which represent the index of this process within the set of processes and
# the total number of processes, respectively.
#
# To learn more about `torchrun`, see
# https://pytorch.org/docs/stable/elastic/run.html
#
# 示例的调用方法: 
# torchrun --nproc-per-node 4 llama2_pp_demo.py

def main():
    from torch.distributed.pipelining import ScheduleGPipe, PipelineStage

    torch.manual_seed(11)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Figure out device to use
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")


    # Create the pipeline
    config = LlamaConfig()
    full_model: Llama = Llama(config).to(device)

    batch_size = config.batch_size
    max_seq = config.max_seq
    chunks = 3

    # Initialize distributed environment
    import torch.distributed as dist

    dist.init_process_group(rank=rank, world_size=world_size)

    # Pipeline stage is our main pipeline runtime. It takes in the pipe object,
    # the rank of this process, and the device.
    blocks_per_stage = config.n_layer // world_size

    if rank == 0:
        module = LlamaPipeStage(full_model, (0, blocks_per_stage))
    elif rank == world_size - 1:
        module = LlamaPipeStage(full_model, (rank * blocks_per_stage, config.n_layer))
    else:
        module = LlamaPipeStage(full_model, (rank * blocks_per_stage, (rank + 1) * blocks_per_stage))
    stage = PipelineStage(module, rank, world_size, device)

    # Attach to a schedule
    schedule = ScheduleGPipe(stage, chunks)

    # Input data
    random_input = torch.randint(0, config.vocab_size, (batch_size, max_seq))
    random_target = torch.randint(0, config.vocab_size, (batch_size, max_seq))

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
