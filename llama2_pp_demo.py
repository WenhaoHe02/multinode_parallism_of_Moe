import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import *
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe, PipelineStage
import os

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

        for i in range(config.n_layer):
            super().add_module(f"block{i}", Block(config))
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

        for i in range(self.n_layer):
            block = getattr(self, f"block{i}")
            x = block(x)
        
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
    

class LlamaEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.token_embd_table = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embd_table = nn.Embedding(config.max_seq, config.hidden_dim)

    def forward(self, idx: torch.Tensor):
        batch, seq_len = idx.size()
        token_embd = self.token_embd_table(idx)  # [batch_size, seq_len, hidden_dim]
        postion_embd = self.position_embd_table(
            torch.arange(seq_len, device=idx.device)
        )
        x = token_embd + postion_embd
        return x

class LlamaFinal(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.rn_final = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor):
        x = self.rn_final(x)
        logits = self.lm_head(x)
        return logits

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
# OMP_NUM_THREADS=1 torchrun --nproc-per-node 14 llama2_pp_demo.py

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
model: Llama = Llama(config).to(device)

split_spec = { f"block{stage}": SplitPoint.END for stage in range(config.n_layer) }

batch_size = config.batch_size
max_seq = config.max_seq
chunks = 4

# Initialize distributed environment
import torch.distributed as dist

dist.init_process_group(rank=rank, world_size=world_size)

# Pipeline stage is our main pipeline runtime. It takes in the pipe object,
# the rank of this process, and the device.
if rank == 0:
    module = LlamaEmbedding(config)
elif rank == world_size - 1:
    module = LlamaFinal(config)
else:
    module = Block(config)
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
