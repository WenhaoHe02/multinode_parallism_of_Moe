import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass

torch.manual_seed(1024)

@dataclass
class GPTConfig:
    max_seq: int = 512
    batch_size: int = 12
    n_layer: int = 12
    n_head: int = 12
    hidden_dim: int = 768
    dropout: float = 0.1
    head_size: int = hidden_dim // n_head
    vocab_size: int = 50257
class SingleHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.key = nn.Linear(config.hidden_dim, config.head_size)
        self.value = nn.Linear(config.hidden_dim, config.head_size)
        self.query = nn.Linear(config.hidden_dim, config.head_size)
        self.register_buffer('attention_mask', torch.tril(torch.ones(config.max_seq, config.max_seq)))
        self.dropout = nn.Dropout(config.dropout)
        self.head_size = config.head_size
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, hidden_dim = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weight = q @ k.transpose(-2, -1)
        weight = weight.masked_fill(self.attention_mask[:seq_len, :seq_len] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1) // math.sqrt(self.head_size)
        self = self.dropout(weight)
        output = weight @ v
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(config)
                for _ in range(config.n_head)
            ]
        )
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        output = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )
        output = self.proj(x)
        output = self.dropout(x)
        return output

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.rn1 = nn.RMSNorm(config.hidden_dim)
        self.rn2 = nn.RMSNorm(config.hidden_dim)

    def forward(self, x):
        x = x + self.att(self.rn1(x))
        x = x + self.mlp(self.rn2(x))
        return x
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.token_embd_table = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embd_table = nn.Embedding(config.max_seq, config. hidden_dim)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.rn_final = nn.RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        # 使用tie weight减少嵌入参数
        # linear(4->8)，实际上shape是8,4
        self.token_embd_table.weight = self.lm_head.weight

    def _init_weights(self, module):
        if (isinstance(module, nn.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, target=None):
        batch, seq_len = idx.size()
        token_embd = self.token_embd_table(idx) # [batch_size, seq_len, hidden_dim]
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
            batch_size, seq_len,vocab_size = logits.size()
            logits = logits.view(batch_size * seq_len, vocab_size)
            target = target.view(batch_size * seq_len)
            loss = F.cross_entropy(logits, target)
        return logits, loss




config = GPTConfig()


model = GPT(config)


batch_size = config.batch_size
max_seq = config.max_seq


random_input = torch.randint(0, config.vocab_size, (batch_size, max_seq))


random_target = torch.randint(0, config.vocab_size, (batch_size, max_seq))


logits, loss = model(random_input, random_target)

print("logits shape:", logits.shape)
print("loss:", loss.item())



