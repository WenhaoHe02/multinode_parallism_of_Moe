from dataclasses import dataclass

@dataclass
class MoeConfig:
    def __init__(self, hidden_dim, expert_num, topk, shared_expert_num=2):
        self.hidden_dim = hidden_dim
        self.expert_num = expert_num
        self.topk = topk
        self.shared_expert_num = shared_expert_num

@dataclass
class DistConfig:
    world_size: int = 1
    backend: str = "nccl"
    device = None
    capacity_factor = 3

@dataclass
class LlamaConfig:
    max_seq: int = 512
    batch_size: int = 12
    n_layer: int = 12
    n_head: int = 12
    hidden_dim: int = 384
    dropout: float = 0.1
    head_size: int = hidden_dim // n_head
    # vocab_size: int = 50257
    vocab_size: int = 384