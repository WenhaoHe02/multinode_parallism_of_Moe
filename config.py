class DistConfig:
    world_size: int = 8
    backend: str = "nccl"
    device = None
    capacity_factor = 0.5


class MoeConfig:
    def __init__(self, hidden_dim, expert_num, topk, shared_expert_num=2):
        self.hidden_dim = hidden_dim
        self.expert_num = expert_num
        self.topk = topk
        self.shared_expert_num = shared_expert_num
