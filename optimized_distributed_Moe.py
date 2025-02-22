import torch
from torch import nn
import torch.nn.functional as F
from llama2_and_deepseek_Moe import MoeRouter, BasicExpert
import torch.distributed as dist
import time
from config import DistConfig, MoeConfig

class _AllToAll():
    @staticmethod
    def forward(group: dist.ProcessGroup, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        input = input.contiguous()
        output = torch.empty_like(input)
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output, input, group=group)
        else:
            assert group is None
            output = input
        return output
def get_fused_cumsum_sub_one():
    return lambda mask: torch.cumsum(mask, dim=0) - 1

def all_to_all_wrapper(self, input: torch.Tensor):
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    cpu_start = time.time() * 1000
    cuda_start.record()
    output = _AllToAll.forward(self.all2all_group, input)
    cuda_end.record()
    cpu_end = time.time() * 1000
    return output

def get_all2all_group(expert_num:int):
    if torch.distributed.is_initialized():
        world_size = dist.get_world_size()
        assert world_size <= expert_num
        assert expert_num % world_size == 0
        all2all_groups_list = [[i for i in range(world_size)]]


        _all2all_group_idx = all2all_groups_list
        _all2all_groups = [
            dist.new_group(g) for g in all2all_groups_list
        ]

        my_group_idx = _find_my_group_index(_all2all_group_idx)
        return _all2all_groups[my_group_idx]

def _find_my_group_index(grouped_ranks):
    my_rank = torch.distributed.get_rank()
    for i, group in enumerate(grouped_ranks):
        if my_rank in group:
            return i
    raise RuntimeError


class DistSparseMoe(nn.Module):
    def __init__(self, config: MoeConfig, dist_config: DistConfig):
        super().__init__()
        self.config = config
        self.topk = config.topk
        self.hidden_dim = config.hidden_dim
        self.expert_num = config.expert_num
        self.experts = nn.ModuleList(BasicExpert(config.hidden_dim, config.hidden_dim))
        self.router = MoeRouter(config)
        self.capacity_factor = dist_config.capacity_factor
        self.all2all_group = get_all2all_group(self.expert_num)
        self.all2all_size = dist.get_world_size()

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, hidden_dim = x.size()
        capacity = batch_size * seq_len // self.expert_num * self.capacity_factor
        hidden_states = x.view(-1, hidden_dim)
        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)
        expert_layer = self.experts
        normalized_logits = F.softmax(router_logits, dim=1)
        index_of_best_expert = torch.argmax(normalized_logits, dim=1)
        optimal_index = torch.argsort(index_of_best_expert)
        sorted_decision = index_of_best_expert[optimal_index] # indexing,并行执行
        _, size_list = torch.unique(sorted_decision, sorted=False, return_counts=True)
        recv_sizes = all_to_all_wrapper(size_list) #计算隐藏延迟
        send_chunks = torch.split(hidden_states, size_list)
        token_size = recv_sizes.sum()
        recv_tokens = torch.zeros(token_size, hidden_dim)
        dist.all_to_all(recv_tokens, send_chunks)
        expert_outputs = torch.zeros(token_size, hidden_dim)
        for chunk, expert in zip(recv_tokens, self.experts):
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=0)
        expert_output = all_to_all_wrapper(self.all2all_group, expert_output)
        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(
            self.all2all_size * 1, -1, hidden_dim
        ) # 1是local expert
        # einsum("sec,ecm->sm")
        combined_output = combine_weight.view(batch_size * seq_len, self.expert_num * capacity).mm(
            expert_output.view(self.expert_num * capacity, self.hidden_dim)
        )
        # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
        combined_output = combined_output[ : batch_size * seq_len, :]
        combined_output = combined_output.reshape(batch_size, seq_len, hidden_dim)
        combined_output = combined_output[: batch_size, :, :]



        return expert_output