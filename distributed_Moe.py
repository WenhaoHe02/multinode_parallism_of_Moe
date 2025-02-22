import torch
from torch import nn
import torch.nn.functional as F
from llama2_and_deepseek_Moe import MoeRouter, BasicExpert
import torch.distributed as dist
import time
from config import DistConfig, MoeConfig


class _AllToAll:
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


def all_to_all_wrapper(all2all_group: dist.ProcessGroup, input: torch.Tensor):
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    cpu_start = time.time() * 1000
    cuda_start.record()
    output = _AllToAll.forward(all2all_group, input)
    cuda_end.record()
    cpu_end = time.time() * 1000
    return output


def get_all2all_group(expert_num: int):
    if torch.distributed.is_initialized():
        world_size = dist.get_world_size()
        assert world_size <= expert_num
        assert expert_num % world_size == 0
        all2all_groups_list = [[i for i in range(world_size)]]

        _all2all_group_idx = all2all_groups_list
        _all2all_groups = [dist.new_group(g) for g in all2all_groups_list]

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
        router_logits, router_weights, selected_experts_indices, expert_mask = (
            self.router(hidden_states)
        )
        expert_layer = self.experts
        normalized_logits = F.softmax(router_logits, dim=1)
        index_of_best_expert = torch.argmax(normalized_logits, dim=1)
        classified_expert_mask = F.one_hot(
            index_of_best_expert, num_classes=self.expert_num
        ).unsqueeze(-1)
        normalized_logits_sum = (normalized_logits * classified_expert_mask).sum(dim=1)
        locations = get_fused_cumsum_sub_one()(classified_expert_mask)
        classified_expert_mask = (
            classified_expert_mask * torch.lt(locations, capacity).float()
        )
        locations_sum = torch.sum(locations * classified_expert_mask, dim=1)
        new_normalized_logits = normalized_logits_sum.unsqueeze(
            -1
        ) * classified_expert_mask.to(
            normalized_logits_sum.dtype
        )  # einsum("s,se->se")
        classified_location = F.one_hot(locations_sum, num_classes=capacity).unsqueeze(
            -1
        )
        combine_weight = torch.bmm(
            # einsum("se,sc->sec")
            new_normalized_logits.unsqueeze(-1),
            classified_location.to(new_normalized_logits).unsqueeze(1),
        )
        dispatch_mask = combine_weight.bool()
        dispatch_mask = dispatch_mask.to(hidden_states.dtype).permute(
            1, 2, 0
        )  # S,E,C -> E,C,S
        # E, C, S = dispatch_mask.size()
        dispatched_input = torch.mm(
            dispatch_mask.view(self.expert_num * capacity, batch_size * seq_len),
            hidden_states,
        )  # -> (E*C),M
        dispatched_input = all_to_all_wrapper(self.all2all_group, dispatched_input)
        dispatched_input = dispatched_input.reshape(
            self.all2all_size, 1, -1, hidden_dim
        )  # 1是local expert数量
        chunks = dispatched_input.chunk(1, dim=1)  # 1是local expert数量
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            expert_outputs += [expert(chunk)]
        print(f"expert_outputs: {expert_outputs}")
        expert_output = torch.cat(expert_outputs, dim=1)
        expert_output = all_to_all_wrapper(self.all2all_group, expert_output)
        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(
            self.all2all_size * 1, -1, hidden_dim
        )  # 1是local expert
        # einsum("sec,ecm->sm")
        combined_output = combine_weight.view(
            batch_size * seq_len, self.expert_num * capacity
        ).mm(expert_output.view(self.expert_num * capacity, self.hidden_dim))
        # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
        combined_output = combined_output[: batch_size * seq_len, :]
        combined_output = combined_output.reshape(batch_size, seq_len, hidden_dim)
        combined_output = combined_output[:batch_size, :, :]
        return combined_output


class DistShareExpertMOE(nn.Module):
    def __init__(self, config, dist_config: DistConfig):
        super().__init__()

        self.moe_model = DistSparseMoe(config, dist_config)
        self.shared_experts = nn.ModuleList(
            [
                BasicExpert(config.hidden_dim, config.hidden_dim)
                for _ in range(config.shared_expert_num)
            ]
        )

    def forward(self, x):
        # x shape 是 (b, s, hidden_dim)
        # 首先过 moe 模型
        sparse_moe_out, router_logits = self.moe_model(x)

        # 针对的还是 x 的每一个
        # 然后过 shared experts
        shared_experts_out = [
            expert(x) for expert in self.shared_experts
        ]  # 每一个 expert 的输出 shape 是 (b, s, hidden_dim)

        shared_experts_out = torch.stack(shared_experts_out, dim=0).sum(dim=0)

        # 把 sparse_moe_out 和 shared_experts_out 加起来
        return sparse_moe_out + shared_experts_out, router_logits
