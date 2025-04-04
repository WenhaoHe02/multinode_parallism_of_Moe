import torch
from torch import nn
import torch.nn.functional as F
from llama2_and_deepseek_Moe import MoeRouter, BasicExpert
import torch.distributed as dist
from torch.profiler import profile, tensorboard_trace_handler, schedule
import time
from config import DistConfig, MoeConfig


class _AllToAll:
    @staticmethod
    def forward(input: torch.Tensor) -> torch.Tensor:  # type: ignore
        input = input.contiguous()
        output = torch.empty_like(input)
        if dist.is_initialized():
            dist.all_to_all_single(output, input)
        else:
            output = input
        return output


def get_fused_cumsum_sub_one():
    return lambda mask: torch.cumsum(mask, dim=0) - 1


def all_to_all_wrapper(input: torch.Tensor):
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    cpu_start = time.time() * 1000
    # cuda_start.record()
    output = _AllToAll.forward(input)
    # cuda_end.record()
    cpu_end = time.time() * 1000
    return output


def get_all2all_group(expert_num: int):
    if dist.is_initialized():
        world_size = dist.get_world_size()
        assert world_size <= expert_num
        assert expert_num % world_size == 0
        all2all_groups_list = [[i for i in range(world_size)]]

        _all2all_group_idx = all2all_groups_list
        _all2all_groups = [dist.new_group(g) for g in all2all_groups_list]

        my_group_idx = _find_my_group_index(_all2all_group_idx)
        return _all2all_groups[my_group_idx]


def _find_my_group_index(grouped_ranks):
    my_rank = dist.get_rank()
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
        self.experts = nn.ModuleList([BasicExpert(config.hidden_dim, config.hidden_dim)])
        self.router = MoeRouter(config)
        self.dist_config = dist_config
        self.all2all_group = get_all2all_group(self.expert_num)
        self.all2all_size = dist.get_world_size()

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, hidden_dim = x.size()
        hidden_states = x.view(-1, hidden_dim)
        # print(f"hidden_states: {hidden_states.shape}")
        router_logits, router_weights, selected_experts_indices, expert_mask = (
            self.router(hidden_states)
        )
        normalized_logits = F.softmax(router_logits, dim=1)
        index_of_best_expert = torch.argmax(normalized_logits, dim=1)
        optimal_index = torch.argsort(index_of_best_expert)
        start = time.time()
        sorted_decision = index_of_best_expert[optimal_index]  # indexing,并行执行
        size_list = torch.bincount(sorted_decision, minlength=self.expert_num)
       
        recv_sizes = torch.zeros_like(size_list)
        dist.all_to_all_single(recv_sizes, size_list)

        # 主流继续执行计算
        sorted_hidden_states = hidden_states[optimal_index]
        send_chunks = list(torch.split(sorted_hidden_states, size_list.tolist()))

        # print(f"send_chunks: {send_chunks}, shape: {send_chunks[0].shape}")
        recv_tokens = [torch.zeros(int(recv_sizes[i].item()), hidden_dim, dtype=torch.float64, device=self.dist_config.device) for i in range(self.all2all_size)]
        # print(f"recv_tokens: {recv_tokens}, shape: {recv_tokens[0].shape}, length: {len(recv_tokens)}")

        dist.all_to_all(recv_tokens, send_chunks)

        # print(f"rank: {dist.get_rank()}, recv_tokens: {recv_tokens}, shape: {recv_tokens[1].shape}, length: {len(recv_tokens)}")
        chunk = torch.cat(recv_tokens, dim=0)  # 沿着第 0 维拼接
        # for chunk, expert in zip(recv_tokens, self.experts):
        expert_outputs = [self.experts[0](chunk)]
        expert_output =list(torch.split(expert_outputs[0], recv_sizes.tolist(), dim=0))
        # print(f"expert_outputs: {expert_outputs}, len: {len(expert_outputs)}, shape: {expert_outputs[0].shape}")
        rerecv_tokens = [torch.zeros(int(size_list[i].item()), hidden_dim, dtype=torch.float64, device=self.dist_config.device) for i in range(self.all2all_size)]
        dist.all_to_all(rerecv_tokens, expert_output)

        combined_output = torch.cat(rerecv_tokens, dim=0)  # 沿着第 0 维拼接
        combined_output = combined_output.reshape(batch_size, seq_len, hidden_dim)

        best_expert_probabilities = torch.gather(normalized_logits, dim=1, index=index_of_best_expert.unsqueeze(1))
        best_expert_probabilities = best_expert_probabilities.squeeze(1)  # 去掉多余的维度
        
        combined_output = combined_output * best_expert_probabilities.view(batch_size, seq_len, 1)

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

    # def forward(self, x):
    #     # x shape 是 (b, s, hidden_dim)
    #     # 1. 使用 tensorboard_trace_handler 记录 moe 模型部分的 trace
    #     with torch.no_grad():
    #         with profile(
    #             record_shapes=True,
    #             profile_memory=True,
    #             on_trace_ready=tensorboard_trace_handler('./log_dir/moe_opt')
    #         ) as prof_moe:
    #             sparse_moe_out = self.moe_model(x)
    #         print("MOE 模型部分 trace 已保存至 ./log_dir/moe")
        
    #     # 2. 使用 tensorboard_trace_handler 记录 shared experts 部分的 trace
    #     with torch.no_grad():
    #         with profile(
    #             record_shapes=True,
    #             profile_memory=True,
    #             on_trace_ready=tensorboard_trace_handler('./log_dir/shared_experts')
    #         ) as prof_experts:
    #             shared_experts_out = [expert(x) for expert in self.shared_experts]
    #         print("Shared Experts 部分 trace 已保存至 ./log_dir/shared_experts")
        
    #     shared_experts_out = torch.stack(shared_experts_out, dim=0).sum(dim=0)

    #     # 合并 moe 和 shared experts 部分的输出
    #     return sparse_moe_out + shared_experts_out


    def forward(self, x):
    # x 的形状为 (b, s, hidden_dim)
        with torch.no_grad():
            # 1. 通过 moe 模型
            sparse_moe_out = self.moe_model(x)
        
        # 2. 通过 shared experts
        shared_experts_out = [expert(x) for expert in self.shared_experts]
        shared_experts_out = torch.stack(shared_experts_out, dim=0).sum(dim=0)
    
        # 合并 moe 和 shared experts 部分的输出
        return sparse_moe_out + shared_experts_out


