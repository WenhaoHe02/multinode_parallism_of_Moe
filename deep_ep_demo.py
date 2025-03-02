# import torch
# import torch.distributed as dist
# from torch import nn
# import torch.nn.functional as F
# from deep_ep import Buffer, EventOverlap
# from typing import List, Tuple, Optional, Union

# from config import *
# from llama2_and_deepseek_Moe import BasicExpert, MoeRouter


# # Communication buffer (will allocate at runtime)
# _buffer: Optional[Buffer] = None

# # Set the number of SMs to use
# # NOTES: this is a static variable
# Buffer.set_num_sms(24)


# def get_buffer(group: dist.ProcessGroup, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int) -> Buffer:
#     # NOTES: the low-latency mode will consume much more space than the normal mode
#     # So we recommend that `num_max_dispatch_tokens_per_rank` (the actual batch size in the decoding engine) should be less than 256
#     global _buffer
#     num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, group.size(), num_experts)

#     # Allocate a buffer if not existed or not enough buffer size
#     if _buffer is None or _buffer.group != group or not _buffer.low_latency_mode or _buffer.num_rdma_bytes < num_rdma_bytes:
#         # NOTES: for best performance, the QP number **must** be equal to the number of the local experts
#         assert num_experts % group.size() == 0
#         _buffer = Buffer(group, 0, num_rdma_bytes, low_latency_mode=True, num_qps_per_rank=num_experts // group.size())
#     return _buffer


# def dispatch_forward(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
#                      topk_idx: torch.Tensor, topk_weights: torch.Tensor,
#                      num_experts: int, previous_event: Optional[EventOverlap] = None) -> \
#         Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor, List, Tuple, EventOverlap]:
#     # NOTES: an optional `previous_event` means a CUDA event captured that you want to make it as a dependency 
#     # of the dispatch kernel, it may be useful with communication-computation overlap. For more information, please
#     # refer to the docs of `Buffer.dispatch`
#     global _buffer

#     # Calculate layout before actual dispatch
#     num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = \
#         _buffer.get_dispatch_layout(topk_idx, num_experts,
#                                     previous_event=previous_event, async_finish=True,
#                                     allocate_on_comm_stream=previous_event is not None)
#     # Do MoE dispatch
#     # NOTES: the CPU will wait for GPU's signal to arrive, so this is not compatible with CUDA graph
#     # For more advanced usages, please refer to the docs of the `dispatch` function
#     recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
#         _buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights,
#                          num_tokens_per_rank=num_tokens_per_rank, num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
#                          is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
#                          previous_event=previous_event, async_finish=True,
#                          allocate_on_comm_stream=True)
#     # For event management, please refer to the docs of the `EventOverlap` class
#     return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event


# def combine_forward(x: torch.Tensor, handle: Tuple, previous_event: Optional[EventOverlap] = None) -> \
#         Tuple[torch.Tensor, EventOverlap]:
#     global _buffer

#     # Do MoE combine
#     # For more advanced usages, please refer to the docs of the `combine` function
#     combined_x, _, event = _buffer.combine(x, handle, async_finish=True, previous_event=previous_event,
#                                            allocate_on_comm_stream=previous_event is not None)

#     # For event management, please refer to the docs of the `EventOverlap` class
#     return combined_x, event


# class DistributedSparseMoe(nn.Module):
#     def __init__(self, config: MoeConfig):
#         super().__init__()
#         self.config = config
#         self.topk = config.topk
#         self.hidden_dim = config.hidden_dim
#         self.expert_num = config.expert_num

#         # 初始化通信缓冲区
#         self.buffer = None
#         self.group = dist.group.WORLD  # 使用默认进程组，可按需修改

#         # 每个rank只保留部分专家
#         self.local_expert_num = config.expert_num // dist.get_world_size()
#         self.experts = nn.ModuleList(
#             [
#                 BasicExpert(config.hidden_dim, config.hidden_dim)
#                 for _ in range(self.local_expert_num)
#             ]
#         )

#         self.router = MoeRouter(config)

#         # 初始化缓冲区
#         self._init_buffer()

#     def _init_buffer(self):
#         # 初始化通信缓冲区
#         hidden_bytes = self.hidden_dim * torch.finfo(torch.float16).bits // 8
#         self.buffer = get_buffer(
#             group=self.group,
#             hidden_bytes=hidden_bytes,
#             num_max_dispatch_tokens_per_rank=config.max_batch_size * config.max_seq_len,
#             num_experts=self.expert_num,
#         )

#     def forward(self, x: torch.Tensor):
#         batch_size, seq_len, hidden_dim = x.size()
#         hidden_states = x.view(-1, hidden_dim)

#         # 1. 路由计算
#         _, router_weights, selected_experts_indices, _ = self.router(hidden_states)

#         # 2. 使用DeepEP进行分发
#         (
#             recv_x,
#             recv_topk_idx,
#             recv_topk_weights,
#             num_recv_tokens_list,
#             handle,
#             event,
#         ) = dispatch_forward(
#             hidden_states, selected_experts_indices, router_weights, self.expert_num
#         )

#         # 3. 本地专家计算
#         expert_outputs = []
#         for expert_idx in range(self.local_expert_num):
#             # 筛选当前专家处理的tokens
#             mask = (recv_topk_idx % dist.get_world_size() == dist.get_rank()) & (
#                 recv_topk_idx // dist.get_world_size() == expert_idx
#             )
#             selected_tokens = recv_x[mask]

#             if selected_tokens.size(0) > 0:
#                 expert_out = self.experts[expert_idx](selected_tokens)
#                 expert_outputs.append(
#                     expert_out * recv_topk_weights[mask].unsqueeze(-1)
#                 )

#         # 合并本地专家结果
#         local_output = (
#             torch.cat(expert_outputs, dim=0)
#             if expert_outputs
#             else torch.zeros_like(recv_x)
#         )

#         # 4. 使用DeepEP进行结果聚合
#         combined_output, combine_event = combine_forward(local_output, handle)

#         # 等待所有通信完成
#         combine_event.synchronize()

#         # 恢复原始形状
#         final_hidden_states = combined_output.reshape(batch_size, seq_len, hidden_dim)

#         return final_hidden_states, None  # 保持接口兼容


# class ShareExpertMOE(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.moe_model = DistributedSparseMoe(config)
#         self.shared_experts = nn.ModuleList(
#             [
#                 BasicExpert(config.hidden_dim, config.hidden_dim)
#                 for _ in range(config.shared_expert_num)
#             ]
#         )

#     def forward(self, x):
#         # 稀疏MOE部分
#         sparse_moe_out, _ = self.moe_model(x)

#         # 共享专家部分
#         shared_out = torch.stack(
#             [expert(x) for expert in self.shared_experts], dim=0
#         ).sum(dim=0)

#         return sparse_moe_out + shared_out, None
