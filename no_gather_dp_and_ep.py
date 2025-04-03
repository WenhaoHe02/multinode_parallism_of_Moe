import torch
from torch import nn
import torch.nn.functional as F
import math
import time
import torch.distributed as dist

# 从配置文件中导入各配置参数
from config import LlamaConfig, MoeConfig, DistConfig
# 从模型模块中导入必要模块（Llama 用于前置 transformer 层，MoeRouter 和 BasicExpert 用于分布式 MoE 部分）
from llama2_and_deepseek_Moe import Llama, MoeRouter, BasicExpert

##############################################
# 如果 Llama 模型中没有 encode 方法，可添加如下：
##############################################
if not hasattr(Llama, "encode"):
    def encode(self, idx: torch.Tensor):
        # idx: [batch, seq_len]
        batch, seq_len = idx.size()
        token_embd = self.token_embd_table(idx)  # [b, s, hidden_dim]
        position_embd = self.position_embd_table(torch.arange(seq_len, device=idx.device))
        x = token_embd + position_embd
        x = self.blocks(x)
        hidden = self.rn_final(x)
        return hidden
    Llama.encode = encode

##############################################
# 分布式 MoE 部分
##############################################

class _AllToAll:
    @staticmethod
    def forward(group: dist.ProcessGroup, input: torch.Tensor) -> torch.Tensor:
        input = input.contiguous()
        output = torch.empty_like(input)
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output, input, group=group)
        else:
            # 若未初始化分布式，则直接返回输入
            assert group is None
            output = input
        return output

def get_fused_cumsum_sub_one():
    return lambda mask: torch.cumsum(mask, dim=0) - 1

def all_to_all_wrapper(all2all_group: dist.ProcessGroup, input: torch.Tensor):
    return _AllToAll.forward(all2all_group, input)

def get_all2all_group(expert_num: int):
    if torch.distributed.is_initialized():
        world_size = dist.get_world_size()
        assert world_size <= expert_num
        assert expert_num % world_size == 0
        all2all_groups_list = [[i for i in range(world_size)]]
        _all2all_groups = [dist.new_group(g) for g in all2all_groups_list]
        my_group_idx = _find_my_group_index(all2all_groups_list)
        return _all2all_groups[my_group_idx]
    return None

def _find_my_group_index(grouped_ranks):
    my_rank = dist.get_rank()
    for i, group in enumerate(grouped_ranks):
        if my_rank in group:
            return i
    raise RuntimeError("Current rank not found in any group.")

class DistSparseMoe(nn.Module):
    def __init__(self, config: MoeConfig, dist_config: DistConfig):
        super().__init__()
        self.config = config
        self.topk = config.topk
        self.hidden_dim = config.hidden_dim
        self.expert_num = config.expert_num
        # 示例中只创建一个专家，实际应用时可增加数量
        self.experts = nn.ModuleList([BasicExpert(config.hidden_dim, config.hidden_dim)])
        self.router = MoeRouter(config)
        self.capacity_factor = dist_config.capacity_factor
        self.all2all_group = get_all2all_group(self.expert_num)
        self.all2all_size = dist.get_world_size() if torch.distributed.is_initialized() else 1

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, hidden_dim = x.size()
        capacity = 100  # 固定容量
        hidden_states = x.view(-1, hidden_dim)
        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)
        normalized_logits = F.softmax(router_logits, dim=1)
        index_of_best_expert = torch.argmax(normalized_logits, dim=1)
        classified_expert_mask = F.one_hot(index_of_best_expert, num_classes=self.expert_num)
        normalized_logits_sum = (normalized_logits * classified_expert_mask).sum(dim=1)
        locations = get_fused_cumsum_sub_one()(classified_expert_mask)
        classified_expert_mask = classified_expert_mask * torch.lt(locations, capacity).int()
        locations_sum = torch.sum(locations * classified_expert_mask, dim=1, dtype=torch.int64)
        new_normalized_logits = normalized_logits_sum.unsqueeze(-1) * classified_expert_mask.to(normalized_logits_sum.dtype)
        classified_location = F.one_hot(locations_sum, num_classes=capacity)
        combine_weight = torch.bmm(new_normalized_logits.unsqueeze(-1),
                                   classified_location.to(new_normalized_logits).unsqueeze(1))
        dispatch_mask = combine_weight.bool()
        dispatch_mask = dispatch_mask.to(hidden_states.dtype).permute(1, 2, 0)
        dispatched_input = torch.mm(dispatch_mask.view(self.expert_num * capacity, batch_size * seq_len),
                                      hidden_states)
        dispatched_input = all_to_all_wrapper(self.all2all_group, dispatched_input)
        dispatched_input = dispatched_input.reshape(self.all2all_size, 1, -1, hidden_dim)
        chunks = dispatched_input.chunk(1, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=1)
        expert_output = all_to_all_wrapper(self.all2all_group, expert_output)
        expert_output = expert_output.reshape(self.all2all_size * 1, -1, hidden_dim)
        combined_output = combine_weight.view(batch_size * seq_len, self.expert_num * capacity).mm(
            expert_output.view(self.expert_num * capacity, self.hidden_dim))
        combined_output = combined_output[: batch_size * seq_len, :]
        combined_output = combined_output.reshape(batch_size, seq_len, hidden_dim)
        combined_output = combined_output[:batch_size, :, :]
        return combined_output

class DistShareExpertMOE(nn.Module):
    def __init__(self, config: MoeConfig, dist_config: DistConfig):
        super().__init__()
        self.moe_model = DistSparseMoe(config, dist_config)
        self.shared_experts = nn.ModuleList(
            [BasicExpert(config.hidden_dim, config.hidden_dim) for _ in range(config.shared_expert_num)]
        )

    def forward(self, x):
        # x 的形状为 (batch, seq_len, hidden_dim)
        with torch.no_grad():
            sparse_moe_out = self.moe_model(x)
        shared_experts_out = [expert(x) for expert in self.shared_experts]
        shared_experts_out = torch.stack(shared_experts_out, dim=0).sum(dim=0)
        return sparse_moe_out + shared_experts_out

##############################################
# 自定义 DataParallel：不聚合各 GPU 输出（输出保留在原设备上）
##############################################
class NoGatherDataParallel(nn.DataParallel):
    def gather(self, outputs, output_device):
        return outputs

##############################################
# 分布式初始化函数
##############################################
def init_dist(dist_config: DistConfig):
    dist.init_process_group(
        backend=dist_config.backend, world_size=dist_config.world_size
    )
    dist_config.device = torch.device(f"cuda:{dist.get_rank()}")
    print(f"Rank {dist.get_rank()} using device {dist_config.device}")

##############################################
# Main 函数：先调用 Llama 层，然后分布式 MoE 层
##############################################
def main():
    # 分布式环境及 MoE 部分相关配置
    moe_config = MoeConfig(16, 3, 1)   # MoeConfig 中需包含 batch_size, max_seq, hidden_dim, topk, expert_num, shared_expert_num 等字段
    dist_config = DistConfig()
    dist_config.world_size = 3
    dist_config.backend = "nccl"  # 或 "gloo" 根据实际环境选择
    init_dist(dist_config)

    # ----- 第一阶段：使用 Llama 层编码（输入 token id，输出隐藏状态） -----
    llama_config = LlamaConfig()
    llama_model = Llama(llama_config)
    llama_model.eval()
    # 使用自定义 DataParallel 保持各 GPU 输出不聚合
    llama_model = NoGatherDataParallel(llama_model)
    llama_model.to(dist_config.device)


    # 构造 token id 输入，形状：(batch_size, max_seq)
    dummy_input = torch.randint(0, llama_config.vocab_size, (llama_config.batch_size, llama_config.max_seq), device=dist_config.device)
    with torch.no_grad():
        # 调用 encode 方法获得隐藏表示，形状：(batch_size, max_seq, hidden_dim)
        _, _, hidden_states = llama_model(dummy_input)

    # ----- 第二阶段：分布式 MoE 层处理 Llama 输出的隐藏状态 -----
    moe_model = DistShareExpertMOE(moe_config, dist_config)
    moe_model.to(dist_config.device)
    moe_model.eval()
    
    with torch.no_grad():
        moe_output = moe_model(hidden_states)
    
    # moe_output 为列表，每个元素为对应 GPU 上的结果
    if isinstance(moe_output, list):
        for i, out in enumerate(moe_output):
            print(f"Device (rank) {i} MoE output shape: {out.shape}")
    else:
        print("MoE output shape:", moe_output.shape)

if __name__ == "__main__":
    main()
