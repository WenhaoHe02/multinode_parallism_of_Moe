import math
import time
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import *
from argparse import ArgumentParser
from config import LlamaConfig, MoeConfig

torch.manual_seed(11)





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
        weight = F.softmax(weight / math.sqrt(self.head_size), dim=-1)
        weight = self.dropout(weight)
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
        output = self.proj(output)
        output = self.dropout(output)
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


class BasicExpert(nn.Module):
    def __init__(self, feature_in: int, feature_out: int):
        super().__init__()
        self.fc = nn.Linear(feature_in, feature_out)

    def forward(self, x: torch.Tensor):
        return self.fc(x)


class BasicMoe(nn.Module):
    def __init__(self, feature_in: int, feature_out: int, num_expert):
        super().__init__()
        self.gate = nn.Linear(in_features=feature_in, out_features=num_expert)
        self.experts = nn.ModuleList(
            BasicExpert(feature_in, feature_out) for _ in range(num_expert)
        )

    def forward(self, x):
        # x: [bs, fin]
        # expert_weights: [bs, num_expert]
        expert_weights = self.gate(x)
        expert_out_list = [expert(x) for expert in self.experts]
        # expert_outputs [bs, 1, fout]
        expert_outputs = [expert_out.unsqueeze(1) for expert_out in expert_out_list]
        # expert_output[bs, num_experts, fout]
        expert_output = torch.concat(expert_outputs, dim=1)
        expert_weights = F.softmax(expert_weights, dim=1)
        # expert_weights: [bs, 1, num_experts]
        expert_weights = expert_weights.unsqueeze(1)
        # OUTPUT: [bs, 1, fout]
        output = expert_weights @ expert_output
        return output.squeeze(1)





class MoeRouter(nn.Module):
    def __init__(self, config: MoeConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_dim, config.expert_num)
        self.expert_num = config.expert_num
        self.top_k = config.topk

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.gate(x)
        # router_logits[bs*seq_len, expert_num]
        router_probs = F.softmax(router_logits, dim=1, dtype=torch.float32)
        router_weights, selected_experts_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        router_weights = router_weights / router_weights.sum(dim=-1, keepdims=True)
        router_weights = router_weights.to(x.dtype)
        expert_mask = F.one_hot(selected_experts_indices, num_classes=self.expert_num)
        expert_mask = expert_mask.permute(2, 1, 0)
        return router_logits, router_weights, selected_experts_indices, expert_mask

        # router_logits [bs*seq_len, expert_num]
        # router_weights[bs*seq_len, topk]
        # selected_experts_indices[bs * seq_len, topk]
        # expert_mask (expert_num, topk, bs * seq_len)


class SparseMoe(nn.Module):
    def __init__(self, config: MoeConfig):
        super().__init__()
        self.config = config
        self.topk = config.topk
        self.hidden_dim = config.hidden_dim
        self.expert_num = config.expert_num
        self.experts = nn.ModuleList(
            BasicExpert(
                config.hidden_dim,
                config.hidden_dim,
            )
            for _ in range(config.expert_num)
        )
        self.router = MoeRouter(config)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, hidden_dim = x.size()
        hidden_states = x.view(-1, hidden_dim)
        router_logits, router_weights, selected_experts_indices, expert_mask = (
            self.router(hidden_states)
        )
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for expert_idx in range(self.expert_num):
            expert_layer = self.experts[expert_idx]
            # expert_mask[expert_idx] shape 是 (top_k, b * s)
            idx, top_x = torch.where(expert_mask[expert_idx])
            # idx 和 top_x 都是一维 tensor
            # idx 的值是 0 或 1, 表示这个 token 是作为当前专家的 top1 还是 top2
            # top_x 的值是 token 在 batch*seq_len 中的位置索引
            # 例如对于 batch_size=2, seq_len=4 的输入:
            # top_x 的值范围是 0-7, 表示在展平后的 8 个 token 中的位置
            # idx 的值是 0/1, 表示这个 token 把当前专家作为其 top1/top2 专家

            # hidden_states 的 shape 是 (bs * seq_len, hidden_dim)
            # 需要取到 top_x 对应的 hidden_states
            current_state = hidden_states.unsqueeze(0)[:, top_x, :].reshape(
                -1, hidden_dim
            )  # （selected_token_number, hidden_dim）

            # router_weight 的 shape 是 (b * s, top_k)
            current_hidden_states = expert_layer(current_state) * router_weights[
                top_x, idx
            ].unsqueeze(
                -1
            )  # （selected_token_number, 1） 这里有广播c

            # 把当前专家的输出加到 final_hidden_states 中
            # 方式1 的写法性能更好，并且方式1容易出现
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
            # 方式2
            # final_hidden_states[top_x] += current_hidden_states.to(hidden_states.dtype)
            # 方式2 的写法性能更差，并且方式2容易出现错误，+= 操作在处理重复索引时需要多次读写内存，可能会导致竞争条件

            # 把 final_hidden_states 还原到原来的 shape
        final_hidden_states = final_hidden_states.reshape(
            batch_size, seq_len, hidden_dim
        )

        return final_hidden_states, router_logits  # shape 是 (b * s, expert_number)


class ShareExpertMOE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.moe_model = SparseMoe(config)
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


# def test_share_expert_moe():
#     # Construct input data
#     batch_size, seq_len, hidden_dim = 2, 128, 8192
#     # batch_size, seq_len, hidden_dim = 1, 1, 16
#     x = torch.arange(batch_size * seq_len * hidden_dim, dtype=torch.float32).reshape(batch_size, seq_len, hidden_dim)

#     parser = ArgumentParser()
#     parser.add_argument("--device_id", type=int, default=-1, help="Specify GPU device id, use default -1 for CPU")
#     args = parser.parse_args()

#     # Determine the device based on the device_id
#     device = f"cuda:{args.device_id}" if args.device_id != -1 and torch.cuda.is_available() else "cpu"
#     x = x.to(device)

#     # Construct configuration, note MoeConfig includes the shared_expert_num field
#     config = MoeConfig(hidden_dim=8192, expert_num=3, topk=1, shared_expert_num=1)
#     share_expert_moe = ShareExpertMOE(config).to(device)

#     # Warmup phase: run multiple forward passes without timing
#     warmup = 10
#     for _ in range(warmup):
#         _ = share_expert_moe(x)
#     if device.startswith("cuda"):
#         torch.cuda.synchronize()

#     # GPU timing using torch.cuda.Event
#     run_iterations = 1
#     start_event = torch.cuda.Event(enable_timing=True)
#     end_event = torch.cuda.Event(enable_timing=True)

#     start_event.record()  # Start recording GPU time
#     for _ in range(run_iterations):
#         out = share_expert_moe(x)
#         if device.startswith("cuda"):
#             torch.cuda.synchronize()  # Ensure all CUDA operations have finished

#     end_event.record()  # End recording GPU time
#     torch.cuda.synchronize()  # Ensure the final operation is completed

#     # Measure elapsed time in milliseconds
#     elapsed_time_ms = start_event.elapsed_time(end_event)  # Time in milliseconds

#     # Print the first few data and average execution time
#     print(f"Average execution time: {elapsed_time_ms / run_iterations:.6f} ms per iteration")
def test_share_expert_moe():
    # 构造输入数据
    batch_size, seq_len, hidden_dim = 2, 128, 8192
    # batch_size, seq_len, hidden_dim = 1, 1, 16
    x = torch.arange(batch_size * seq_len * hidden_dim, dtype=torch.float32).reshape(batch_size, seq_len, hidden_dim)

    parser = ArgumentParser()
    parser.add_argument("--device_id", type=int, default=-1, help="指定 GPU 设备 id, 不使用 GPU 则保持默认值 -1")
    args = parser.parse_args()

    # 根据 device_id 确定运行设备
    device = f"cuda:{args.device_id}" if args.device_id != -1 and torch.cuda.is_available() else "cpu"
    x = x.to(device)

    # 构造配置，注意 MoeConfig 中应包含 shared_expert_num 字段
    config = MoeConfig(hidden_dim=8192, expert_num=3, topk=1, shared_expert_num=1)
    share_expert_moe = ShareExpertMOE(config).to(device)

    # warmup 阶段：多次前向传播，但不计时
    warmup = 10
    for _ in range(warmup):
        _ = share_expert_moe(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # 正式计时 run_iterations 次，计算平均时间（单位：毫秒）
    run_iterations = 1
    start_time = time.time()
    for _ in range(run_iterations):
        out = share_expert_moe(x)
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / run_iterations * 1000
    print(f"Average time per forward pass: {avg_time_ms:.2f} ms")
    # 打印前3个数据


if __name__ == "__main__":
    test_share_expert_moe()

# if __name__ == "__main__":

#     config = LlamaConfig()


#     model = Llama(config)


#     batch_size = config.batch_size
#     max_seq = config.max_seq


#     random_input = torch.randint(0, config.vocab_size, (batch_size, max_seq))


#     random_target = torch.randint(0, config.vocab_size, (batch_size, max_seq))


#     logits, loss = model(random_input, random_target)

#     print("logits shape:", logits.shape)
#     print("loss:", loss.item())
