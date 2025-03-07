import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional
from enum import Enum
from dataclasses import dataclass

class ParallelMode(Enum):
    """Enumeration for different parallel modes"""
    COLUMN = "column"
    ROW = "row"
    COMBINED = "combined"

@dataclass
class ParallelConfig:
    """Configuration class for parallel layers"""
    mode: ParallelMode
    world_size: int
    rank: int
    gather_output: bool = True

class ParallelLinear(nn.Module):
    """
    Unified parallel linear layer supporting both column and row parallelism.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        parallel_config: Configuration for parallelization
        bias: Whether to include bias term
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: str = "combined",
        gather_output: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(mode, str):
            mode = ParallelMode[mode.upper()]

        self.config = ParallelConfig(
            mode=mode,
            world_size=dist.get_world_size(),
            rank=dist.get_rank(),
            gather_output=gather_output,
        )

        self.in_features = in_features
        self.out_features = out_features
        
        # Calculate partition sizes
        self.input_size_per_partition = in_features // self.config.world_size
        self.output_size_per_partition = out_features // self.config.world_size
        
        # Initialize appropriate linear layer based on mode
        if self.config.mode == ParallelMode.COLUMN:
            self.linear = nn.Linear(
                in_features,
                self.output_size_per_partition,
                bias=bias
            )
        elif self.config.mode == ParallelMode.ROW:
            self.linear = nn.Linear(
                self.input_size_per_partition,
                out_features,
                bias=bias and self.config.rank == 0
            )
        else:  # COMBINED mode
            self.linear = nn.Linear(
                self.input_size_per_partition,
                self.output_size_per_partition,
                bias=bias and self.config.rank == 0
            )
    
    @torch.no_grad()
    def _gather_output(self, output_parallel: torch.Tensor, dim: Optional[int]=-1) -> torch.Tensor:
        """Gather outputs from all parallel processes"""
        output_list = [torch.empty_like(output_parallel) for _ in range(self.config.world_size)]
        dist.all_gather(output_list, output_parallel)
        return torch.cat(output_list, dim=dim)
    
    @torch.no_grad()
    def _scatter_input(self, input_: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(input_, self.config.world_size, dim=-1)
        return chunks[self.config.rank]
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """       
        Args:
            input_: Input tensor of shape [*, in_features]
        
        Returns:
            Output tensor of shape [*, out_features]
        """
        # Handle input partitioning
        if self.config.mode in [ParallelMode.ROW, ParallelMode.COMBINED]:
            input_parallel = self._scatter_input(input_)
        else:
            input_parallel = input_
            
        # Local forward pass
        output_parallel = self.linear(input_parallel)
        
        # Handle output reduction/gathering
        if self.config.mode in [ParallelMode.ROW, ParallelMode.COMBINED]:
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)
            
        if self.config.gather_output and self.config.mode in [ParallelMode.COLUMN, ParallelMode.COMBINED]:
            output = self._gather_output(output_parallel)
            return output
            
        return output_parallel

    def _load_from_state_dict(
        self, 
        state_dict: dict, 
        prefix: str, 
        local_metadata: dict, 
        strict: bool,
        missing_keys: list, 
        unexpected_keys: list, 
        error_msgs: list,
    ) -> None:
        """Custom state dict loading for parallel linear layer."""
        weight_key = prefix + 'linear.weight'
        bias_key = prefix + 'linear.bias'
        
        if weight_key in state_dict:
            weight = state_dict[weight_key]
            
            # Broadcast weight from rank 0 to all other ranks
            if self.config.rank != 0:
                weight = torch.empty_like(weight, device=self.linear.weight.device)
            dist.broadcast(weight, src=0)
            
            # Remove the weight key so parent class doesn't try to load it
            del state_dict[weight_key]
            
            # Split and load the weight according to parallel mode
            with torch.no_grad():
                if self.config.mode == ParallelMode.COLUMN:
                    start_idx = self.config.rank * self.output_size_per_partition
                    end_idx = start_idx + self.output_size_per_partition
                    self.linear.weight.copy_(weight[start_idx:end_idx, :])
                    
                elif self.config.mode == ParallelMode.ROW:
                    start_idx = self.config.rank * self.input_size_per_partition
                    end_idx = start_idx + self.input_size_per_partition
                    self.linear.weight.copy_(weight[:, start_idx:end_idx])
                    
                else:  # COMBINED mode
                    in_start = self.config.rank * self.input_size_per_partition
                    in_end = in_start + self.input_size_per_partition
                    out_start = self.config.rank * self.output_size_per_partition
                    out_end = out_start + self.output_size_per_partition
                    self.linear.weight.copy_(weight[out_start:out_end, in_start:in_end])
        
        # Handle bias if present
        if bias_key in state_dict:
            bias = state_dict[bias_key]
            
            # Broadcast bias from rank 0 to all other ranks
            if self.config.rank != 0:
                bias = torch.empty_like(bias, device=self.linear.weight.device)
            dist.broadcast(bias, src=0)
            
            # Remove the bias key
            del state_dict[bias_key]
            
            # Load bias according to parallel mode
            if self.linear.bias is not None:
                with torch.no_grad():
                    if self.config.mode == ParallelMode.COLUMN:
                        start_idx = self.config.rank * self.output_size_per_partition
                        end_idx = start_idx + self.output_size_per_partition
                        self.linear.bias.copy_(bias[start_idx:end_idx])
                    elif self.config.mode == ParallelMode.ROW and self.config.rank == 0:
                        self.linear.bias.copy_(bias)
                    elif self.config.mode == ParallelMode.COMBINED and self.config.rank == 0:
                        out_start = self.config.rank * self.output_size_per_partition
                        out_end = out_start + self.output_size_per_partition
                        self.linear.bias.copy_(bias[out_start:out_end])

        # Call parent class to handle any remaining tensors
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, 
            missing_keys, unexpected_keys, error_msgs
        )

def convert_to_parallel(module, mode="column"):
    """
    Convert nn.Linear layers to ParallelLinear layers.
    It should be called after the model is constructed and before the model is moved to the device.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            parallel_linear = ParallelLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                mode=mode,
                bias=child.bias is not None
            )

            if dist.get_rank() == 0:
                state_dict = {
                    'linear.weight': child.weight.data,
                    'linear.bias': child.bias.data if child.bias is not None else None
                }
                parallel_linear.load_state_dict(state_dict)
            setattr(module, name, parallel_linear)
        else:
            convert_to_parallel(child, mode)