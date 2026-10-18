# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Configuration-based training FLOPs estimate for Qwen3-MoE style models.

The estimate follows the MindFormers/Megatron-LM ``num_floating_point_operations``
convention: only the main matrix multiplications are counted, a ``[M, K] x [K, N]``
GEMM costs ``2 * M * K * N`` FLOPs, causal attention scores count half of the
``S x S`` matrix, and one training step costs three forward passes (forward,
activation gradient and weight gradient). Router, norm, RoPE and softmax costs
are ignored.
"""

__all__ = [
    "estimate_train_flops",
]

from typing import Any


def _attention_flops(tokens: int, seq_len: int, config: Any) -> int:
    """Return forward FLOPs of Q/K/V projections, causal attention and output projection."""
    hidden = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", None) or num_heads
    head_dim = getattr(config, "head_dim", None) or hidden // num_heads
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_proj = 2 * tokens * hidden * (q_dim + 2 * kv_dim)
    # QK^T and PV each cost 2 * S * S * head_dim per head; causal masking keeps half.
    core = 2 * (2 * tokens * seq_len * q_dim) // 2
    out_proj = 2 * tokens * q_dim * hidden
    return qkv_proj + core + out_proj


def _is_moe_layer(layer_idx: int, config: Any) -> bool:
    """Mirror the Hugging Face Qwen3-MoE decoder-layer MLP selection."""
    num_experts = getattr(config, "num_experts", 0) or 0
    sparse_step = getattr(config, "decoder_sparse_step", 1) or 1
    mlp_only_layers = getattr(config, "mlp_only_layers", None) or []
    return (
        layer_idx not in mlp_only_layers
        and num_experts > 0
        and (layer_idx + 1) % sparse_step == 0
    )


def _mlp_flops(tokens: int, layer_idx: int, config: Any) -> int:
    """Return forward FLOPs of one gated MLP: top-k routed experts or a dense MLP."""
    hidden = config.hidden_size
    if _is_moe_layer(layer_idx, config):
        # Each token visits top-k experts; each expert runs gate, up and down GEMMs.
        return config.num_experts_per_tok * 3 * 2 * tokens * hidden * config.moe_intermediate_size
    return 3 * 2 * tokens * hidden * config.intermediate_size


def estimate_train_flops(config: Any, global_batch_size: int, seq_len: int) -> int:
    """Estimate FLOPs of one global optimizer step from the model configuration.

    Args:
        config: Hugging Face style model config exposing Qwen3-MoE fields.
        global_batch_size: Samples consumed by one optimizer step across all ranks.
        seq_len: Tokens per sample.

    Returns:
        Estimated forward-plus-backward FLOPs for the whole global step.

    Raises:
        ValueError: If the batch size or sequence length is not positive.
    """
    if global_batch_size <= 0 or seq_len <= 0:
        raise ValueError(
            "global_batch_size and seq_len must be positive, "
            f"got global_batch_size={global_batch_size}, seq_len={seq_len}"
        )
    tokens = global_batch_size * seq_len
    forward = sum(
        _attention_flops(tokens, seq_len, config) + _mlp_flops(tokens, layer_idx, config)
        for layer_idx in range(config.num_hidden_layers)
    )
    forward += 2 * tokens * config.hidden_size * config.vocab_size
    return 3 * forward
