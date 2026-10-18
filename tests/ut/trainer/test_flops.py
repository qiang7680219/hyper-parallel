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
"""Unit tests for the configuration-based training FLOPs estimate."""

from types import SimpleNamespace
import unittest

from hyper_parallel.trainer.config.training import TrainingConfig
from hyper_parallel.trainer.runtime.flops import estimate_train_flops
from tests.common.mark_utils import arg_mark


def qwen3_30b_a3b_config(**overrides: object) -> SimpleNamespace:
    """Return the FLOPs-relevant fields of the public Qwen3-30B-A3B config."""
    fields = {
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "vocab_size": 151936,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _reference_flops(config: SimpleNamespace, batch: int, seq: int, moe_layers: int) -> int:
    """Spell out every GEMM of one global step independently of the implementation."""
    tokens = batch * seq
    hidden = config.hidden_size
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    attention = (
        2 * tokens * hidden * q_dim            # Q projection
        + 2 * tokens * hidden * kv_dim         # K projection
        + 2 * tokens * hidden * kv_dim         # V projection
        + 2 * tokens * seq * q_dim // 2        # causal QK^T
        + 2 * tokens * seq * q_dim // 2        # causal PV
        + 2 * tokens * q_dim * hidden          # output projection
    )
    expert = 3 * 2 * tokens * hidden * config.moe_intermediate_size
    dense = 3 * 2 * tokens * hidden * config.intermediate_size
    dense_layers = config.num_hidden_layers - moe_layers
    forward = (
        config.num_hidden_layers * attention
        + moe_layers * config.num_experts_per_tok * expert
        + dense_layers * dense
        + 2 * tokens * hidden * config.vocab_size
    )
    return 3 * forward


class TestEstimateTrainFlops(unittest.TestCase):
    """Verify the per-step FLOPs estimate against hand-derived GEMM sums."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_cropped_qwen3_moe_matches_hand_calculation(self) -> None:
        """Match the exact FLOPs of the four-layer eight-card example.

        Feature: TFLOP/s estimate.
        Description: Estimate the cropped Qwen3-30B-A3B used by run_parallel_offline.sh.
        Expectation: The value equals the hand-calculated GEMM total.
        """
        config = qwen3_30b_a3b_config(num_hidden_layers=4)
        flops = estimate_train_flops(config, global_batch_size=8, seq_len=128)
        self.assertEqual(flops, 3_316_251_623_424, f"unexpected cropped FLOPs, got={flops}")
        self.assertEqual(flops, _reference_flops(config, 8, 128, moe_layers=4))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_full_qwen3_moe_counts_topk_experts_only(self) -> None:
        """Count top-k experts rather than all experts for the full model.

        Feature: TFLOP/s estimate.
        Description: Estimate full Qwen3-30B-A3B and compare with doubling num_experts.
        Expectation: The value matches the reference and ignores the total expert count.
        """
        config = qwen3_30b_a3b_config()
        flops = estimate_train_flops(config, global_batch_size=8, seq_len=4096)
        self.assertEqual(flops, _reference_flops(config, 8, 4096, moe_layers=48))
        self.assertEqual(
            flops,
            estimate_train_flops(qwen3_30b_a3b_config(num_experts=256), 8, 4096),
            "the total expert count must not change the activated-expert FLOPs",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_dense_layers_follow_qwen3_moe_layer_selection(self) -> None:
        """Use dense MLP FLOPs for mlp_only_layers and non-sparse steps.

        Feature: TFLOP/s estimate.
        Description: Mark layers dense through mlp_only_layers and decoder_sparse_step.
        Expectation: Dense layers use intermediate_size instead of top-k experts.
        """
        only_dense = qwen3_30b_a3b_config(num_hidden_layers=4, mlp_only_layers=[0, 2])
        self.assertEqual(
            estimate_train_flops(only_dense, 2, 64),
            _reference_flops(only_dense, 2, 64, moe_layers=2),
        )
        sparse_step = qwen3_30b_a3b_config(num_hidden_layers=4, decoder_sparse_step=2)
        self.assertEqual(
            estimate_train_flops(sparse_step, 2, 64),
            _reference_flops(sparse_step, 2, 64, moe_layers=2),
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_batch_is_linear_and_attention_is_quadratic_in_sequence(self) -> None:
        """Scale linearly with batch while attention adds a sequence-squared term.

        Feature: TFLOP/s estimate.
        Description: Compare estimates across batch sizes and sequence lengths.
        Expectation: Batch doubles FLOPs exactly; sequence doubling adds only the causal core term.
        """
        config = qwen3_30b_a3b_config(num_hidden_layers=2)
        base = estimate_train_flops(config, 1, 256)
        self.assertEqual(estimate_train_flops(config, 2, 256), 2 * base)
        q_dim = config.num_attention_heads * config.head_dim
        # Doubling S doubles linear terms and quadruples the causal core 2*S*S*q_dim per layer.
        core = 3 * config.num_hidden_layers * 2 * 256 * 256 * q_dim
        self.assertEqual(estimate_train_flops(config, 1, 512), 2 * base + 2 * core)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_invalid_shape_and_override_are_rejected(self) -> None:
        """Reject non-positive batch, sequence and override values.

        Feature: TFLOP/s estimate.
        Description: Pass zero shape arguments and a zero tflops_seq_length.
        Expectation: ValueError is raised for each invalid input.
        """
        config = qwen3_30b_a3b_config(num_hidden_layers=1)
        with self.assertRaises(ValueError):
            estimate_train_flops(config, 0, 128)
        with self.assertRaises(ValueError):
            estimate_train_flops(config, 8, 0)
        with self.assertRaises(ValueError):
            TrainingConfig(tflops_seq_length=0)


if __name__ == "__main__":
    unittest.main()
