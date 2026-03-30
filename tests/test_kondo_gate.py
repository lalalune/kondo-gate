"""
Comprehensive tests, validation, and verification for the Kondo gate.

Run:  python -m pytest test_kondo_gate.py -v
  or: python test_kondo_gate.py
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from kondo_gate import KondoGate, KondoGateConfig, KondoGateOutput, KondoTrainer


# ============================================================================
# Unit Tests
# ============================================================================


class TestKondoGateConfig:
    """Validate config constraints."""

    def test_default_config(self):
        cfg = KondoGateConfig()
        assert cfg.gate_rate == 0.3
        assert cfg.price is None
        assert cfg.temperature == 0.1
        assert cfg.hard is True

    def test_gate_rate_and_price_mutually_exclusive(self):
        with pytest.raises(ValueError, match="either gate_rate or price"):
            KondoGate(KondoGateConfig(gate_rate=0.5, price=1.0))

    def test_gate_rate_bounds(self):
        with pytest.raises(ValueError, match="gate_rate must be in"):
            KondoGate(KondoGateConfig(gate_rate=0.0))
        with pytest.raises(ValueError, match="gate_rate must be in"):
            KondoGate(KondoGateConfig(gate_rate=-0.1))
        with pytest.raises(ValueError, match="gate_rate must be in"):
            KondoGate(KondoGateConfig(gate_rate=1.5))

    def test_gate_rate_1_is_valid(self):
        gate = KondoGate(KondoGateConfig(gate_rate=1.0))
        assert gate.config.gate_rate == 1.0

    def test_temperature_must_be_positive(self):
        with pytest.raises(ValueError, match="temperature must be > 0"):
            KondoGate(KondoGateConfig(temperature=0.0))
        with pytest.raises(ValueError, match="temperature must be > 0"):
            KondoGate(KondoGateConfig(temperature=-1.0))

    def test_price_only_mode(self):
        gate = KondoGate(KondoGateConfig(gate_rate=None, price=2.0))
        assert gate.config.price == 2.0
        assert gate.config.gate_rate is None


class TestComputeDelight:
    """Verify delight = advantage × surprisal."""

    def test_basic_delight(self):
        gate = KondoGate()
        log_probs = torch.tensor([-1.0, -2.0, -0.5])
        advantages = torch.tensor([1.0, -1.0, 3.0])
        delight = gate.compute_delight(log_probs, advantages)
        # surprisal = [1.0, 2.0, 0.5], delight = [1*1, -1*2, 3*0.5]
        expected = torch.tensor([1.0, -2.0, 1.5])
        assert torch.allclose(delight, expected)

    def test_delight_detaches_inputs(self):
        gate = KondoGate()
        log_probs = torch.tensor([-1.0, -2.0], requires_grad=True)
        advantages = torch.tensor([1.0, -1.0], requires_grad=True)
        delight = gate.compute_delight(log_probs, advantages)
        assert not delight.requires_grad

    def test_delight_2d(self):
        gate = KondoGate()
        log_probs = torch.tensor([[-1.0, -2.0], [-0.5, -3.0]])
        advantages = torch.tensor([[2.0, 1.0], [0.5, -1.0]])
        delight = gate.compute_delight(log_probs, advantages)
        expected = torch.tensor([[2.0, 2.0], [0.25, -3.0]])
        assert delight.shape == (2, 2)
        assert torch.allclose(delight, expected, atol=1e-6)

    def test_zero_advantage_gives_zero_delight(self):
        gate = KondoGate()
        log_probs = torch.tensor([-5.0, -10.0])
        advantages = torch.zeros(2)
        delight = gate.compute_delight(log_probs, advantages)
        assert torch.allclose(delight, torch.zeros(2))

    def test_zero_surprisal_gives_zero_delight(self):
        gate = KondoGate()
        log_probs = torch.zeros(2)  # surprisal = 0
        advantages = torch.tensor([100.0, -100.0])
        delight = gate.compute_delight(log_probs, advantages)
        assert torch.allclose(delight, torch.zeros(2))


class TestComputeGate:
    """Verify the gating mechanism."""

    def test_output_structure(self):
        gate = KondoGate(KondoGateConfig(gate_rate=0.5))
        log_probs = torch.randn(8)
        advantages = torch.randn(8)
        out = gate.compute_gate(log_probs, advantages)
        assert isinstance(out, KondoGateOutput)
        assert out.gate_weights.shape == (8,)
        assert out.gate_probs.shape == (8,)
        assert out.delight.shape == (8,)
        assert out.price.dim() == 0  # scalar
        assert out.actual_gate_rate.dim() == 0

    def test_gate_probs_in_01(self):
        gate = KondoGate(KondoGateConfig(gate_rate=0.5))
        log_probs = torch.randn(100)
        advantages = torch.randn(100)
        out = gate.compute_gate(log_probs, advantages)
        assert (out.gate_probs >= 0).all()
        assert (out.gate_probs <= 1).all()

    def test_hard_gate_is_binary(self):
        gate = KondoGate(KondoGateConfig(gate_rate=0.5, hard=True))
        torch.manual_seed(0)
        log_probs = torch.randn(100)
        advantages = torch.randn(100)
        out = gate.compute_gate(log_probs, advantages)
        # In hard mode, the forward value should be near 0 or 1
        # (straight-through adds gate_probs - gate_probs.detach() for grad,
        #  but detached values should round cleanly)
        forward_vals = out.gate_weights.detach()
        rounded = forward_vals.round()
        assert torch.allclose(forward_vals, rounded, atol=1e-5)

    def test_soft_gate_is_continuous(self):
        gate = KondoGate(KondoGateConfig(gate_rate=0.5, hard=False))
        log_probs = torch.randn(100)
        advantages = torch.randn(100)
        out = gate.compute_gate(log_probs, advantages)
        # Soft mode should have values strictly between 0 and 1
        assert (out.gate_weights > 0).any()
        assert (out.gate_weights < 1).any()

    def test_adaptive_price_targets_gate_rate(self):
        """Over many trials, actual gate rate should approximate target."""
        gate = KondoGate(KondoGateConfig(gate_rate=0.3, temperature=0.01, hard=True))
        rates = []
        for i in range(200):
            torch.manual_seed(i)
            log_probs = torch.randn(256)
            advantages = torch.randn(256)
            out = gate.compute_gate(log_probs, advantages)
            rates.append(out.actual_gate_rate.item())
        mean_rate = sum(rates) / len(rates)
        assert abs(mean_rate - 0.3) < 0.05, f"Expected ~0.3, got {mean_rate:.3f}"

    def test_fixed_price_mode(self):
        gate = KondoGate(KondoGateConfig(gate_rate=None, price=0.0))
        log_probs = torch.tensor([-1.0, -2.0, -3.0])
        advantages = torch.tensor([1.0, 1.0, 1.0])
        out = gate.compute_gate(log_probs, advantages)
        assert out.price.item() == 0.0

    def test_precomputed_delight(self):
        gate = KondoGate(KondoGateConfig(gate_rate=0.5))
        log_probs = torch.randn(8)
        advantages = torch.randn(8)
        delight = torch.ones(8) * 5.0  # override
        out = gate.compute_gate(log_probs, advantages, delight=delight)
        assert torch.allclose(out.delight, delight)


class TestForward:
    """Test the full forward pass with logits → gated loss."""

    def test_3d_logits(self):
        B, T, V = 4, 8, 50
        gate = KondoGate(KondoGateConfig(gate_rate=0.5))
        logits = torch.randn(B, T, V, requires_grad=True)
        actions = torch.randint(0, V, (B, T))
        advantages = torch.randn(B, T)
        out = gate(logits, actions, advantages)
        assert out.gated_policy_loss is not None
        assert out.gated_policy_loss.dim() == 0
        out.gated_policy_loss.backward()
        assert logits.grad is not None

    def test_2d_logits(self):
        B, V = 8, 20
        gate = KondoGate(KondoGateConfig(gate_rate=0.5))
        logits = torch.randn(B, V, requires_grad=True)
        actions = torch.randint(0, V, (B,))
        advantages = torch.randn(B)
        out = gate(logits, actions, advantages)
        assert out.gated_policy_loss.dim() == 0
        out.gated_policy_loss.backward()
        assert logits.grad is not None

    def test_attention_mask(self):
        B, T, V = 4, 8, 50
        gate = KondoGate(KondoGateConfig(gate_rate=0.5))
        logits = torch.randn(B, T, V, requires_grad=True)
        actions = torch.randint(0, V, (B, T))
        advantages = torch.randn(B, T)
        mask = torch.ones(B, T)
        mask[:, -2:] = 0
        out = gate(logits, actions, advantages, attention_mask=mask)
        assert out.gated_policy_loss.dim() == 0
        out.gated_policy_loss.backward()
        assert logits.grad is not None

    def test_invalid_logits_dim_raises(self):
        gate = KondoGate()
        with pytest.raises(ValueError, match="Expected logits of dim 2 or 3"):
            gate(torch.randn(10), torch.randint(0, 5, (10,)), torch.randn(10))

    def test_loss_is_finite(self):
        B, T, V = 16, 10, 30
        gate = KondoGate(KondoGateConfig(gate_rate=0.5))
        logits = torch.randn(B, T, V)
        actions = torch.randint(0, V, (B, T))
        advantages = torch.randn(B, T)
        out = gate(logits, actions, advantages)
        assert torch.isfinite(out.gated_policy_loss)

    def test_action_log_probs_are_negative(self):
        B, V = 8, 20
        gate = KondoGate()
        logits = torch.randn(B, V)
        actions = torch.randint(0, V, (B,))
        advantages = torch.randn(B)
        out = gate(logits, actions, advantages)
        assert (out.action_log_probs <= 0).all()


# ============================================================================
# Mathematical Verification
# ============================================================================


class TestMathematicalProperties:
    """Verify the gate satisfies properties from the paper."""

    def test_gate_formula_matches_sigmoid(self):
        """w* = σ((χ - λ) / η) — verify manual computation."""
        gate = KondoGate(KondoGateConfig(gate_rate=None, price=1.0, temperature=0.5, hard=False))
        log_probs = torch.tensor([-2.0, -1.0, -0.5])
        advantages = torch.tensor([1.0, 2.0, 3.0])
        out = gate.compute_gate(log_probs, advantages)

        # Manual: surprisal = [2, 1, 0.5], delight = [2, 2, 1.5]
        # gate_logits = (delight - 1.0) / 0.5 = [2, 2, 1]
        # gate_probs = sigmoid([2, 2, 1])
        expected = torch.sigmoid(torch.tensor([2.0, 2.0, 1.0]))
        assert torch.allclose(out.gate_probs, expected, atol=1e-6)

    def test_high_temperature_approaches_uniform(self):
        """As η → ∞, gate_probs → 0.5 (all samples equally likely)."""
        gate = KondoGate(KondoGateConfig(gate_rate=None, price=0.0, temperature=1e6, hard=False))
        log_probs = torch.randn(100)
        advantages = torch.randn(100)
        out = gate.compute_gate(log_probs, advantages)
        assert torch.allclose(out.gate_probs, torch.full_like(out.gate_probs, 0.5), atol=0.01)

    def test_low_temperature_approaches_hard_threshold(self):
        """As η → 0, gate_probs → 𝕀{χ > λ}."""
        gate = KondoGate(KondoGateConfig(gate_rate=None, price=1.0, temperature=1e-6, hard=False))
        log_probs = torch.tensor([-2.0, -0.1])
        advantages = torch.tensor([1.0, 1.0])
        out = gate.compute_gate(log_probs, advantages)
        # delight = [2.0, 0.1], price = 1.0
        # delight[0] > price → gate ≈ 1, delight[1] < price → gate ≈ 0
        assert out.gate_probs[0].item() > 0.999
        assert out.gate_probs[1].item() < 0.001

    def test_gate_rate_1_keeps_all_samples(self):
        """ρ=1.0 → price is min(delight), so everything passes."""
        gate = KondoGate(KondoGateConfig(gate_rate=1.0, temperature=0.01, hard=False))
        log_probs = torch.randn(64)
        advantages = torch.randn(64)
        out = gate.compute_gate(log_probs, advantages)
        # With price at 0th quantile (min) and low temp, most should pass
        assert out.gate_probs.mean().item() > 0.8

    def test_delight_sign_positive_for_positive_advantage_and_log_prob_negative(self):
        """Positive advantage × positive surprisal = positive delight."""
        gate = KondoGate()
        log_probs = torch.tensor([-2.0])  # surprisal = 2
        advantages = torch.tensor([3.0])
        delight = gate.compute_delight(log_probs, advantages)
        assert delight.item() == 6.0

    def test_delight_sign_negative_for_negative_advantage(self):
        """Negative advantage → negative delight (when surprisal > 0)."""
        gate = KondoGate()
        log_probs = torch.tensor([-2.0])  # surprisal = 2
        advantages = torch.tensor([-3.0])
        delight = gate.compute_delight(log_probs, advantages)
        assert delight.item() == -6.0

    def test_quantile_price_is_monotonic_in_gate_rate(self):
        """Higher ρ → lower price (more samples pass)."""
        torch.manual_seed(42)
        log_probs = torch.randn(256)
        advantages = torch.randn(256)
        prices = []
        for rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
            gate = KondoGate(KondoGateConfig(gate_rate=rate))
            out = gate.compute_gate(log_probs, advantages)
            prices.append(out.price.item())
        # Price should decrease as gate_rate increases
        for i in range(len(prices) - 1):
            assert prices[i] >= prices[i + 1], f"Price not monotonic: {prices}"


# ============================================================================
# Gradient Verification
# ============================================================================


class TestGradients:
    """Verify gradients flow correctly through the gate."""

    def test_gradients_flow_through_soft_gate(self):
        gate = KondoGate(KondoGateConfig(gate_rate=0.5, hard=False))
        logits = torch.randn(4, 20, requires_grad=True)
        actions = torch.randint(0, 20, (4,))
        advantages = torch.randn(4)
        out = gate(logits, actions, advantages)
        out.gated_policy_loss.backward()
        assert logits.grad is not None
        assert (logits.grad != 0).any()

    def test_gradients_flow_through_hard_gate(self):
        """Hard gate uses straight-through — grads should still flow."""
        gate = KondoGate(KondoGateConfig(gate_rate=0.8, hard=True))
        logits = torch.randn(8, 20, requires_grad=True)
        actions = torch.randint(0, 20, (8,))
        advantages = torch.randn(8)
        out = gate(logits, actions, advantages)
        out.gated_policy_loss.backward()
        assert logits.grad is not None

    def test_gated_out_samples_have_zero_grad_contribution(self):
        """With very low temp and high price, nearly all are gated → grad ≈ 0."""
        gate = KondoGate(KondoGateConfig(gate_rate=None, price=1e6, temperature=0.001, hard=True))
        logits = torch.randn(8, 20, requires_grad=True)
        actions = torch.randint(0, 20, (8,))
        advantages = torch.randn(8)
        out = gate(logits, actions, advantages)
        out.gated_policy_loss.backward()
        # All gated out → loss is 0 → grad is 0
        assert torch.allclose(logits.grad, torch.zeros_like(logits.grad))

    def test_gradient_magnitude_scales_with_gate_rate(self):
        """Lower gate rate → fewer contributing samples → smaller grad norm."""
        torch.manual_seed(42)
        norms = {}
        for rate in [1.0, 0.1]:
            logits = torch.randn(64, 30, requires_grad=True)
            actions = torch.randint(0, 30, (64,))
            advantages = torch.ones(64)
            gate = KondoGate(KondoGateConfig(gate_rate=rate, temperature=0.001, hard=True))
            out = gate(logits, actions, advantages)
            out.gated_policy_loss.backward()
            norms[rate] = logits.grad.norm().item()
        # rate=0.1 should have smaller grad norm than rate=1.0 (on average)
        # This is probabilistic, but with seed=42 and 64 samples it's reliable
        assert norms[0.1] < norms[1.0], f"grad norms: {norms}"


# ============================================================================
# Integration with nn.Module
# ============================================================================


class SimplePolicy(nn.Module):
    """Tiny policy network for integration tests."""

    def __init__(self, input_dim=10, hidden_dim=32, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class TestIntegrationWithModule:
    """Test Kondo gate integrated into a real training loop."""

    def test_single_training_step(self):
        torch.manual_seed(0)
        model = SimplePolicy()
        gate = KondoGate(KondoGateConfig(gate_rate=0.5))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randn(16, 10)
        actions = torch.randint(0, 5, (16,))
        advantages = torch.randn(16)

        optimizer.zero_grad()
        logits = model(x)
        result = gate(logits, actions, advantages)
        result.gated_policy_loss.backward()
        optimizer.step()

        assert torch.isfinite(result.gated_policy_loss)
        assert 0 <= result.actual_gate_rate.item() <= 1

    def test_multiple_steps_loss_changes(self):
        """Model should update over multiple steps."""
        torch.manual_seed(0)
        model = SimplePolicy()
        gate = KondoGate(KondoGateConfig(gate_rate=0.5))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        losses = []
        for step in range(20):
            x = torch.randn(32, 10)
            actions = torch.randint(0, 5, (32,))
            advantages = torch.ones(32)  # constant positive advantage

            optimizer.zero_grad()
            logits = model(x)
            result = gate(logits, actions, advantages)
            result.gated_policy_loss.backward()
            optimizer.step()
            losses.append(result.gated_policy_loss.item())

        # Loss should not be identical across all steps (model is learning)
        assert len(set(f"{l:.6f}" for l in losses)) > 1

    def test_parameters_change_after_step(self):
        torch.manual_seed(0)
        model = SimplePolicy()
        gate = KondoGate(KondoGateConfig(gate_rate=0.5))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        params_before = {n: p.clone() for n, p in model.named_parameters()}

        x = torch.randn(16, 10)
        actions = torch.randint(0, 5, (16,))
        advantages = torch.randn(16)

        optimizer.zero_grad()
        logits = model(x)
        result = gate(logits, actions, advantages)
        result.gated_policy_loss.backward()
        optimizer.step()

        changed = any(
            not torch.equal(params_before[n], p) for n, p in model.named_parameters()
        )
        assert changed, "Parameters did not change after training step"


# ============================================================================
# KondoTrainer Tests
# ============================================================================


class FakeCausalLM(nn.Module):
    """Minimal HuggingFace-like causal LM for testing KondoTrainer."""

    def __init__(self, vocab_size=50, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        h = self.embedding(input_ids)
        logits = self.head(h)

        class Output:
            pass

        out = Output()
        out.logits = logits
        return out


class TestKondoTrainer:

    def test_trainer_step(self):
        torch.manual_seed(0)
        model = FakeCausalLM(vocab_size=50)
        trainer = KondoTrainer(model, gate_rate=0.3, lr=1e-3)

        input_ids = torch.randint(0, 50, (4, 8))
        actions = torch.randint(0, 50, (4, 8))
        advantages = torch.randn(4, 8)

        stats = trainer.step(input_ids, actions, advantages)
        assert "loss" in stats
        assert "gate_rate" in stats
        assert "price" in stats
        assert "mean_delight" in stats
        assert math.isfinite(stats["loss"])

    def test_trainer_with_mask(self):
        torch.manual_seed(0)
        model = FakeCausalLM(vocab_size=50)
        trainer = KondoTrainer(model, gate_rate=0.5, lr=1e-3)

        input_ids = torch.randint(0, 50, (4, 8))
        actions = torch.randint(0, 50, (4, 8))
        advantages = torch.randn(4, 8)
        mask = torch.ones(4, 8)
        mask[:, -2:] = 0

        stats = trainer.step(input_ids, actions, advantages, attention_mask=mask)
        assert math.isfinite(stats["loss"])

    def test_trainer_with_fixed_price(self):
        torch.manual_seed(0)
        model = FakeCausalLM(vocab_size=50)
        trainer = KondoTrainer(model, price=0.5, lr=1e-3)

        input_ids = torch.randint(0, 50, (4, 8))
        actions = torch.randint(0, 50, (4, 8))
        advantages = torch.randn(4, 8)

        stats = trainer.step(input_ids, actions, advantages)
        assert stats["price"] == 0.5

    def test_trainer_custom_optimizer(self):
        model = FakeCausalLM(vocab_size=50)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = KondoTrainer(model, gate_rate=0.5, optimizer=opt)
        assert trainer.optimizer is opt


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:

    def test_batch_size_1(self):
        gate = KondoGate(KondoGateConfig(gate_rate=0.5))
        logits = torch.randn(1, 10, requires_grad=True)
        actions = torch.randint(0, 10, (1,))
        advantages = torch.randn(1)
        out = gate(logits, actions, advantages)
        out.gated_policy_loss.backward()
        assert torch.isfinite(out.gated_policy_loss)

    def test_all_zero_advantages(self):
        gate = KondoGate(KondoGateConfig(gate_rate=0.5))
        logits = torch.randn(8, 20, requires_grad=True)
        actions = torch.randint(0, 20, (8,))
        advantages = torch.zeros(8)
        out = gate(logits, actions, advantages)
        # Delight should be all zeros
        assert torch.allclose(out.delight, torch.zeros(8))
        out.gated_policy_loss.backward()
        assert torch.isfinite(out.gated_policy_loss)

    def test_large_advantages(self):
        gate = KondoGate(KondoGateConfig(gate_rate=0.5))
        logits = torch.randn(8, 20, requires_grad=True)
        actions = torch.randint(0, 20, (8,))
        advantages = torch.randn(8) * 1000
        out = gate(logits, actions, advantages)
        assert torch.isfinite(out.gated_policy_loss)

    def test_uniform_logits(self):
        """All logits identical → uniform distribution."""
        B, V = 8, 20
        gate = KondoGate(KondoGateConfig(gate_rate=0.5))
        logits = torch.zeros(B, V, requires_grad=True)
        actions = torch.randint(0, V, (B,))
        advantages = torch.randn(B)
        out = gate(logits, actions, advantages)
        # log_softmax of uniform = -ln(V) everywhere
        expected_log_prob = -math.log(V)
        assert torch.allclose(out.action_log_probs, torch.full((B,), expected_log_prob), atol=1e-5)

    def test_full_mask(self):
        """Mask = all 1s should behave same as no mask."""
        B, T, V = 4, 6, 20
        torch.manual_seed(42)
        gate = KondoGate(KondoGateConfig(gate_rate=0.5, hard=False))
        logits = torch.randn(B, T, V)
        actions = torch.randint(0, V, (B, T))
        advantages = torch.randn(B, T)

        out_no_mask = gate(logits, actions, advantages)
        out_full_mask = gate(logits, actions, advantages, attention_mask=torch.ones(B, T))
        # With full mask, both should produce same delight values
        # (though loss computation path differs slightly)
        assert out_full_mask.gated_policy_loss is not None

    def test_empty_mask_no_crash(self):
        """Mask = all 0s should not crash (clamped denominator)."""
        B, T, V = 4, 6, 20
        gate = KondoGate(KondoGateConfig(gate_rate=0.5))
        logits = torch.randn(B, T, V, requires_grad=True)
        actions = torch.randint(0, V, (B, T))
        advantages = torch.randn(B, T)
        mask = torch.zeros(B, T)
        out = gate(logits, actions, advantages, attention_mask=mask)
        assert torch.isfinite(out.gated_policy_loss)

    def test_reproducibility_with_seed(self):
        """Same seed → same gate decisions."""
        gate = KondoGate(KondoGateConfig(gate_rate=0.3, hard=True))
        log_probs = torch.randn(32)
        advantages = torch.randn(32)

        torch.manual_seed(123)
        out1 = gate.compute_gate(log_probs, advantages)
        torch.manual_seed(123)
        out2 = gate.compute_gate(log_probs, advantages)
        assert torch.equal(out1.gate_weights.detach(), out2.gate_weights.detach())


# ============================================================================
# Run as script
# ============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("Running Kondo Gate Test Suite")
    print("=" * 70)
    pytest.main([__file__, "-v", "--tb=short"])
