"""
Kondo Gate: Selective Backward-Pass Gating for Policy Gradient Training

Standalone PyTorch + HuggingFace Transformers compatible implementation.

Based on arXiv:2603.20526 — "Delightful Policy Gradients with Kondo Gating"

The Kondo gate computes "delight" (advantage × surprisal) for each sample,
then stochastically gates backward passes so only high-value samples
contribute gradients. This preserves learning quality while skipping
most backward passes.

Core formula:
    delight  χ = U · ℓ           (advantage × surprisal)
    gate weight  w* = σ((χ - λ) / η)   (sigmoid soft threshold)
    gate sample  G ~ Bernoulli(w*)
    gradient     Δθ += G · U · ∇θ log πθ(a|s)
"""

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KondoGateConfig:
    """Configuration for the Kondo gate.

    Args:
        gate_rate: Target fraction of backward passes to keep (ρ ∈ (0, 1]).
            When set, λ is adaptively computed as the (1-ρ)-quantile of delight.
            Mutually exclusive with `price`.
        price: Fixed compute price threshold (λ ≥ 0). Samples with delight
            below this are likely gated out. Mutually exclusive with `gate_rate`.
        temperature: Softness of the gate (η > 0). Lower → harder threshold.
            As η → 0, becomes indicator 𝕀{χ > λ}.
            As η → ∞, recovers standard (ungated) policy gradient.
        hard: If True, use hard Bernoulli sampling (straight-through in forward).
            If False, use soft (differentiable) gating weights.
    """
    gate_rate: Optional[float] = 0.3
    price: Optional[float] = None
    temperature: float = 0.1
    hard: bool = True


class KondoGate(nn.Module):
    """Kondo gate module for selective backward-pass gating.

    Computes delight = advantage × surprisal for each sample, then
    produces a stochastic binary gate that determines whether to
    include each sample's gradient in the update.

    Compatible with any model that produces log-probabilities and
    can be used as a drop-in training utility with HuggingFace
    Transformers models.

    Example — standalone REINFORCE with Kondo gating::

        gate = KondoGate(KondoGateConfig(gate_rate=0.1, temperature=0.05))
        logits = model(input_ids).logits
        result = gate(logits=logits, actions=target_ids, advantages=advantages)
        loss = result.gated_policy_loss
        loss.backward()

    Example — gating an existing per-token loss::

        gate = KondoGate(KondoGateConfig(gate_rate=0.1))
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        result = gate.compute_gate(
            log_probs=action_log_probs,
            advantages=advantages,
        )
        gated_loss = -(result.gate_weights * advantages * action_log_probs).mean()
    """

    def __init__(self, config: Optional[KondoGateConfig] = None):
        super().__init__()
        self.config = config or KondoGateConfig()
        if self.config.gate_rate is not None and self.config.price is not None:
            raise ValueError("Specify either gate_rate or price, not both.")
        if self.config.gate_rate is not None:
            if not 0.0 < self.config.gate_rate <= 1.0:
                raise ValueError("gate_rate must be in (0, 1].")
        if self.config.temperature <= 0:
            raise ValueError("temperature must be > 0.")

    @torch.no_grad()
    def _compute_price(self, delight: torch.Tensor) -> torch.Tensor:
        """Adaptively set price λ as the (1-ρ)-quantile of delight."""
        if self.config.gate_rate is not None:
            q = 1.0 - self.config.gate_rate
            return torch.quantile(delight.float().detach(), q)
        return torch.tensor(self.config.price, device=delight.device, dtype=delight.dtype)

    def compute_delight(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute delight χ = U · ℓ = advantage × surprisal.

        Args:
            log_probs: Log-probabilities of the taken actions, shape (B,) or (B, T).
            advantages: Advantage estimates, same shape as log_probs.

        Returns:
            Delight values, same shape as input.
        """
        surprisal = -log_probs.detach()
        return advantages.detach() * surprisal

    def compute_gate(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        delight: Optional[torch.Tensor] = None,
    ) -> "KondoGateOutput":
        """Compute gate decisions for a batch.

        Args:
            log_probs: Log-probabilities of taken actions, shape (B,) or (B, T).
            advantages: Advantage estimates, same shape.
            delight: Pre-computed delight (optional). Computed if not given.

        Returns:
            KondoGateOutput with gate weights, delight, price, and stats.
        """
        if delight is None:
            delight = self.compute_delight(log_probs, advantages)

        # Flatten for quantile computation if multi-dimensional
        flat_delight = delight.reshape(-1)
        price = self._compute_price(flat_delight)

        # Gate probability: w* = σ((χ - λ) / η)
        gate_logits = (delight - price) / self.config.temperature
        gate_probs = torch.sigmoid(gate_logits)

        if self.config.hard:
            # Bernoulli sample with straight-through gradient
            gate_samples = torch.bernoulli(gate_probs.detach())
            # Straight-through: forward uses hard samples, backward uses soft probs
            gate_weights = gate_samples + gate_probs - gate_probs.detach()
        else:
            gate_weights = gate_probs

        # Stats
        with torch.no_grad():
            actual_rate = gate_samples.mean() if self.config.hard else gate_probs.mean()

        return KondoGateOutput(
            gate_weights=gate_weights,
            gate_probs=gate_probs,
            delight=delight,
            price=price,
            actual_gate_rate=actual_rate,
        )

    def forward(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> "KondoGateOutput":
        """Full forward pass: compute log-probs, delight, gate, and gated loss.

        This is the main entry point for use with HuggingFace model outputs.

        Args:
            logits: Model output logits, shape (B, T, V) or (B, V).
            actions: Action/token indices taken, shape (B, T) or (B,).
            advantages: Per-sample or per-token advantages, shape (B, T) or (B,).
            attention_mask: Optional mask, shape (B, T). 1 = keep, 0 = ignore.

        Returns:
            KondoGateOutput including gated_policy_loss ready for .backward().
        """
        # Compute log-probabilities of taken actions
        if logits.dim() == 3:
            # (B, T, V) → per-token log-probs
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        elif logits.dim() == 2:
            # (B, V) → per-sample log-probs
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError(f"Expected logits of dim 2 or 3, got {logits.dim()}")

        # Apply attention mask to get per-sample delight
        if attention_mask is not None and action_log_probs.dim() == 2:
            # Average over valid tokens for per-sample values
            mask = attention_mask.float()
            masked_log_probs = (action_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
            masked_advantages = (advantages * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
            delight = self.compute_delight(masked_log_probs, masked_advantages)

            # Compute gate at sample level (B,)
            result = self.compute_gate(masked_log_probs, masked_advantages, delight=delight)

            # Expand gate back to token level for loss
            gate_weights_expanded = result.gate_weights.unsqueeze(-1)  # (B, 1)
            gated_loss = -(gate_weights_expanded * advantages * action_log_probs * mask).sum() / mask.sum().clamp(min=1)
        elif action_log_probs.dim() == 2:
            # No mask, per-token gating
            delight = self.compute_delight(action_log_probs, advantages)
            result = self.compute_gate(action_log_probs, advantages, delight=delight)
            gated_loss = -(result.gate_weights * advantages * action_log_probs).mean()
        else:
            # Per-sample (1D)
            delight = self.compute_delight(action_log_probs, advantages)
            result = self.compute_gate(action_log_probs, advantages, delight=delight)
            gated_loss = -(result.gate_weights * advantages * action_log_probs).mean()

        result.gated_policy_loss = gated_loss
        result.action_log_probs = action_log_probs
        return result


@dataclass
class KondoGateOutput:
    """Output from the Kondo gate.

    Attributes:
        gate_weights: Per-sample (or per-token) gate values used in the
            gradient computation. Shape matches the input log_probs.
        gate_probs: Soft gate probabilities σ((χ - λ)/η) before sampling.
        delight: Computed delight values χ = U · ℓ.
        price: The compute price λ used (adaptive or fixed).
        actual_gate_rate: Fraction of samples that passed the gate.
        gated_policy_loss: Scalar loss ready for .backward() (only from forward()).
        action_log_probs: Log-probabilities of taken actions (only from forward()).
    """
    gate_weights: torch.Tensor
    gate_probs: torch.Tensor
    delight: torch.Tensor
    price: torch.Tensor
    actual_gate_rate: torch.Tensor
    gated_policy_loss: Optional[torch.Tensor] = None
    action_log_probs: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Convenience: KondoTrainer wrapping a HuggingFace model
# ---------------------------------------------------------------------------

class KondoTrainer:
    """Lightweight training wrapper that applies Kondo gating to any
    HuggingFace causal LM for REINFORCE-style training.

    Example::

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        trainer = KondoTrainer(model, gate_rate=0.1, lr=3e-4)

        # Training loop
        for batch in dataloader:
            stats = trainer.step(
                input_ids=batch["input_ids"],
                actions=batch["target_ids"],
                advantages=batch["advantages"],
                attention_mask=batch["attention_mask"],
            )
            print(f"loss={stats['loss']:.4f}  gate_rate={stats['gate_rate']:.2%}")
    """

    def __init__(
        self,
        model: nn.Module,
        gate_rate: float = 0.3,
        temperature: float = 0.1,
        price: Optional[float] = None,
        hard: bool = True,
        lr: float = 3e-4,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.gate = KondoGate(KondoGateConfig(
            gate_rate=gate_rate if price is None else None,
            price=price,
            temperature=temperature,
            hard=hard,
        ))
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)

    def step(
        self,
        input_ids: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run one training step with Kondo gating.

        Returns dict with loss, gate_rate, price, and mean_delight.
        """
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        result = self.gate(
            logits=logits,
            actions=actions,
            advantages=advantages,
            attention_mask=attention_mask,
        )

        result.gated_policy_loss.backward()
        self.optimizer.step()

        return {
            "loss": result.gated_policy_loss.item(),
            "gate_rate": result.actual_gate_rate.item(),
            "price": result.price.item(),
            "mean_delight": result.delight.mean().item(),
        }


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, V = 8, 16, 100  # batch, seq_len, vocab

    # Simulate model outputs
    logits = torch.randn(B, T, V, requires_grad=True)
    actions = torch.randint(0, V, (B, T))
    advantages = torch.randn(B, T)
    mask = torch.ones(B, T)
    mask[:, -3:] = 0  # mask out last 3 tokens

    print("=== Kondo Gate Demo ===\n")

    for rate in [1.0, 0.5, 0.1, 0.03]:
        gate = KondoGate(KondoGateConfig(gate_rate=rate, temperature=0.05))
        result = gate(logits=logits, actions=actions, advantages=advantages, attention_mask=mask)
        print(f"gate_rate={rate:.0%}  actual={result.actual_gate_rate:.2%}  "
              f"loss={result.gated_policy_loss:.4f}  price={result.price:.3f}")

    print("\n=== Low-level API ===\n")
    gate = KondoGate(KondoGateConfig(gate_rate=0.3, temperature=0.1))
    log_probs = torch.randn(B)
    advs = torch.randn(B)
    out = gate.compute_gate(log_probs=log_probs, advantages=advs)
    print(f"gate_weights: {out.gate_weights}")
    print(f"gate_probs:   {out.gate_probs}")
    print(f"actual_rate:  {out.actual_gate_rate:.2%}")
    print(f"price:        {out.price:.3f}")

    print("\nDone.")
