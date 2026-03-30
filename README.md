# Kondo Gate

Selective backward-pass gating for policy gradient training. A standalone PyTorch implementation compatible with HuggingFace Transformers.

Based on [arXiv:2603.20526](https://arxiv.org/abs/2603.20526) — *Delightful Policy Gradients with Kondo Gating*.

## What it does

The Kondo gate computes **delight** for each training sample — the product of advantage and surprisal — then skips backward passes for low-value samples. This preserves learning quality while dramatically reducing compute.

| Method | Weights gradient by | Backward passes per batch |
|--------|---------------------|---------------------------|
| **PG** | Advantage only | All B |
| **DG** | sigmoid(delight) | All B |
| **DG-K** | Top-k by delight | ~rho x B |

At 3% gate rate, that means ~3 backward passes out of 100 — and it still matches or beats full DG.

## Install

```bash
pip install kondo-gate

# From source with dev dependencies:
pip install -e ".[dev]"
```

## Quick start

### High-level: with HuggingFace model logits

```python
from kondo_gate import KondoGate, KondoGateConfig

gate = KondoGate(KondoGateConfig(gate_rate=0.03))  # keep top 3%

# logits from any model (B, T, V), actions (B, T), advantages (B, T)
result = gate(logits=logits, actions=actions, advantages=advantages)
result.gated_policy_loss.backward()
```

### KondoTrainer: drop-in training wrapper

```python
from transformers import AutoModelForCausalLM
from kondo_gate import KondoTrainer

model = AutoModelForCausalLM.from_pretrained("gpt2")
trainer = KondoTrainer(model, gate_rate=0.03, lr=3e-4)

stats = trainer.step(
    input_ids=input_ids,
    actions=target_ids,
    advantages=advantages,
)
# stats = {"loss": ..., "gate_rate": ..., "price": ..., "mean_delight": ...}
```

### Standalone loss functions (PG, DG)

```python
from kondo_gate import pg_loss, dg_loss, expected_confidence_baseline

# Standard REINFORCE
loss = pg_loss(logits, actions, advantages)

# Delightful Gradient (sigmoid-weighted, all backward passes)
loss, gate_weights = dg_loss(logits, actions, advantages, eta=1.0)

# Expected confidence baseline (used in reference implementation)
baseline = expected_confidence_baseline(probs)  # b = sum pi(a)^2
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gate_rate` | `0.3` | Target fraction of backward passes to keep (rho). Mutually exclusive with `price`. |
| `price` | `None` | Fixed compute price threshold (lambda). Mutually exclusive with `gate_rate`. |
| `temperature` | `0.1` | Gate softness (eta). Used in stochastic/soft modes. |
| `hard` | `True` | Binary gating (True) vs soft sigmoid weights (False). |
| `deterministic` | `True` | Deterministic top-k selection (True, reference impl) vs Bernoulli sampling (False, Algorithm 1). Only applies when `hard=True`. |

### Three gating modes

1. **Deterministic top-k** (`hard=True, deterministic=True`, default) — Matches the [reference Colab implementation](https://colab.research.google.com/drive/1aZ4Zq-PbpczDYJ_8gOQ65_de2M6pklww). Keeps the top rho fraction of samples ranked by delight. Binary, no randomness.

2. **Stochastic Bernoulli** (`hard=True, deterministic=False`) — Matches Algorithm 1 in the paper. Samples G ~ Bernoulli(sigma((chi - lambda) / eta)).

3. **Soft sigmoid** (`hard=False`) — Weights each sample by sigma((chi - lambda) / eta). All backward passes computed, gradient weighted by gate probability.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

60 tests across 10 categories:
- Config validation (bounds, mutual exclusivity, defaults)
- Delight computation (formula correctness, detachment, edge cases)
- Gate mechanism (output shapes, hard/soft modes, adaptive rate targeting)
- Full forward pass (2D/3D logits, attention masking, loss finiteness)
- Mathematical properties (sigmoid formula, temperature limits, price monotonicity)
- Gradient verification (flow through hard/soft gates, zero-grad for gated-out samples)
- Integration (multi-step training loops, parameter updates)
- Edge cases (batch=1, zero advantages, empty masks, reproducibility)
- Deterministic mode (top-k selection, reference impl match, reproducibility)
- Loss functions (PG, DG, DG-K structure, baseline computation)

## Examples

### MNIST contextual bandit (PG vs DG vs DG-K)

Replicates the paper's MNIST experiment. Requires `torchvision`.

```bash
pip install torchvision
python examples/mnist_bandit.py
```

### Token reversal

Trains a small causal transformer to reverse sequences at different gate rates.

```bash
python examples/token_reversal.py
```

## How it works

1. **Forward pass:** Compute log-probabilities for taken actions, then `delight = advantage x surprisal`
2. **Gate decision:** Set price as the (1-rho)-quantile of delight; keep samples with delight >= price
3. **Gated backward:** `loss = -mean(log_pi * stop_grad(gate * advantage))` — only gated-in samples contribute gradients

The gate filters out gradient noise from uninformative samples (low surprisal) and unreliable samples (low advantage magnitude), keeping only the samples that teach the most per unit of compute.

**Why delight, not something simpler?** Neither advantage nor surprisal alone tells the right story. High advantage with low surprisal = the model already knew. High surprisal with zero advantage = unusual but unremarkable. The multiplicative product targets the intersection: something surprising *and* valuable. Unlike additive combinations, the product is sign-consistent across all problem parameters (Proposition 2 in the paper).

## Citation

```bibtex
@article{kondogate2026,
  title={Delightful Policy Gradients with Kondo Gating},
  year={2026},
  eprint={2603.20526},
  archivePrefix={arXiv},
}
```

## License

MIT
