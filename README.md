# Kondo Gate

Selective backward-pass gating for policy gradient training. A standalone PyTorch implementation compatible with HuggingFace Transformers.

Based on [arXiv:2603.20526](https://arxiv.org/abs/2603.20526) — *Delightful Policy Gradients with Kondo Gating*.

## What it does

The Kondo gate computes **delight** for each training sample — the product of advantage and surprisal — then stochastically skips backward passes for low-value samples. This preserves learning quality while dramatically reducing compute.

```
delight    chi = advantage * surprisal
gate prob  w*  = sigmoid((chi - price) / temperature)
gate       G   ~ Bernoulli(w*)
gradient   only computed when G = 1
```

At a 10% gate rate (skipping 90% of backward passes), the gate retains nearly all learning quality. At 3%, it still learns effectively with two orders of magnitude fewer backward passes.

## Install

```bash
pip install -e .

# With HuggingFace Transformers support:
pip install -e ".[transformers]"

# With dev/test dependencies:
pip install -e ".[dev]"
```

## Quick start

### High-level: with HuggingFace model logits

```python
from kondo_gate import KondoGate, KondoGateConfig

gate = KondoGate(KondoGateConfig(gate_rate=0.1, temperature=0.05))

# logits from any model (B, T, V), actions (B, T), advantages (B, T)
result = gate(logits=logits, actions=actions, advantages=advantages)
result.gated_policy_loss.backward()
```

### KondoTrainer: drop-in training wrapper

```python
from transformers import AutoModelForCausalLM
from kondo_gate import KondoTrainer

model = AutoModelForCausalLM.from_pretrained("gpt2")
trainer = KondoTrainer(model, gate_rate=0.1, lr=3e-4)

stats = trainer.step(
    input_ids=input_ids,
    actions=target_ids,
    advantages=advantages,
    attention_mask=attention_mask,
)
# stats = {"loss": ..., "gate_rate": ..., "price": ..., "mean_delight": ...}
```

### Low-level: bring your own log-probs

```python
from kondo_gate import KondoGate, KondoGateConfig

gate = KondoGate(KondoGateConfig(gate_rate=0.3))
result = gate.compute_gate(log_probs=action_log_probs, advantages=advantages)

# result.gate_weights  — per-sample gate values (0 or 1 in hard mode)
# result.gate_probs    — soft gate probabilities
# result.delight       — computed delight values
# result.price         — adaptive threshold

gated_loss = -(result.gate_weights * advantages * action_log_probs).mean()
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gate_rate` | `0.3` | Target fraction of backward passes to keep (rho). Mutually exclusive with `price`. |
| `price` | `None` | Fixed compute price threshold (lambda). Mutually exclusive with `gate_rate`. |
| `temperature` | `0.1` | Gate softness (eta). Lower = harder threshold. |
| `hard` | `True` | Bernoulli sampling with straight-through gradient (True) vs soft weights (False). |

**Adaptive pricing:** When `gate_rate` is set, the price is automatically computed as the (1-rho)-quantile of delight within each batch.

**Temperature limits:**
- As temperature approaches 0: hard indicator function (pass if delight > price)
- As temperature approaches infinity: uniform 0.5 (recovers standard policy gradient)

## Tests

```bash
pip install -e ".[dev]"
pytest
```

49 tests covering:
- Config validation (bounds, mutual exclusivity)
- Delight computation (formula correctness, detachment, edge cases)
- Gate mechanism (output shapes, hard/soft modes, adaptive rate targeting)
- Full forward pass (2D/3D logits, attention masking, loss finiteness)
- Mathematical properties (sigmoid formula, temperature limits, price monotonicity)
- Gradient verification (flow through hard/soft gates, zero-grad for gated-out samples)
- Integration (multi-step training loops, parameter updates)
- Edge cases (batch=1, zero advantages, empty masks, reproducibility)

## Example

The token reversal example trains a small causal transformer to reverse sequences, comparing learning curves at different gate rates:

```bash
python examples/token_reversal.py
```

## How it works

1. **Forward pass:** Compute log-probabilities for taken actions, then delight = advantage x surprisal
2. **Gate decision:** Set price as the (1-rho)-quantile of delight, compute gate probability via sigmoid, sample Bernoulli
3. **Gated backward:** Only samples passing the gate contribute to the gradient update

The gate filters out gradient noise from uninformative samples (low surprisal) and unreliable samples (low advantage magnitude), keeping only the samples that teach the most per unit of compute.

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
