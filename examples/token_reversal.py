"""
Example: Training a token-reversal transformer with Kondo gating.

This replicates the token reversal task from the paper — a decoder-only
transformer learns to reverse input sequences, trained with REINFORCE
and Kondo-gated backward passes.

The example demonstrates:
  1. Defining a small causal transformer
  2. Generating reversal tasks as a bandit environment
  3. Training with KondoTrainer at different gate rates
  4. Comparing learning curves: full training vs. gated training

No GPU required — runs in ~30 seconds on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kondo_gate import KondoGate, KondoGateConfig, KondoTrainer


# ============================================================================
# 1. Tiny Causal Transformer
# ============================================================================


class TinyCausalTransformer(nn.Module):
    """Minimal decoder-only transformer (matches paper's architecture)."""

    def __init__(self, vocab_size=16, d_model=64, n_heads=2, n_layers=2, max_len=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.max_len = max_len

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        h = self.tok_emb(input_ids) + self.pos_emb(pos)

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=h.device), diagonal=1).bool()
        h = self.transformer(h, mask=causal_mask, is_causal=True)
        logits = self.head(h)

        class Output:
            pass

        out = Output()
        out.logits = logits
        return out


# ============================================================================
# 2. Token Reversal Environment
# ============================================================================


def generate_reversal_batch(batch_size, seq_len, vocab_size, device="cpu"):
    """Generate a batch of reversal tasks.

    Input:  [t1, t2, ..., tn, SEP]
    Target: [tn, ..., t2, t1, SEP]

    Returns input_ids, target_ids (both shape [B, 2*seq_len+1]).
    SEP token is vocab_size - 1.
    """
    sep = vocab_size - 1
    tokens = torch.randint(0, vocab_size - 1, (batch_size, seq_len), device=device)
    reversed_tokens = tokens.flip(dims=[1])
    sep_col = torch.full((batch_size, 1), sep, device=device)

    input_ids = torch.cat([tokens, sep_col, reversed_tokens], dim=1)
    # Target is shifted: predict next token at each position
    # For the reversal section, target[seq_len:] = reversed_tokens + SEP
    target_ids = torch.cat([
        torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),  # don't care
        reversed_tokens,
        sep_col,
    ], dim=1)

    return input_ids, target_ids


def compute_reward(logits, target_ids, seq_len):
    """Reward = fraction of correctly predicted reversal tokens."""
    # Only score the reversal portion (positions seq_len to end)
    pred = logits[:, seq_len:-1].argmax(dim=-1)  # (B, seq_len)
    target = target_ids[:, seq_len:-1]  # (B, seq_len)
    correct = (pred == target).float().mean(dim=-1)  # (B,)
    return correct


# ============================================================================
# 3. Training Loop
# ============================================================================


def train_with_kondo(gate_rate, n_steps=150, batch_size=64, seq_len=5, vocab_size=10, seed=42):
    """Train a reversal model with a given Kondo gate rate.

    Returns list of (step, reward, gate_rate) tuples.
    """
    torch.manual_seed(seed)
    device = "cpu"

    model = TinyCausalTransformer(
        vocab_size=vocab_size, d_model=64, n_heads=2, n_layers=2,
        max_len=2 * seq_len + 2,
    ).to(device)

    gate = KondoGate(KondoGateConfig(gate_rate=gate_rate, temperature=0.05, hard=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    total_len = 2 * seq_len + 1
    # Mask: only score reversal tokens (positions seq_len onward)
    mask = torch.zeros(batch_size, total_len, device=device)
    mask[:, seq_len:] = 1.0

    history = []
    backward_count = 0

    for step in range(n_steps):
        input_ids, target_ids = generate_reversal_batch(batch_size, seq_len, vocab_size, device)

        with torch.no_grad():
            outputs = model(input_ids)
            reward = compute_reward(outputs.logits, target_ids, seq_len)  # (B,)

        # Baseline: running mean reward
        if step == 0:
            baseline = reward.mean()
        else:
            baseline = 0.95 * baseline + 0.05 * reward.mean()

        # Per-token advantages (broadcast from per-sample reward)
        advantages = (reward - baseline).unsqueeze(1).expand(-1, total_len) * mask

        # Forward with gradient
        optimizer.zero_grad()
        outputs = model(input_ids)
        result = gate(outputs.logits, target_ids, advantages, attention_mask=mask)
        result.gated_policy_loss.backward()
        optimizer.step()

        actual_rate = result.actual_gate_rate.item()
        backward_count += actual_rate * batch_size

        if step % 30 == 0 or step == n_steps - 1:
            history.append({
                "step": step,
                "reward": reward.mean().item(),
                "gate_rate": actual_rate,
                "price": result.price.item(),
                "backward_count": backward_count,
            })

    return history, backward_count


# ============================================================================
# 4. Run Comparison
# ============================================================================


def main():
    print("=" * 70)
    print("  Kondo Gate Example: Token Reversal Task")
    print("  Training a small transformer to reverse sequences")
    print("=" * 70)

    configs = [
        ("Full (ρ=1.0)", 1.0),
        ("Gated (ρ=0.3)", 0.3),
        ("Gated (ρ=0.1)", 0.1),
        ("Gated (ρ=0.03)", 0.03),
    ]

    results = {}
    for name, rate in configs:
        print(f"\nTraining: {name} ...")
        history, total_backward = train_with_kondo(gate_rate=rate, n_steps=150)
        results[name] = (history, total_backward)

        for h in history:
            print(f"  step {h['step']:4d} | reward {h['reward']:.3f} | "
                  f"gate_rate {h['gate_rate']:.2%} | price {h['price']:+.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("  Summary: Backward Pass Savings")
    print("=" * 70)
    full_backward = results["Full (ρ=1.0)"][1]
    full_final_reward = results["Full (ρ=1.0)"][0][-1]["reward"]

    for name, (history, total_backward) in results.items():
        final_reward = history[-1]["reward"]
        savings = (1 - total_backward / full_backward) * 100 if full_backward > 0 else 0
        quality = final_reward / full_final_reward * 100 if full_final_reward > 0 else 0
        print(f"  {name:20s} | final reward {final_reward:.3f} | "
              f"backward savings {savings:5.1f}% | quality {quality:5.1f}%")

    # === Low-level API demo ===
    print("\n" + "=" * 70)
    print("  Low-Level API Demo")
    print("=" * 70)

    gate = KondoGate(KondoGateConfig(gate_rate=0.3, temperature=0.1, hard=True))

    # Simulate 10 samples with known delight structure
    torch.manual_seed(0)
    log_probs = torch.tensor([-0.5, -1.0, -2.0, -3.0, -0.1, -4.0, -0.2, -1.5, -2.5, -0.8])
    advantages = torch.tensor([1.0, -0.5, 2.0, 0.1, 0.0, 3.0, -1.0, 0.5, 1.5, -0.2])

    delight = gate.compute_delight(log_probs, advantages)
    result = gate.compute_gate(log_probs, advantages)

    print(f"\n  {'Sample':>8s} | {'LogProb':>8s} | {'Adv':>8s} | {'Surprisal':>9s} | "
          f"{'Delight':>8s} | {'GateProb':>9s} | {'Gate':>5s}")
    print("  " + "-" * 75)
    for i in range(len(log_probs)):
        print(f"  {i:8d} | {log_probs[i]:8.2f} | {advantages[i]:8.2f} | "
              f"{-log_probs[i]:9.2f} | {delight[i]:8.2f} | "
              f"{result.gate_probs[i]:9.4f} | {'PASS' if result.gate_weights[i].item() > 0.5 else 'SKIP':>5s}")

    print(f"\n  Price (λ): {result.price.item():.3f}")
    print(f"  Actual gate rate: {result.actual_gate_rate.item():.0%}")
    print(f"  Samples passing: {int(result.gate_weights.sum().item())}/{len(log_probs)}")

    # === KondoTrainer demo ===
    print("\n" + "=" * 70)
    print("  KondoTrainer with HuggingFace-Style Model")
    print("=" * 70)

    class FakeLM(nn.Module):
        """Mimics HuggingFace AutoModelForCausalLM interface."""
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)
            self.linear = nn.Linear(32, 100)

        def forward(self, input_ids, attention_mask=None):
            h = self.embed(input_ids)
            logits = self.linear(h)
            class Out:
                pass
            out = Out()
            out.logits = logits
            return out

    model = FakeLM()
    trainer = KondoTrainer(model, gate_rate=0.2, temperature=0.05, lr=1e-3)

    print("\n  Step | Loss      | Gate Rate | Price   | Mean Delight")
    print("  " + "-" * 55)
    for step in range(10):
        input_ids = torch.randint(0, 100, (16, 12))
        actions = torch.randint(0, 100, (16, 12))
        advantages = torch.randn(16, 12)

        stats = trainer.step(input_ids, actions, advantages)
        print(f"  {step:4d} | {stats['loss']:9.4f} | {stats['gate_rate']:9.0%} | "
              f"{stats['price']:7.3f} | {stats['mean_delight']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
