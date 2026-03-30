"""
MNIST Contextual Bandit: PG vs DG vs DG-K (Kondo Gate)

PyTorch port of the reference Colab from the paper (arXiv:2603.20526).

An image goes in, the agent picks a digit (0-9), gets reward 1 if correct
and 0 otherwise. Three methods are compared:

  - PG:  Standard REINFORCE. Backward pass on all 100 samples per step.
  - DG:  Sigmoid-gated. All 100 backward passes, weighted by delight.
  - DG-K(3%): Kondo gate. Backward pass on the 3 most delightful samples.
              Skips the other 97.

Same network, same optimizer, same data.

Usage:
    python examples/mnist_bandit.py

Requires: torchvision (for MNIST data loading)
    pip install torchvision
"""

import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from kondo_gate import (
    KondoGate,
    KondoGateConfig,
    dg_loss,
    expected_confidence_baseline,
    pg_loss,
)


# ============================================================================
# Data
# ============================================================================


def load_mnist(device="cpu"):
    """Load MNIST as flat tensors. Falls back to torchvision."""
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # flatten to 784
    ])
    train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_images = torch.stack([img for img, _ in train]).to(device)
    train_labels = torch.tensor([lbl for _, lbl in train], dtype=torch.long, device=device)
    test_images = torch.stack([img for img, _ in test]).to(device)
    test_labels = torch.tensor([lbl for _, lbl in test], dtype=torch.long, device=device)

    return train_images, train_labels, test_images, test_labels


# ============================================================================
# Model
# ============================================================================


class Policy(nn.Module):
    """Two-layer ReLU MLP policy: images -> logits over 10 digits."""

    def __init__(self, input_dim=784, hidden_sizes=(100, 100), num_actions=10):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Training methods
# ============================================================================


@dataclass
class TrainConfig:
    num_steps: int = 10_000
    batch_size: int = 100
    lr: float = 1e-3
    eval_every: int = 200
    device: str = "cpu"


def train_pg(model, config, train_images, train_labels, test_images, test_labels):
    """Train with standard REINFORCE (PG)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = []
    n_train = train_images.shape[0]

    for step in range(config.num_steps):
        idx = torch.randint(0, n_train, (config.batch_size,), device=config.device)
        images = train_images[idx]
        labels = train_labels[idx]

        logits = model(images)
        probs = F.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)
        rewards = (actions == labels).float()
        baseline = expected_confidence_baseline(probs)
        advantages = rewards - baseline.detach()

        optimizer.zero_grad()
        loss = pg_loss(logits, actions, advantages)
        loss.backward()
        optimizer.step()

        if step % config.eval_every == 0 or step == config.num_steps - 1:
            test_err = evaluate(model, test_images, test_labels)
            history.append({
                "step": step,
                "test_error": test_err,
                "reward": rewards.mean().item(),
                "backward_frac": 1.0,
            })

    return history


def train_dg(model, config, train_images, train_labels, test_images, test_labels, eta=1.0):
    """Train with Delightful Gradient (DG) — sigmoid weighting."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = []
    n_train = train_images.shape[0]

    for step in range(config.num_steps):
        idx = torch.randint(0, n_train, (config.batch_size,), device=config.device)
        images = train_images[idx]
        labels = train_labels[idx]

        logits = model(images)
        probs = F.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)
        rewards = (actions == labels).float()
        baseline = expected_confidence_baseline(probs)
        advantages = rewards - baseline.detach()

        optimizer.zero_grad()
        loss, gate = dg_loss(logits, actions, advantages, eta=eta)
        loss.backward()
        optimizer.step()

        if step % config.eval_every == 0 or step == config.num_steps - 1:
            test_err = evaluate(model, test_images, test_labels)
            history.append({
                "step": step,
                "test_error": test_err,
                "reward": rewards.mean().item(),
                "backward_frac": 1.0,
                "gate_mean": gate.mean().item(),
            })

    return history


def train_kondo(
    model, config, train_images, train_labels, test_images, test_labels,
    backward_frac=0.03,
):
    """Train with Kondo gate (DG-K) — deterministic top-k selection."""
    gate = KondoGate(KondoGateConfig(
        gate_rate=backward_frac,
        deterministic=True,
        hard=True,
    ))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = []
    n_train = train_images.shape[0]

    for step in range(config.num_steps):
        idx = torch.randint(0, n_train, (config.batch_size,), device=config.device)
        images = train_images[idx]
        labels = train_labels[idx]

        logits = model(images)
        probs = F.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)
        rewards = (actions == labels).float()
        baseline = expected_confidence_baseline(probs)
        advantages = rewards - baseline.detach()

        optimizer.zero_grad()
        result = gate(logits, actions, advantages)
        result.gated_policy_loss.backward()
        optimizer.step()

        if step % config.eval_every == 0 or step == config.num_steps - 1:
            test_err = evaluate(model, test_images, test_labels)
            history.append({
                "step": step,
                "test_error": test_err,
                "reward": rewards.mean().item(),
                "backward_frac": backward_frac,
                "gate_rate": result.actual_gate_rate.item(),
            })

    return history


def evaluate(model, test_images, test_labels):
    """Compute test error rate."""
    with torch.no_grad():
        logits = model(test_images)
        preds = logits.argmax(dim=-1)
        error = 1.0 - (preds == test_labels).float().mean().item()
    return error


# ============================================================================
# Main
# ============================================================================


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist(device)
    print(f"  Train: {train_images.shape[0]}, Test: {test_images.shape[0]}")

    config = TrainConfig(
        num_steps=10_000,
        batch_size=100,
        lr=1e-3,
        eval_every=200,
        device=device,
    )

    num_seeds = 3
    methods = {
        "PG": lambda model: train_pg(model, config, train_images, train_labels, test_images, test_labels),
        "DG": lambda model: train_dg(model, config, train_images, train_labels, test_images, test_labels),
        "DG-K(3%)": lambda model: train_kondo(model, config, train_images, train_labels, test_images, test_labels, backward_frac=0.03),
    }

    all_results = {}
    for name, train_fn in methods.items():
        print(f"\n{'='*60}")
        print(f"  Training: {name} ({num_seeds} seeds)")
        print(f"{'='*60}")

        seed_results = []
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            model = Policy().to(device)
            t0 = time.time()
            history = train_fn(model)
            elapsed = time.time() - t0
            seed_results.append(history)

            final = history[-1]
            print(f"  seed {seed}: test_error={final['test_error']:.4f}  "
                  f"reward={final['reward']:.3f}  time={elapsed:.1f}s")

        all_results[name] = seed_results

    # ── Summary table ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Summary (mean +/- std across {num_seeds} seeds)")
    print(f"{'='*70}")
    print(f"  {'Method':<12s} | {'Final Error':>12s} | {'Backward %':>11s} | {'Effective Cost':>14s}")
    print(f"  {'-'*55}")

    for name, seed_results in all_results.items():
        final_errors = [h[-1]["test_error"] for h in seed_results]
        bfrac = seed_results[0][-1]["backward_frac"]
        mean_err = sum(final_errors) / len(final_errors)
        std_err = (sum((e - mean_err)**2 for e in final_errors) / len(final_errors)) ** 0.5
        cost = bfrac  # backward cost relative to PG
        print(f"  {name:<12s} | {mean_err:.4f} +/- {std_err:.4f} | {bfrac:>10.0%} | {cost:>13.0%}")

    # ── Learning curves ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Learning Curves (seed 0, by forward passes)")
    print(f"{'='*70}")
    print(f"  {'Step':>6s} | {'PG':>10s} | {'DG':>10s} | {'DG-K(3%)':>10s}")
    print(f"  {'-'*44}")

    pg_hist = all_results["PG"][0]
    dg_hist = all_results["DG"][0]
    dk_hist = all_results["DG-K(3%)"][0]

    for i in range(min(len(pg_hist), len(dg_hist), len(dk_hist))):
        step = pg_hist[i]["step"]
        if step % 1000 == 0 or i == len(pg_hist) - 1:
            print(f"  {step:6d} | {pg_hist[i]['test_error']:10.4f} | "
                  f"{dg_hist[i]['test_error']:10.4f} | {dk_hist[i]['test_error']:10.4f}")

    # ── Backward pass comparison ───────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Learning Curves (seed 0, by backward passes)")
    print(f"{'='*70}")
    print(f"  {'Back Steps':>10s} | {'PG':>10s} | {'DG':>10s} | {'DG-K(3%)':>10s}")
    print(f"  {'-'*48}")

    for i in range(min(len(pg_hist), len(dg_hist), len(dk_hist))):
        step = pg_hist[i]["step"]
        if step % 1000 == 0 or i == len(pg_hist) - 1:
            pg_back = step * 1.0
            dg_back = step * 1.0
            dk_back = step * 0.03
            print(f"  {pg_back:10.0f} | {pg_hist[i]['test_error']:10.4f} | "
                  f"{dg_hist[i]['test_error']:10.4f} | {dk_hist[i]['test_error']:10.4f}")

    print(f"""
Key result from the paper:
  Measured in forward passes, DG-K matches DG almost exactly.
  Measured in backward passes, DG-K is ~30x cheaper to reach the same error.
  You can throw away 97% of the backward compute and still beat PG.
""")


if __name__ == "__main__":
    main()
