"""
Full MNIST comparison: PG vs DG vs DG-K at multiple gate rates.
Saves CSV data and generates publication-quality figures.
"""

import csv
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from kondo_gate import (
    KondoGate,
    KondoGateConfig,
    dg_loss,
    expected_confidence_baseline,
    pg_loss,
)


def load_mnist(device="cpu"):
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_images = torch.stack([img for img, _ in train]).to(device)
    train_labels = torch.tensor([lbl for _, lbl in train], dtype=torch.long, device=device)
    test_images = torch.stack([img for img, _ in test]).to(device)
    test_labels = torch.tensor([lbl for _, lbl in test], dtype=torch.long, device=device)
    return train_images, train_labels, test_images, test_labels


class Policy(nn.Module):
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


def evaluate(model, test_images, test_labels):
    with torch.no_grad():
        logits = model(test_images)
        preds = logits.argmax(dim=-1)
        return 1.0 - (preds == test_labels).float().mean().item()


def train_method(method, model, train_images, train_labels, test_images, test_labels,
                 num_steps=10000, batch_size=100, lr=1e-3, eval_every=100,
                 gate_rate=0.03, eta=1.0, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = train_images.shape[0]
    history = []

    gate = None
    if method.startswith("DG-K"):
        gate = KondoGate(KondoGateConfig(gate_rate=gate_rate, deterministic=True, hard=True))

    for step in range(num_steps):
        idx = torch.randint(0, n_train, (batch_size,), device=device)
        images, labels = train_images[idx], train_labels[idx]

        logits = model(images)
        probs = F.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)
        rewards = (actions == labels).float()
        baseline = expected_confidence_baseline(probs)
        advantages = rewards - baseline.detach()

        optimizer.zero_grad()

        if method == "PG":
            loss = pg_loss(logits, actions, advantages)
            backward_frac = 1.0
        elif method == "DG":
            loss, _ = dg_loss(logits, actions, advantages, eta=eta)
            backward_frac = 1.0
        else:
            result = gate(logits, actions, advantages)
            loss = result.gated_policy_loss
            backward_frac = gate_rate

        loss.backward()
        optimizer.step()

        if step % eval_every == 0 or step == num_steps - 1:
            test_err = evaluate(model, test_images, test_labels)
            history.append({
                "step": step,
                "test_error": test_err,
                "reward": rewards.mean().item(),
                "backward_frac": backward_frac,
                "loss": loss.item(),
            })

    return history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist(device)

    num_steps = 10_000
    num_seeds = 5
    eval_every = 100

    methods = [
        ("PG", {}),
        ("DG", {"eta": 1.0}),
        ("DG-K(10%)", {"gate_rate": 0.10}),
        ("DG-K(3%)", {"gate_rate": 0.03}),
        ("DG-K(1%)", {"gate_rate": 0.01}),
    ]

    all_rows = []

    for method_name, kwargs in methods:
        print(f"\n{'='*60}")
        print(f"  {method_name} ({num_seeds} seeds, {num_steps} steps)")
        print(f"{'='*60}")

        for seed in range(num_seeds):
            torch.manual_seed(seed)
            model = Policy().to(device)
            t0 = time.time()

            m = method_name.split("(")[0].strip()
            history = train_method(
                m, model, train_images, train_labels, test_images, test_labels,
                num_steps=num_steps, eval_every=eval_every, device=device, **kwargs,
            )
            elapsed = time.time() - t0
            final = history[-1]
            print(f"  seed {seed}: error={final['test_error']:.4f}  "
                  f"reward={final['reward']:.3f}  {elapsed:.1f}s")

            for h in history:
                all_rows.append({
                    "method": method_name,
                    "seed": seed,
                    **h,
                })

    # Save CSV
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "mnist_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "seed", "step", "test_error",
                                                "reward", "backward_frac", "loss"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {csv_path}")

    # Generate plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        colors = {
            "PG": "#d62728",
            "DG": "#1f77b4",
            "DG-K(10%)": "#2ca02c",
            "DG-K(3%)": "#ff7f0e",
            "DG-K(1%)": "#9467bd",
        }

        # Organize data
        data = {}
        for row in all_rows:
            key = (row["method"], row["step"])
            if key not in data:
                data[key] = []
            data[key].append(row["test_error"])

        method_names = [m for m, _ in methods]
        steps_set = sorted(set(r["step"] for r in all_rows))
        steps_arr = np.array(steps_set)

        # --- Plot 1: by forward passes ---
        ax = axes[0]
        for mname in method_names:
            means, sems = [], []
            for s in steps_set:
                vals = data.get((mname, s), [])
                if vals:
                    m = np.mean(vals)
                    se = np.std(vals) / np.sqrt(len(vals))
                    means.append(m)
                    sems.append(se)
            means = np.array(means)
            sems = np.array(sems)
            fwd = steps_arr * 100 / 1e6  # forward passes in millions
            ax.plot(fwd, means, label=mname, color=colors[mname], linewidth=2)
            ax.fill_between(fwd, means - sems, means + sems, color=colors[mname], alpha=0.15)

        ax.set_yscale("log")
        ax.set_xlabel("Forward passes (millions)", fontsize=13)
        ax.set_ylabel("Test error", fontsize=13)
        ax.set_title("Learning curves by forward passes", fontsize=14)
        ax.legend(fontsize=11, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0.02)

        # --- Plot 2: by backward passes ---
        ax = axes[1]
        bfrac_map = {"PG": 1.0, "DG": 1.0, "DG-K(10%)": 0.10, "DG-K(3%)": 0.03, "DG-K(1%)": 0.01}

        for mname in method_names:
            means, sems = [], []
            for s in steps_set:
                vals = data.get((mname, s), [])
                if vals:
                    means.append(np.mean(vals))
                    sems.append(np.std(vals) / np.sqrt(len(vals)))
            means = np.array(means)
            sems = np.array(sems)
            bf = bfrac_map[mname]
            back = steps_arr * 100 * bf / 1e6
            ax.plot(back, means, label=mname, color=colors[mname], linewidth=2)
            ax.fill_between(back, means - sems, means + sems, color=colors[mname], alpha=0.15)

        ax.set_yscale("log")
        ax.set_xlabel("Backward passes (millions)", fontsize=13)
        ax.set_ylabel("Test error", fontsize=13)
        ax.set_title("Learning curves by backward passes", fontsize=14)
        ax.legend(fontsize=11, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0.02)

        fig.suptitle("MNIST Contextual Bandit: PG vs DG vs Kondo Gate (DG-K)",
                     fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()

        fig_path = os.path.join(out_dir, "mnist_comparison.png")
        fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved figure to {fig_path}")

        # --- Summary table as figure ---
        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        ax2.axis("off")

        # Compute summary stats
        table_data = []
        for mname in method_names:
            final_errors = [data.get((mname, num_steps - 1), data.get((mname, 9999), [0]))]
            # flatten
            fe = []
            for s in range(num_seeds):
                key = (mname, num_steps - 1)
                if key not in data:
                    key = (mname, 9999)
                vals = data.get(key, [])
                if s < len(vals):
                    fe.append(vals[s])
            mean_e = np.mean(fe) if fe else 0
            std_e = np.std(fe) if fe else 0
            bf = bfrac_map[mname]

            # Steps to reach 5% error (approximate)
            steps_to_5 = "N/A"
            for s in steps_set:
                vals = data.get((mname, s), [])
                if vals and np.mean(vals) <= 0.05:
                    steps_to_5 = f"{s:,}"
                    break

            table_data.append([
                mname,
                f"{mean_e:.2%} ± {std_e:.2%}",
                f"{bf:.0%}",
                f"{bf * num_steps * 100 / 1e6:.2f}M",
                steps_to_5,
            ])

        table = ax2.table(
            cellText=table_data,
            colLabels=["Method", "Final Error", "Backward %", "Total Backward", "Steps to 5%"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.8)

        # Style header
        for j in range(5):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            color = "#f0f4ff" if i % 2 == 0 else "white"
            for j in range(5):
                table[i, j].set_facecolor(color)

        ax2.set_title("Summary: MNIST Contextual Bandit Results (10K steps, 5 seeds)",
                      fontsize=14, fontweight="bold", pad=20)

        fig2_path = os.path.join(out_dir, "mnist_summary_table.png")
        fig2.savefig(fig2_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved table to {fig2_path}")

    except ImportError:
        print("matplotlib not available — skipping plots. Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
