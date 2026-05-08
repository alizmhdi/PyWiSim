"""Evaluate Binary Spray and Wait under lossy encounters."""
import argparse
import csv
import math
import os
from pathlib import Path
import statistics as stats
import sys
import tempfile

cache_root = Path(tempfile.gettempdir()) / 'pywisim-mpl-cache'
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(cache_root))
os.environ.setdefault('XDG_CACHE_HOME', str(cache_root))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spray_wait import run_binary_spray_wait


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--node-count', type=int, default=16,
                   help='Representative network size for the loss sweep')
    p.add_argument('--copies', nargs='+', default=['1', '4', 'N'],
                   help="Copy budgets K to evaluate; use integers or 'N' for K=node_count")
    p.add_argument('--losses', nargs='+', type=float, default=[0.0, 0.05, 0.10, 0.15],
                   help='Loss probabilities to evaluate during encounters')
    p.add_argument('--trials', type=int, default=50,
                   help='Independent trials per (loss, K) setting')
    p.add_argument('--until', type=float, default=60.0,
                   help='Simulation horizon per trial')
    p.add_argument('--rate', type=float, default=1.5,
                   help='Encounter rate for EncounterManager')
    p.add_argument('--duration', type=float, default=1.0,
                   help='Encounter duration for EncounterManager')
    p.add_argument('--seed-base', type=int, default=5000,
                   help='Base seed used to derive per-trial seeds')
    p.add_argument('--outdir', type=Path, default=Path('results/spray_wait_loss'),
                   help='Directory for CSV and plot outputs')
    return p.parse_args()


def mean_or_nan(values):
    return math.nan if not values else stats.fmean(values)


def resolve_copies(spec, node_count):
    text = str(spec).strip().upper()
    if text == 'N':
        return node_count, 'N'
    copies = int(text)
    if copies < 1:
        raise ValueError('copies must be >= 1')
    return copies, str(copies)


def run_sweep(args):
    rows = []
    for copy_spec in args.copies:
        copies, copies_label = resolve_copies(copy_spec, args.node_count)
        for loss in args.losses:
            delivered = 0
            delays, data_txs, control_txs = [], [], []
            for trial in range(args.trials):
                seed = args.seed_base + copies * 1000 + int(loss * 1000) * 10 + trial
                result = run_binary_spray_wait(
                    node_count=args.node_count,
                    copies=copies,
                    rate=args.rate,
                    duration=args.duration,
                    loss=loss,
                    seed=seed,
                    until=args.until,
                    verbose=False,
                )
                data_txs.append(result['data_tx'])
                control_txs.append(result['total_tx'] - result['data_tx'])
                if result['delivered']:
                    delivered += 1
                    delays.append(result['delay'])

            row = {
                'node_count': args.node_count,
                'copies': copies,
                'copies_label': copies_label,
                'loss': loss,
                'trials': args.trials,
                'delivery_rate': delivered / args.trials,
                'delay_mean': mean_or_nan(delays),
                'delay_std': 0.0 if len(delays) < 2 else stats.stdev(delays),
                'data_tx_mean': mean_or_nan(data_txs),
                'data_tx_std': 0.0 if len(data_txs) < 2 else stats.stdev(data_txs),
                'control_tx_mean': mean_or_nan(control_txs),
                'control_tx_std': 0.0 if len(control_txs) < 2 else stats.stdev(control_txs),
            }
            rows.append(row)
            delay_text = f"{row['delay_mean']:.2f}" if not math.isnan(row['delay_mean']) else 'n/a'
            print(
                f"loss={loss:>4.2f}, K={copies_label:>2} | delivery={row['delivery_rate']:.2%} "
                f"| delay={delay_text} | data_overhead={row['data_tx_mean']:.2f}"
            )
    return rows


def write_csv(rows, outdir):
    csv_path = outdir / 'spray_wait_loss_results.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def plot_rows(rows, outdir):
    copy_labels = []
    seen = set()
    for row in rows:
        label = row['copies_label']
        if label not in seen:
            copy_labels.append(label)
            seen.add(label)
    losses = sorted({row['loss'] for row in rows})
    colors = plt.cm.plasma([i / max(1, len(copy_labels) - 1) for i in range(len(copy_labels))])

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), dpi=140, sharex=True)
    for color, copies_label in zip(colors, copy_labels):
        subset = [row for row in rows if row['copies_label'] == copies_label]
        subset.sort(key=lambda row: row['loss'])
        axes[0].plot(losses, [row['delivery_rate'] for row in subset], marker='o',
                     linewidth=2, color=color, label=f'K={copies_label}')
        axes[1].plot(losses, [row['delay_mean'] for row in subset], marker='o',
                     linewidth=2, color=color, label=f'K={copies_label}')
        axes[2].plot(losses, [row['data_tx_mean'] for row in subset], marker='o',
                     linewidth=2, color=color, label=f'K={copies_label}')

    axes[0].set_ylabel('Delivery rate')
    axes[0].set_title('Binary Spray and Wait under lossy encounters')
    axes[1].set_ylabel('Average delay')
    axes[2].set_ylabel('Average data transmissions')
    axes[2].set_xlabel('Loss probability')
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(title='Copy budget')
    axes[2].set_xticks(losses)
    fig.tight_layout()

    plot_path = outdir / 'spray_wait_loss_metrics.png'
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    return plot_path


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = run_sweep(args)
    csv_path = write_csv(rows, args.outdir)
    plot_path = plot_rows(rows, args.outdir)
    print(f"\nSaved results to {csv_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == '__main__':
    main()
