"""Evaluate Binary Spray and Wait across node counts and copy budgets."""
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
    p.add_argument('--nodes', nargs='+', type=int, default=[8, 12, 16, 20],
                   help='Node counts to evaluate')
    p.add_argument('--copies', nargs='+', default=['2', '4', '8'],
                   help="Copy budgets K to evaluate; use integers or 'N' for K=node_count")
    p.add_argument('--trials', type=int, default=20,
                   help='Independent trials per (nodes, K) setting')
    p.add_argument('--until', type=float, default=60.0,
                   help='Simulation horizon per trial')
    p.add_argument('--rate', type=float, default=1.5,
                   help='Encounter rate for EncounterManager')
    p.add_argument('--duration', type=float, default=1.0,
                   help='Encounter duration for EncounterManager')
    p.add_argument('--seed-base', type=int, default=1000,
                   help='Base seed used to derive per-trial seeds')
    p.add_argument('--outdir', type=Path, default=Path('results/spray_wait'),
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
    for node_count in args.nodes:
        for copy_spec in args.copies:
            copies, copies_label = resolve_copies(copy_spec, node_count)
            delays, total_txs, data_txs, control_txs = [], [], [], []
            delivered = 0
            for trial in range(args.trials):
                seed = args.seed_base + node_count * 1000 + copies * 100 + trial
                result = run_binary_spray_wait(
                    node_count=node_count,
                    copies=copies,
                    rate=args.rate,
                    duration=args.duration,
                    seed=seed,
                    until=args.until,
                    verbose=False,
                )
                total_txs.append(result['total_tx'])
                data_txs.append(result['data_tx'])
                control_txs.append(result['total_tx'] - result['data_tx'])
                if result['delivered']:
                    delivered += 1
                    delays.append(result['delay'])

            row = {
                'node_count': node_count,
                'copies': copies,
                'copies_label': copies_label,
                'trials': args.trials,
                'delivery_rate': delivered / args.trials,
                'delay_mean': mean_or_nan(delays),
                'delay_std': 0.0 if len(delays) < 2 else stats.stdev(delays),
                'total_tx_mean': mean_or_nan(total_txs),
                'total_tx_std': 0.0 if len(total_txs) < 2 else stats.stdev(total_txs),
                'data_tx_mean': mean_or_nan(data_txs),
                'data_tx_std': 0.0 if len(data_txs) < 2 else stats.stdev(data_txs),
                'control_tx_mean': mean_or_nan(control_txs),
                'control_tx_std': 0.0 if len(control_txs) < 2 else stats.stdev(control_txs),
            }
            rows.append(row)
            delay_text = f"{row['delay_mean']:.2f}" if not math.isnan(row['delay_mean']) else 'n/a'
            print(
                f"n={node_count:>2}, K={copies_label:>2} | delivery={row['delivery_rate']:.2%} "
                f"| delay={delay_text} | data_overhead={row['data_tx_mean']:.2f} "
                f"| control_overhead={row['control_tx_mean']:.2f}"
            )
    return rows


def write_csv(rows, outdir):
    csv_path = outdir / 'spray_wait_results.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def plot_rows(rows, outdir):
    node_counts = sorted({row['node_count'] for row in rows})
    copy_labels = []
    seen = set()
    for row in rows:
        label = row['copies_label']
        if label not in seen:
            copy_labels.append(label)
            seen.add(label)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), dpi=140, sharex=True)
    colors = plt.cm.viridis([i / max(1, len(copy_labels) - 1) for i in range(len(copy_labels))])

    for color, copies_label in zip(colors, copy_labels):
        subset = [row for row in rows if row['copies_label'] == copies_label]
        subset.sort(key=lambda row: row['node_count'])
        delays = [row['delay_mean'] for row in subset]
        overheads = [row['data_tx_mean'] for row in subset]
        axes[0].plot(node_counts, delays, marker='o', linewidth=2, color=color, label=f'K={copies_label}')
        axes[1].plot(node_counts, overheads, marker='o', linewidth=2, color=color, label=f'K={copies_label}')

    axes[0].set_ylabel('Average delay')
    axes[0].set_title('Binary Spray and Wait: Delay vs. node count')
    axes[1].set_ylabel('Average data transmissions')
    axes[1].set_xlabel('Number of nodes')
    axes[1].set_title('Binary Spray and Wait: Data overhead vs. node count')
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(title='Copy budget')
    axes[1].set_xticks(node_counts)

    fig.tight_layout()
    metrics_path = outdir / 'spray_wait_metrics.png'
    fig.savefig(metrics_path, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    for color, copies_label in zip(colors, copy_labels):
        subset = [row for row in rows if row['copies_label'] == copies_label]
        subset.sort(key=lambda row: row['node_count'])
        controls = [row['control_tx_mean'] for row in subset]
        ax.plot(node_counts, controls, marker='o', linewidth=2, color=color, label=f'K={copies_label}')

    ax.set_ylabel('Average control transmissions')
    ax.set_xlabel('Number of nodes')
    ax.set_title('Binary Spray and Wait: Control overhead vs. node count')
    ax.set_xticks(node_counts)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Copy budget')
    fig.tight_layout()
    control_path = outdir / 'spray_wait_control_overhead.png'
    fig.savefig(control_path, bbox_inches='tight')
    plt.close(fig)
    return metrics_path, control_path


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = run_sweep(args)
    csv_path = write_csv(rows, args.outdir)
    metrics_path, control_path = plot_rows(rows, args.outdir)
    print(f"\nSaved results to {csv_path}")
    print(f"Saved metrics plot to {metrics_path}")
    print(f"Saved control-overhead plot to {control_path}")


if __name__ == '__main__':
    main()
