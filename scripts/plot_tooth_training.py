"""Plot the recorded validation detection metrics for the demo evidence."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with args.csv.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error('The training record has no completed epochs')
    epochs = [int(row['epoch']) for row in rows]
    loose = [float(row['metrics/mAP50(B)']) for row in rows]
    strict = [float(row['metrics/mAP50-95(B)']) for row in rows]
    best = max(range(len(rows)), key=lambda index: strict[index])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor('white')
    ax.plot(epochs, loose, color='#167c80', marker='o', label='Box mAP50')
    ax.plot(epochs, strict, color='#b16b25', marker='o', label='Box mAP50–95')
    ax.scatter([epochs[best]], [strict[best]], color='#142c3a', s=85, zorder=4)
    ax.annotate(f'Selected epoch {epochs[best]}', (epochs[best], strict[best]),
                xytext=(0, -22), textcoords='offset points', ha='center', fontsize=10)
    ax.set(title='DentiLens tooth detector: validation results', xlabel='Completed training epoch',
           ylabel='Mean average precision', ylim=(0, 1.03), xticks=epochs)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.18)
    ax.legend(loc='lower right', frameon=False)
    fig.text(0.08, 0.01, '246 supplied validation images. One tooth class. Detection boxes only; no mask or clinical accuracy claim.', fontsize=9, color='#52606a')
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches='tight')
    plt.close(fig)
