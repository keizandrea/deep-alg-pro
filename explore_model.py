import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

from model import convATTnet

AA = 'ACDEFGHIKLMNPQRSTVWY'
# same encoding as predict.py: A=1 ... Y=20, 0 = padding
codeadict = {a: i + 1 for i, a in enumerate(AA)}
METRICS = ['loss', 'accuracy', 'recall', 'precision', 'auroc', 'f1']

# default protein, taken from protein_visualization.py
DEFAULT_SEQ = 'MKIVLAIASLLALSAVYARPSSIKTFEEYKKAFNKSYATFEDEEAARKNFLESVKYVQSNGGAINHLSDLSLDEFKNRFLMSAEAFEHLKTQFDLNAETNACSINGNAPAEIDLRQMRTVTPIRMQGGCGSCWAFSGVAATESAYLAYRNQSLDLAEQELVDCASQHGCHGDTIPRGIEYIQHNGVVQESYYRYVAREQSCRRPNAQRFGISNYCQIYPPNVNKIREALAQTHSAIAVIIGIKDLDAFRHYDGRTIIQRDNGYQPNYHAVNIVGYSNAQGVDYWIVRNSWDTNWGDNGYGYFAANIDLMMIEEYPYVVIL'


def plot_learning_curves(valid_file, out_png):
    df = pd.read_csv(valid_file, sep='\t', header=None, names=['epoch'] + METRICS)
    # Train.py appends to this file, so each restart of epoch 0 is a new run
    df['run'] = (df['epoch'] == 0).cumsum()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric in zip(axes.flat, METRICS):
        for run, d in df.groupby('run'):
            ax.plot(d['epoch'], d[metric], marker='o', ms=3, label=f'run {run}')
        ax.set_title(metric)
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.suptitle(f'Validation metrics per epoch ({valid_file})')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f'saved {out_png}')


def encode(seq):
    seq = seq[:1000]
    return [0] * (1000 - len(seq)) + [codeadict[a] for a in seq]


def load_model(path):
    model = convATTnet()
    model.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
    model.eval()
    # capture the pre-sigmoid logit: confident predictions saturate near 1.0,
    # which would hide the effect of single mutations
    logits = []
    model.fc1.register_forward_hook(lambda m, i, o: logits.append(o.squeeze(-1)))
    return model, logits


@torch.no_grad()
def mutagenesis(model, logits, seq, batch_size=64):
    """Score every single-residue substitution. Returns (wt_logit, 20 x L delta matrix)."""
    seq = seq[:1000]
    offset = 1000 - len(seq)
    wt = encode(seq)

    variants = []
    for pos in range(len(seq)):
        for aa in AA:
            v = list(wt)
            v[offset + pos] = codeadict[aa]
            variants.append(v)

    x = torch.tensor([wt] + variants, dtype=torch.long)
    logits.clear()
    for i in range(0, len(x), batch_size):
        model(x[i:i + batch_size])
    scores = torch.cat(logits).numpy()

    wt_logit = scores[0]
    delta = (scores[1:] - wt_logit).reshape(len(seq), len(AA)).T
    return wt_logit, delta


def plot_mutagenesis(seq, wt_logit, delta, title, out_png):
    L = len(seq)
    # per-position importance: average loss of logit when the residue is replaced
    importance = -delta.mean(axis=0)
    lim = np.abs(delta).max()

    fig, ax = plt.subplots(2, 1, figsize=(max(12, L / 25), 7), sharex=True,
                           gridspec_kw={'height_ratios': [1, 2.2]})
    ax[0].bar(np.arange(L), importance, width=1.0,
              color=np.where(importance > 0, 'tab:red', 'tab:blue'))
    for i, a in enumerate(seq):
        if a == 'C':
            ax[0].axvline(i, color='gold', lw=0.8, alpha=0.7)
    ax[0].axhline(0, color='gray', lw=0.5)
    ax[0].set_ylabel('importance\n(-mean Δlogit)')
    ax[0].set_title(f'{title}  |  WT logit {wt_logit:.2f}  '
                    f'(p={1 / (1 + np.exp(-wt_logit)):.4f})  |  gold = Cys')

    im = ax[1].imshow(delta, aspect='auto', cmap='RdBu', vmin=-lim, vmax=lim,
                      interpolation='nearest')
    ax[1].set_yticks(range(len(AA)), list(AA))
    ax[1].set_ylabel('substituted to')
    ax[1].set_xlabel('position')
    fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.04, pad=0.08,
                 label='Δlogit vs wild type (blue = less allergenic)')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'saved {out_png}')

    top = np.argsort(-importance)[:10]
    print('top positions (1-based):')
    for i in top:
        print(f'  {seq[i]}{i + 1}: importance {importance[i]:.3f}, '
              f'worst substitution -> {AA[delta[:, i].argmin()]} ({delta[:, i].min():.3f})')


def main():
    p = argparse.ArgumentParser(description='Learning curves and in-silico mutagenesis for DeepAlgPro')
    p.add_argument('--valid', default='model.1.fold.everyepoch.valid.txt')
    p.add_argument('--model', default='model.1.pt')
    p.add_argument('--fasta', help='FASTA to take the protein from (default: built-in sequence)')
    p.add_argument('--id', help='record id inside --fasta (default: first record)')
    args = p.parse_args()

    plot_learning_curves(args.valid, 'learning_curves.png')

    name, seq = 'protein_visualization.py sequence', DEFAULT_SEQ
    if args.fasta:
        for rec in SeqIO.parse(args.fasta, 'fasta'):
            if args.id is None or rec.id == args.id:
                name, seq = rec.id, str(rec.seq).upper()
                break
        else:
            raise SystemExit(f'{args.id} not found in {args.fasta}')
    if len(seq) > 1000:
        print(f'warning: {name} has {len(seq)} residues, only the first 1000 are used')

    model, logits = load_model(args.model)
    wt_logit, delta = mutagenesis(model, logits, seq)
    plot_mutagenesis(seq[:1000], wt_logit, delta, f'{name} ({args.model})', 'mutagenesis.png')


if __name__ == '__main__':
    main()
