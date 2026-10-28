import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from Bio import SeqIO

from explore_model import DEFAULT_SEQ, encode, load_model

K = 16  # conv1 kernel size: window j covers padded positions j .. j+15
N_WIN = 985  # number of conv windows (1000 - 16 + 1)
# the three position-major (1, 985, C) feature maps we can explain
LAYERS = {'conv': 'conv1 + ReLU (64 filters)',
          'pool': 'after maxpool (12 features)',
          'attn': 'self-attention output (24 features)'}


def capture(model):
    """Hook each layer's feature map and its gradient. Returns (acts, grads, handles)."""
    acts, grads = {}, {}

    def keep(name, t):
        acts[name] = t
        t.register_hook(lambda g: grads.__setitem__(name, g))

    handles = [
        # maxpool's input is the post-ReLU conv output, already permuted to (1, 985, 64)
        model.maxpool.register_forward_hook(lambda m, i, o: keep('conv', i[0])),
        # dropout's output is the attention input; model.py calls selfattention.forward()
        # directly, which bypasses hooks on selfattention itself
        model.dropout.register_forward_hook(lambda m, i, o: keep('pool', o)),
        # fc1's input is the flattened attention output (1, 985*24)
        model.fc1.register_forward_hook(lambda m, i, o: keep('attn', i[0])),
    ]
    return acts, grads, handles


def gradcam(model, logits, seq, mode='gradcam'):
    """Grad-CAM (or gradient x activation) w.r.t. the pre-sigmoid logit.
    Returns (logit, {layer: (985,) cam}, conv contrib (985, 64), attention input)."""
    acts, grads, handles = capture(model)
    x = torch.tensor([encode(seq)], dtype=torch.long)
    logits.clear()
    model.zero_grad()
    model(x)
    # the logit, not the sigmoid: p saturates near 1.0 and its gradient vanishes
    logit = logits[0][0]
    logit.backward()
    for h in handles:
        h.remove()

    cams = {}
    for name in LAYERS:
        A = acts[name].detach().view(N_WIN, -1)
        G = grads[name].view(N_WIN, -1)
        if mode == 'gradcam':
            G = G.mean(0)  # one weight per channel: average gradient over all windows
        # gradxact keeps the per-window gradient: fc1 has a separate weight for every
        # window, so averaging it away (Grad-CAM) loses position-specific information
        contrib = A * G  # (985, C) how much each channel at each window pushes the logit
        cams[name] = contrib.sum(1).numpy()
        if name == 'conv':
            conv_contrib = contrib.numpy()
    return logit.item(), cams, conv_contrib, acts['pool'].detach()


@torch.no_grad()
def attention_map(model, x_att):
    """Recompute the attention probabilities (not returned by model.py), averaged over the 8 heads."""
    sa = model.selfattention
    q = sa.trans_to_multiple_heads(sa.query_layer(x_att))
    k = sa.trans_to_multiple_heads(sa.key_layer(x_att))
    scores = q @ k.transpose(-1, -2) / math.sqrt(sa.attention_head_size)
    return F.softmax(scores, dim=-1)[0].mean(0).numpy()  # (985, 985)


def to_residues(w, L):
    """Spread per-window scores (985, ...) onto the L real residues: mean over the windows covering each residue."""
    n = np.convolve(np.ones(N_WIN), np.ones(K))  # windows covering each of the 1000 padded positions
    spread = lambda col: np.convolve(col, np.ones(K)) / n
    out = np.apply_along_axis(spread, 0, w)
    return out[1000 - L:]


def bars(ax, y, label):
    ax.bar(np.arange(1, len(y) + 1), y, width=1.0,
           color=np.where(y > 0, 'tab:red', 'tab:blue'))
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel(label)


def plot_gradcam(seq, logit, cams, conv_contrib, title, mode, out_png):
    L = len(seq)
    fig, ax = plt.subplots(4, 1, figsize=(max(12, L / 25), 11), sharex=True,
                           gridspec_kw={'height_ratios': [1, 1, 1, 2.2]})
    for a, (name, desc) in zip(ax, LAYERS.items()):
        bars(a, to_residues(cams[name], L), f'{name}\n{mode}')
        a.set_title(desc, fontsize=9, loc='left')
    fig.suptitle(f'{title}  |  logit {logit:.2f}  (p={1 / (1 + np.exp(-logit)):.4f})  |  '
                 'red = pushes toward allergen, blue = against', y=0.92)

    per_filter = to_residues(conv_contrib, L).T  # (64, L)
    lim = np.abs(per_filter).max()
    im = ax[3].imshow(per_filter, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim,
                      interpolation='nearest', extent=(0.5, L + 0.5, 63.5, -0.5))
    ax[3].set_ylabel('conv filter')
    ax[3].set_xlabel('residue position')
    fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.03, pad=0.06,
                 label='per-filter contribution to the logit (conv layer)')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'saved {out_png}')

    offset = 1000 - L
    for name in LAYERS:
        real = cams[name][offset:]  # windows that start on a real residue
        top = np.argsort(-real)[:5]
        print(f'top windows [{name}] (1-based start):')
        for s in top:
            print(f'  {s + 1:4d}-{s + K:4d}  {seq[s:s + K]:16s}  {real[s]:+.3f}')
    totals = conv_contrib[offset:].sum(0)
    print('top conv filters:', ', '.join(f'#{f} ({totals[f]:+.2f})' for f in np.argsort(-totals)[:5]))


def plot_attention(seq, att, title, out_png):
    L = len(seq)
    offset = 1000 - L
    # share of each real window's attention that lands on windows starting in the padding
    pad_share = att[offset:, :offset].sum(1).mean()
    real = att[offset:, offset:]
    fig, ax = plt.subplots(figsize=(8, 7))
    n = real.shape[0]
    im = ax.imshow(real, cmap='viridis', interpolation='nearest',
                   extent=(0.5, n + 0.5, n + 0.5, 0.5))
    ax.set_xlabel('key window (start residue)')
    ax.set_ylabel('query window (start residue)')
    ax.set_title(f'{title}\nattention, mean of 8 heads  |  {pad_share:.0%} of attention goes to padding',
                 fontsize=10)
    fig.colorbar(im, ax=ax, label='attention probability')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'saved {out_png}')


def main():
    p = argparse.ArgumentParser(description='Grad-CAM and attention maps for DeepAlgPro')
    p.add_argument('--model', default='model.pt')
    p.add_argument('--fasta', help='FASTA to take the protein from (default: built-in sequence)')
    p.add_argument('--id', help='record id inside --fasta (default: first record)')
    p.add_argument('--mode', default='gradcam', choices=['gradcam', 'gradxact'],
                   help='gradcam: channel weights = mean gradient; gradxact: per-window gradient x activation')
    p.add_argument('-o', '--out', default='gradcam', help='output prefix')
    args = p.parse_args()

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
    seq = seq[:1000]
    if len(seq) < K:
        raise SystemExit(f'{name} is shorter than the {K}-residue conv window')

    model, logits = load_model(args.model)
    logit, cams, conv_contrib, x_att = gradcam(model, logits, seq, args.mode)
    title = f'{name} ({args.model})'
    plot_gradcam(seq, logit, cams, conv_contrib, title, args.mode, f'{args.out}.png')
    plot_attention(seq, attention_map(model, x_att), title, f'{args.out}.attention.png')


if __name__ == '__main__':
    main()
