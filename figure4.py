"""Figure 4 of He et al. 2023 (bbad246).

Three modes, following the paper's "Finding key regions" methods section:

  panels   Grad-CAM tracks for Gad m 1 / Fel d 1 / Ara h 1 with the known IgE
           epitopes marked below the axis                        -> panels F-H
  extract  the 15-mers (+-7 AA around every residue scoring above 80% of the
           per-protein max) that MEME takes as input             -> step 1 of A-E
  logos    stacked sequence logos for the best Tomtom matches between the
           epitope motifs and the Grad-CAM motifs                -> panels A-E

The paper uses standard Grad-CAM, so `--mode gradcam` is the faithful setting;
`--mode gradxact` is the position-aware variant (see gradcam.py).
"""
import argparse
import math
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from explore_model import load_model
from gradcam import K, gradcam, to_residues

# the three allergens of Figure 4F-H and their published linear IgE epitopes.
# positions are found by substring search, so they stay correct if the record changes.
PANELS = [
    ('F', 'allergen_473', 'Gad m 1', ['GDGKIGVDEFGAMIKA']),
    ('G', 'allergen_48', 'Fel d 1 (chain 2)', ['FAVANGNELLLDLS']),
    ('H', 'allergen_134', 'Ara h 1', ['AKSSPYQKKT', 'QEPDDLKQKA', 'EQEERGQRRW']),
]
PROTEIN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
# chemistry colouring, the convention WebLogo and MEME use for protein logos
AA_COLOR = {**{a: '#1f9e3f' for a in 'GSTYC'},      # polar
            **{a: '#9b59b6' for a in 'NQ'},         # neutral
            **{a: '#e74c3c' for a in 'KRH'},        # basic
            **{a: '#2e6fd9' for a in 'DE'},         # acidic
            **{a: '#2c2c2c' for a in 'AVLIPWFM'}}   # hydrophobic


def cam_track(model, logits, seq, mode='gradcam'):
    """Per-residue Grad-CAM weight, ReLU'd and scaled to [0, 1] as the paper does.

    ReLU is applied at the 985-window resolution (standard Grad-CAM keeps only
    evidence *for* the class), to_residues then spreads the windows onto the real
    residues, and dividing by the max makes "80% of the max value" simply
    `track > 0.8`.
    """
    logit, cams, _, _ = gradcam(model, logits, seq, mode)
    track = to_residues(np.maximum(cams['conv'], 0.0), len(seq))
    peak = track.max()
    return logit, (track / peak if peak > 0 else track)


def find_epitopes(seq, peptides):
    """Locate each peptide. Returns [(start, end, peptide)], 1-based inclusive."""
    spans = []
    for pep in peptides:
        i = seq.find(pep)
        if i < 0:
            print(f'  warning: epitope {pep} not found, skipped')
            continue
        spans.append((i + 1, i + len(pep), pep))
    return sorted(spans)


def panel_stats(track, spans, name):
    """The quantitative version of "the highest weights were in the epitope region"."""
    inside = np.zeros(len(track), dtype=bool)
    for s, e, _ in spans:
        inside[s - 1:e] = True
    top = int(track.argmax()) + 1
    hits = track > 0.8
    print(f'{name}: {len(track)} aa')
    print(f'  peak residue              {top}  ({"inside" if inside[top - 1] else "OUTSIDE"} an epitope)')
    print(f'  mean weight in epitopes   {track[inside].mean():.3f}')
    print(f'  mean weight elsewhere     {track[~inside].mean():.3f}')
    if hits.any():
        print(f'  residues above 0.8        {hits.sum()}, '
              f'{100 * (hits & inside).sum() / hits.sum():.0f}% inside an epitope')


def plot_panels(model, logits, fasta, mode, out_png):
    seqs = {r.id: str(r.seq).upper() for r in SeqIO.parse(fasta, 'fasta')}
    rows = [p for p in PANELS if p[1] in seqs]
    if not rows:
        raise SystemExit(f'none of {[p[1] for p in PANELS]} are in {fasta}')

    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 3.3 * len(rows)))
    axes = np.atleast_1d(axes)
    for ax, (letter, rec_id, name, peptides) in zip(axes, rows):
        seq = seqs[rec_id]
        logit, track = cam_track(model, logits, seq, mode)
        spans = find_epitopes(seq, peptides)
        panel_stats(track, spans, f'({letter}) {name}')

        x = np.arange(1, len(seq) + 1)
        ax.fill_between(x, track, color='#c0392b', alpha=0.85, lw=0)
        ax.plot(x, track, color='#7b241c', lw=0.6)
        ax.set_xlim(1, len(seq))
        ax.set_ylim(-0.62, 1.08)
        ax.set_yticks([0, 0.5, 0.8, 1.0])
        ax.axhline(0.8, color='gray', ls=':', lw=0.8)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_ylabel('Grad-CAM weight')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines['bottom'].set_position(('data', 0))
        p = 1 / (1 + math.exp(-logit))
        ax.set_title(f'({letter}) {name}   p(allergen) = {p:.4f}', loc='left', fontsize=11)

        # epitopes go BELOW the axis so they never hide the curve the figure argues about.
        # neighbouring epitopes alternate between two rows or their labels collide.
        for row, (s, e, pep) in enumerate(spans):
            y = -0.22 - (row % 2) * 0.18
            ax.axvspan(s, e, color='#2e6fd9', alpha=0.08, lw=0)
            ax.add_patch(Rectangle((s, y), e - s, 0.045, color='#2e6fd9',
                                   clip_on=False, lw=0))
            ax.text((s + e) / 2, y - 0.145, pep, ha='center', va='bottom', fontsize=7.5,
                    family='monospace', color='#1a4a99')
            if (e - s) / len(seq) > 0.06:   # wide enough to flank the sequence label
                ax.text(s, y - 0.072, str(s), ha='left', va='bottom', fontsize=6.5, color='#1a4a99')
                ax.text(e, y - 0.072, str(e), ha='right', va='bottom', fontsize=6.5, color='#1a4a99')
            else:                           # too narrow: the two numbers would overlap
                ax.text((s + e) / 2, y - 0.072, f'{s}-{e}', ha='center', va='bottom',
                        fontsize=6.5, color='#1a4a99')

    axes[-1].set_xlabel('residue position', labelpad=68)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f'saved {out_png}')


def extract_windows(model, logits, fasta, out_fa, mode, thresh, flank, min_p, limit):
    """Write every +-flank window around a residue scoring above `thresh` -> MEME input."""
    kept, skipped_wrong, skipped_short, n_prot = [], 0, 0, 0
    for rec in SeqIO.parse(fasta, 'fasta'):
        if not rec.id.startswith('allergen'):      # 'non-allergen_*' fails this, as in Train.py
            continue
        seq = str(rec.seq).upper()[:1000]
        if len(seq) < K:
            skipped_short += 1
            continue
        logit, track = cam_track(model, logits, seq, mode)
        if 1 / (1 + math.exp(-logit)) < min_p:     # explaining a wrong prediction teaches nothing
            skipped_wrong += 1
            continue
        n_prot += 1
        for i in np.flatnonzero(track > thresh):
            a, b = i - flank, i + flank + 1
            if a < 0 or b > len(seq):              # window would run off an end
                continue
            kept.append((f'{rec.id}_pos{i + 1}', seq[a:b]))
        if n_prot % 50 == 0:
            print(f'  {n_prot} proteins, {len(kept)} windows')
        if limit and n_prot >= limit:
            break

    width = 2 * flank + 1
    with open(out_fa, 'w', newline='\n') as fh:
        for name, s in kept:
            fh.write(f'>{name}\n{s}\n')
    print(f'saved {out_fa}')
    print(f'  {n_prot} allergens used, {skipped_wrong} skipped (predicted non-allergen), '
          f'{skipped_short} too short')
    print(f'  {len(kept)} windows of {width} aa, {len({s for _, s in kept})} of them distinct')
    print(f'  next: meme {out_fa} -protein -oc meme_gradcam -nmotifs 200 '
          f'-mod zoops -minw 6 -maxw {width}')


def parse_meme(path):
    """MEME text output -> ({motif id: probability matrix}, alphabet)."""
    text = open(path, encoding='utf-8', errors='replace').read()
    m = re.search(r'ALPHABET=?\s*["\']?([A-Z]{18,25})', text)
    alphabet = m.group(1) if m else PROTEIN_ALPHABET
    motifs = {}
    for hit in re.finditer(r'^MOTIF\s+(\S+)(?:\s+(\S+))?', text, re.M):
        tail = text[hit.end():]
        head = re.search(r'letter-probability matrix:[^\n]*?w=\s*(\d+)', tail)
        if not head:
            continue
        want, rows = int(head.group(1)), []
        for line in tail[head.end():].split('\n')[1:]:
            vals = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
            if len(vals) != len(alphabet):
                break
            rows.append([float(v) for v in vals])
            if len(rows) == want:
                break
        if rows:
            ppm = np.array(rows)
            for key in filter(None, (hit.group(1), hit.group(2))):
                motifs[key] = ppm
    return motifs, alphabet


def _letter(ax, ch, x, y, w, h):
    tp = TextPath((0, 0), ch, size=1, prop=FontProperties(family='DejaVu Sans', weight='bold'))
    bb = tp.get_extents()
    if bb.width <= 0 or bb.height <= 0 or h <= 0:
        return
    tr = (Affine2D().translate(-bb.x0, -bb.y0)
          .scale(w / bb.width, h / bb.height).translate(x, y))
    ax.add_patch(PathPatch(tr.transform_path(tp), fc=AA_COLOR.get(ch, '#555555'), ec='none'))


def draw_logo(ax, ppm, alphabet, x0=0.0, y0=0.0):
    """Information-content sequence logo: column height = log2(20) - entropy."""
    for j, col in enumerate(ppm):
        p = np.clip(col, 1e-9, 1.0)
        ic = max(math.log2(len(alphabet)) + float((p * np.log2(p)).sum()), 0.0)
        y = y0
        for idx in np.argsort(col):                 # smallest letter at the bottom
            h = col[idx] * ic
            if h < 0.02:
                continue
            _letter(ax, alphabet[idx], x0 + j + 0.06, y, 0.88, h)
            y += h


def plot_logos(tsv, epitope_meme, gradcam_meme, top, pmax, out_png):
    import pandas as pd
    df = pd.read_csv(tsv, sep='\t', comment='#')
    df.columns = [c.strip() for c in df.columns]
    pcol = next(c for c in df.columns if c.lower() in ('p-value', 'p_value', 'pvalue'))
    offcol = next((c for c in df.columns if 'offset' in c.lower()), None)
    df = df[df[pcol] < pmax].sort_values(pcol).head(top)
    if df.empty:
        raise SystemExit(f'no Tomtom match with {pcol} < {pmax} in {tsv}')

    q_motifs, alphabet = parse_meme(epitope_meme)
    t_motifs, _ = parse_meme(gradcam_meme)
    maxic = math.log2(len(alphabet))
    gap = 0.5

    fig, axes = plt.subplots(len(df), 1, figsize=(7.5, 3.0 * len(df)))
    axes = np.atleast_1d(axes)
    for ax, letter, (_, row) in zip(axes, 'ABCDEFGH', df.iterrows()):
        q = q_motifs.get(str(row['Query_ID']))
        t = t_motifs.get(str(row['Target_ID']))
        if q is None or t is None:
            ax.text(0.5, 0.5, f"motif {row['Query_ID']} / {row['Target_ID']} "
                              'not found in the MEME output', ha='center', fontsize=8,
                    transform=ax.transAxes)
            ax.axis('off')
            continue
        # Tomtom's optimal offset positions the target relative to the query
        shift = int(row[offcol]) if offcol else 0
        qx, tx = max(0, -shift), max(0, shift)
        draw_logo(ax, q, alphabet, x0=qx, y0=maxic + gap)   # top    = epitope motif
        draw_logo(ax, t, alphabet, x0=tx, y0=0.0)           # bottom = Grad-CAM motif
        ax.axhline(maxic + gap / 2, color='#bbbbbb', lw=0.7)
        ax.set_xlim(0, max(qx + len(q), tx + len(t)))
        ax.set_ylim(0, 2 * maxic + gap)
        ax.set_yticks([maxic / 2, maxic + gap + maxic / 2])
        ax.set_yticklabels(['Grad-CAM', 'epitope'], fontsize=8)
        ax.set_xticks([])
        ax.set_title(f'({letter})  epitope {row["Query_ID"]}  vs  '
                     f'Grad-CAM {row["Target_ID"]}', loc='left', fontsize=9)
        ax.set_xlabel(f'Tomtom p = {row[pcol]:.2e}', fontsize=9)
        ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f'saved {out_png}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('what', choices=['panels', 'extract', 'logos'])
    p.add_argument('--weights', default='model.pt')
    p.add_argument('--fasta', default='data/all.test.fasta')
    p.add_argument('--mode', default='gradcam', choices=['gradcam', 'gradxact'])
    p.add_argument('-o', '--out')
    # extract
    p.add_argument('--thresh', type=float, default=0.8, help='fraction of the per-protein max')
    p.add_argument('--flank', type=int, default=7, help='residues each side -> 2*flank+1 mer')
    p.add_argument('--min-p', type=float, default=0.5, help='skip allergens predicted below this')
    p.add_argument('--limit', type=int, help='stop after N allergens (for a quick test)')
    # logos
    p.add_argument('--tomtom', default='tomtom_out/tomtom.tsv')
    p.add_argument('--epitope-meme', default='meme_epitopes/meme.txt')
    p.add_argument('--gradcam-meme', default='meme_gradcam/meme.txt')
    p.add_argument('--top', type=int, default=5)
    p.add_argument('--pmax', type=float, default=1e-3)
    args = p.parse_args()

    if args.what == 'logos':
        plot_logos(args.tomtom, args.epitope_meme, args.gradcam_meme,
                   args.top, args.pmax, args.out or 'figure4ae.png')
        return

    model, logits = load_model(args.weights)
    if args.what == 'panels':
        plot_panels(model, logits, args.fasta, args.mode, args.out or 'figure4fh.png')
    else:
        extract_windows(model, logits, args.fasta, args.out or 'gradcam_windows.fa',
                        args.mode, args.thresh, args.flank, args.min_p, args.limit)


if __name__ == '__main__':
    main()
