import argparse, os, re, glob
import numpy as np, torch
from Bio import SeqIO
from transformers import AutoModel, AutoTokenizer

def main():
    p = argparse.ArgumentParser(description='Precompute PLM embeddings for DeepAlgPro')
    p.add_argument('--fasta', required=True,
                   help='input FASTA, e.g. data/all.train.fasta')
    p.add_argument('--out', required=True,
                   help='output directory for the .npy files')
    p.add_argument('--plm', default='facebook/esm2_t30_150M_UR50D',
                   help='swapping encoders without editing code (HuggingFace model id)')
    p.add_argument('--batch', default=8, type=int,
                   help='tune down if hit CUDA OOM')
    p.add_argument('--maxlen', default=1000, type=int,
                   help='to match DeepAlgPro fixed input length')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)
    tok, plm, is_t5 = load_plm(args.plm, device)

    records = []
    for rec in SeqIO.parse(args.fasta, 'fasta'):
        s = clean(rec.seq, args.maxlen)
        if s:
            records.append((rec.id, s))

    records.sort(key=lambda r:len(r[1]))
    print(f'{len(records)} seqs, length {len(records[0][1])}-{len(records[-1][1])}')

    done = skipped = 0
    for start in range(0, len(records), args.batch):
        chunk = records[start:start + args.batch]
        ids   = [c[0] for c in chunk]
        seqs  = [c[1] for c in chunk]
        paths = [os.path.join(args.out, i + '.npy') for i in ids]

        if all(os.path.exists(p) for p in paths):
            skipped += len(chunk)
            continue

        batch_in = [' '.join(s) for s in seqs] if is_t5 else seqs
        offset   = 0 if is_t5 else 1
        enc = tok(batch_in, add_special_tokens=True, padding=True, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = plm(**enc).last_hidden_state

        for j, s in enumerate(seqs):
            L = len(s)
            emb = out[j, offset:offset + L].float().cpu().numpy()
            assert emb.shape[0] == L, (ids[j], emb.shape, L)
            assert np.isfinite(emb).all(), f'non-finite in {ids[j]}'
            np.save(paths[j], emb.astype(np.float16))
            done += 1

        if (start // args.batch) % 20 == 0:
            print(f'{done + skipped}/{len(records)}', flush=True)
    files = glob.glob(os.path.join(args.out, '*.npy'))
    dim = np.load(files[0]).shape[1] if files else 0
    mb  = sum(os.path.getsize(f) for f in files) / 1e6
    print(f'\nwrote {done}, skipped {skipped}, {len(files)} files, {mb:.0f} MB')
    print(f'dim = {dim}  ->  train with --in-channels {dim}')


AMBIGUOUS = re.compile(r'[BJOUZ]')

def clean(seq, maxlen):
    return AMBIGUOUS.sub('X', str(seq).upper())[:maxlen]


def load_plm(name, device):
    is_t5 = 't5' in name.lower()
    if is_t5:
        from transformers import T5EncoderModel, T5Tokenizer
        tok = T5Tokenizer.from_pretrained(name, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(name)
    else:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
    if device.type == 'cuda':
        model = model.half()
    return tok, model.to(device).eval(), is_t5

if __name__ == '__main__':
    main()