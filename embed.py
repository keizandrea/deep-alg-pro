import argparse, os, re
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

AMBIGUOUS = re.compile(r'[BJOUZ]')

def clean(seq, maxlen):
    return AMBIGUOUS.sub('X', str(seq).upper())[:maxlen]

def load_plm(name, device):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()

if __name__ == '__main__':
    main()