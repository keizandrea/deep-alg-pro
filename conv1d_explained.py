"""
Step-by-step, hand-written version of the first layers of convATTnet:

    one_hot  ->  Conv1d  ->  ReLU  ->  permute  ->  MaxPool1d

Part 1 runs a tiny example you can check by hand on paper.
Part 2 runs the hand-written functions on a real protein, using the real
trained weights in model.pt, and checks they match PyTorch exactly.

Run:  python conv1d_explained.py
"""
import os
import torch
import torch.nn.functional as F
from model import convATTnet


# ---------------------------------------------------------------------------
# The functions
# ---------------------------------------------------------------------------

def one_hot_manual(codes, num_classes):
    """
    codes       : (B, L) integer tensor, each value in 0..num_classes-1
    num_classes : number of rows (channels) in the one-hot grid
    returns     : (B, num_classes, L) float tensor of 0s and 1s
                  (already channels-first, like the model's x.permute(0, 2, 1))
    """
    B, L = codes.shape
    out = torch.zeros(B, num_classes, L)
    for b in range(B):
        for pos in range(L):
            aa = codes[b, pos]          # which amino acid is at this position
            out[b, aa, pos] = 1.0       # put a 1 in that amino acid's row
    return out


def conv1d_manual(x, weight, bias, stride=1):
    """
    x      : (B, C_in, L)        input, e.g. (B, 21, 1000)
    weight : (C_out, C_in, K)    one (C_in x K) filter per output channel, e.g. (64, 21, 16)
    bias   : (C_out,)            one number per filter, e.g. (64,)
    stride : how many positions the window moves each step

    returns: (B, C_out, L_out) with L_out = (L - K) // stride + 1, e.g. 985
    """
    B, C_in, L = x.shape
    C_out, C_in_w, K = weight.shape
    assert C_in == C_in_w, "filter must have as many rows as the input has channels"

    L_out = (L - K) // stride + 1
    out = torch.zeros(B, C_out, L_out)

    for b in range(B):                      # each protein in the batch
        for f in range(C_out):              # each filter (motif detector)
            for t in range(L_out):          # each window position
                start = t * stride
                window = x[b, :, start:start + K]       # (C_in, K) slice of the protein
                score = (window * weight[f]).sum()      # multiply element-wise, add everything
                out[b, f, t] = score + bias[f]
    return out


def relu_manual(x):
    """Negative numbers -> 0, positive numbers unchanged."""
    return torch.where(x > 0, x, torch.zeros_like(x))


def maxpool_last_dim_manual(x, kernel, stride):
    """
    Max-pool over the LAST dimension, exactly what nn.MaxPool1d does.
    x : (B, L, D)  -> returns (B, L, D_out) with D_out = (D - kernel) // stride + 1
    In convATTnet, x is (B, 985, 64) so this pools groups of 5 filters -> 12.
    """
    B, L, D = x.shape
    D_out = (D - kernel) // stride + 1
    out = torch.zeros(B, L, D_out)
    for g in range(D_out):
        start = g * stride
        out[:, :, g] = x[:, :, start:start + kernel].max(dim=-1).values
    return out


# ---------------------------------------------------------------------------
# Part 1: tiny example (check it by hand)
# ---------------------------------------------------------------------------

def tiny_example():
    print("=" * 70)
    print("PART 1: tiny example")
    print("=" * 70)

    # Reduced alphabet: 0 = pad, 1 = A, 2 = K, 3 = M   (4 channels instead of 21)
    names = ["pad", "A", "K", "M"]
    # Protein "MKA", left-padded to length 5 (the real model pads to 1000)
    codes = torch.tensor([[0, 0, 3, 2, 1]])
    print("codes (B=1, L=5):", codes.tolist(), "  = pad pad M K A")

    x = one_hot_manual(codes, num_classes=4)          # (1, 4, 5)
    print("\none-hot x, shape", tuple(x.shape), " rows = channels, cols = positions")
    for r in range(4):
        print(f"  {names[r]:>3} ", x[0, r].int().tolist())

    # 2 filters, each 4 rows (channels) x 3 columns (kernel_size = 3)
    weight = torch.tensor([
        # filter 0: rewards "M K A" in that order, penalises padding
        [[-1., -1., -1.],   # pad
         [ 0.,  0.,  2.],   # A  wanted in slot 3
         [ 0.,  2.,  0.],   # K  wanted in slot 2
         [ 2.,  0.,  0.]],  # M  wanted in slot 1
        # filter 1: counts padding in the window
        [[ 1.,  1.,  1.],   # pad
         [ 0.,  0.,  0.],
         [ 0.,  0.,  0.],
         [ 0.,  0.,  0.]],
    ])                                                # (C_out=2, C_in=4, K=3)
    bias = torch.tensor([0., -1.])                    # (2,)

    out = conv1d_manual(x, weight, bias, stride=1)    # (1, 2, 5-3+1=3)
    print("\nconv output, shape", tuple(out.shape), " L_out = 5 - 3 + 1 = 3")
    for f in range(2):
        print(f"  filter {f}:", out[0, f].tolist())

    print("\n  how filter 0, window 3 (positions 3-5 = M K A) is computed:")
    print("    slot1 is M -> weight[M][0] = 2")
    print("    slot2 is K -> weight[K][1] = 2")
    print("    slot3 is A -> weight[A][2] = 2")
    print("    2 + 2 + 2 + bias 0 = 6   <- strong match")

    r = relu_manual(out)
    print("\nafter ReLU:")
    for f in range(2):
        print(f"  filter {f}:", r[0, f].tolist())

    ref = F.relu(F.conv1d(x, weight, bias))
    print("\nmatches PyTorch F.conv1d + relu:", torch.allclose(r, ref))


# ---------------------------------------------------------------------------
# Part 2: the real layer with the trained weights
# ---------------------------------------------------------------------------

codeadict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
             'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}


def real_example():
    print("\n" + "=" * 70)
    print("PART 2: real conv1 layer from model.pt")
    print("=" * 70)

    model = convATTnet()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    conv = model.conv1
    print("nn.Conv1d parameters:")
    print("  in_channels  =", conv.in_channels, "  (20 amino acids + padding)")
    print("  out_channels =", conv.out_channels, "  (number of filters)")
    print("  kernel_size  =", conv.kernel_size[0], "  (window width in residues)")
    print("  stride       =", conv.stride[0])
    print("  weight shape =", tuple(conv.weight.shape), "=", conv.weight.numel(), "numbers")
    print("  bias shape   =", tuple(conv.bias.shape))

    seq = ("MAAHTILASAPSHTTFSLISPFSSTPTNALSSSLQSSSFNGLSFKLSPTTQSLSLSTSAASKPLTIVAATKKAV"
           "AVLKGTSNVEGVVTLTQEDDGPTTVNVRISGLAPGKHGFHLHEFGDTTNGCMSTGPHFNPDKKTHGAPEDEVRH")
    codes = torch.tensor([[0] * (1000 - len(seq)) + [codeadict[a] for a in seq]])

    with torch.no_grad():
        x = one_hot_manual(codes, 21)                              # (1, 21, 1000)
        c = relu_manual(conv1d_manual(x, conv.weight, conv.bias))  # (1, 64, 985)
        p = maxpool_last_dim_manual(c.permute(0, 2, 1), 5, 5)      # (1, 985, 12)

        # The same steps the way model.forward does them
        x_ref = F.one_hot(codes, 21).float().permute(0, 2, 1)
        c_ref = F.relu(conv(x_ref))
        p_ref = model.maxpool(c_ref.permute(0, 2, 1))

    print("\nshapes:")
    print("  codes     ", tuple(codes.shape))
    print("  one-hot   ", tuple(x.shape))
    print("  conv+relu ", tuple(c.shape))
    print("  permute   ", tuple(c.permute(0, 2, 1).shape))
    print("  maxpool   ", tuple(p.shape))

    print("\nmanual == PyTorch?")
    print("  one-hot  :", torch.equal(x, x_ref))
    print("  conv+relu:", torch.allclose(c, c_ref, atol=1e-5))
    print("  maxpool  :", torch.allclose(p, p_ref, atol=1e-5))

    last = c.shape[-1] - 1
    print(f"\nlast window (residues {len(seq) - 15}..{len(seq)} of the protein), first 8 filter scores:")
    print("  ", [round(v, 3) for v in c[0, :8, last].tolist()])
    print("same position after pooling (12 numbers):")
    print("  ", [round(v, 3) for v in p[0, last].tolist()])


if __name__ == "__main__":
    tiny_example()
    real_example()
