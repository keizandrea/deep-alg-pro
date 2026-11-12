import argparse
import torch
import pandas as pd
from pathlib import Path

from prott5_dataset import resolve_data_dir, ProtT5FeatureExtractor, AllergenProteinGraphDataset
from fusion_model import MultimodalAllergenModel, LabelSmoothedFocalLoss
from train_full_model import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Test Full Multimodal BiLSTM + GAT Model (model.pt)")
    parser.add_argument("-i", "--input", type=str, default="", help="Path to test FASTA file")
    parser.add_argument("-m", "--model-path", type=str, default="model.pt", help="Path to saved .pt weights")
    parser.add_argument("-o", "--output", type=str, default="test_predictions.tsv", help="Output TSV file")
    parser.add_argument("--concat-3-layers", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent
    fasta_path = Path(args.input) if args.input else (resolve_data_dir() / "all.test.fasta")
    model_path = Path(args.model_path) if Path(args.model_path).is_absolute() else (base_dir / args.model_path)

    extractor = ProtT5FeatureExtractor(
        device=device,
        concat_last_3_layers=args.concat_3_layers,
        cache_dir=base_dir / "cache_prott5",
    )
    prot_dim = 3072 if args.concat_3_layers else 1024

    test_ds = AllergenProteinGraphDataset(fasta_path, extractor=extractor)
    model = MultimodalAllergenModel(prot_dim=prot_dim, desc_dim=227, hidden_dim=192).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)
    model.eval()

    criterion_cls = LabelSmoothedFocalLoss(alpha=0.5, gamma=1.0, smoothing=0.1)
    metrics = evaluate_model(model, test_ds, criterion_cls, device)

    result_line = (
        f"Test Result | ACC: {metrics['acc']:.5f}\tF1: {metrics['f1']:.5f}\t"
        f"Prec: {metrics['prec']:.5f}\tRecall: {metrics['recall']:.5f}\t"
        f"AUROC: {metrics['auc']:.5f}\tMCC: {metrics['mcc']:.5f}"
    )
    print(result_line)
    with open(base_dir / "test.log", "a") as f:
        f.write(result_line + "\n")

    # Export per-protein probability scores
    records = []
    with torch.no_grad():
        for idx in range(len(test_ds)):
            x, edge_index, desc, y, rec_id = test_ds[idx]
            logits, _, _ = model(x.to(device), edge_index.to(device), desc.to(device).unsqueeze(0))
            prob = torch.sigmoid(logits).item()
            pred_label = "allergenicity" if prob >= 0.5 else "non-allergenicity"
            records.append([rec_id, int(y.item()), round(prob, 6), pred_label])

    df = pd.DataFrame(records, columns=["protein", "true_label", "score", "predict_result"])
    df.to_csv(base_dir / args.output, sep="\t", index=False)
    print(f"Saved detailed predictions to {base_dir / args.output}")


if __name__ == "__main__":
    main()