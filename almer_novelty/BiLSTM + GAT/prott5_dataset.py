import hashlib
import re
import torch
from torch.utils.data import Dataset
from pathlib import Path
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel

from phys_evo_features import extract_combined_phys_evo_vector


def resolve_data_dir() -> Path:
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir.parent, current_dir.parent.parent]:
        candidate = parent / "data"
        if candidate.exists() and candidate.is_dir():
            return candidate
    return Path(r"D:\DeepAlgPro\data")


def build_bidirectional_edges(seq_len: int) -> torch.Tensor:
    if seq_len <= 1:
        return torch.tensor([[0], [0]], dtype=torch.long)
    src = list(range(seq_len - 1)) + list(range(1, seq_len))
    dst = list(range(1, seq_len)) + list(range(seq_len - 1))
    return torch.tensor([src, dst], dtype=torch.long)


class ProtT5FeatureExtractor:
    def __init__(
        self,
        model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc",
        device: torch.device = None,
        concat_last_3_layers: bool = False,
        cache_dir: Path = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.concat_last_3_layers = concat_last_3_layers
        self.model_name = model_name
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = None
        self.model = None

    def _lazy_load_model(self):
        if self.model is None:
            print(f"Loading ProtT5 model: {self.model_name} on {self.device}...")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained(
                self.model_name,
                output_hidden_states=self.concat_last_3_layers,
            )
            if self.device.type == "cuda":
                self.model = self.model.half()
            self.model = self.model.to(self.device)
            self.model.eval()

    def unload_from_gpu(self):
        """Frees ~2.4 GB of VRAM on RTX 5050 once embeddings are cached."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.no_grad()
    def extract_node_features(self, sequence: str, rec_id: str = "") -> torch.Tensor:
        if self.cache_dir is not None:
            tag = "3L" if self.concat_last_3_layers else "1L"
            seq_hash = hashlib.md5(sequence.encode("utf-8")).hexdigest()[:10]
            safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", rec_id)
            cache_file = self.cache_dir / f"{safe_id}_{seq_hash}_{tag}.pt"
            if cache_file.exists():
                return torch.load(cache_file, map_location="cpu", weights_only=True).float()

        self._lazy_load_model()
        seq_len = len(sequence)
        cleaned_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence.upper())))

        encoded = self.tokenizer(
            cleaned_seq,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.concat_last_3_layers:
            last_three = outputs.hidden_states[-3:]
            token_repr = torch.cat(last_three, dim=-1)
        else:
            token_repr = outputs.last_hidden_state

        node_features = token_repr[0, :seq_len, :].float().cpu()

        if self.cache_dir is not None:
            # Save in float16 to cut disk space in half (~50% smaller)
            torch.save(node_features.half(), cache_file)

        return node_features


class AllergenProteinGraphDataset(Dataset):
    """
    Pre-computes the 227-dim Physicochemical + Evolutionary descriptors in RAM
    at initialization (eliminating slow FASTA re-parsing in the training loop)
    and fetches cached/extracted ProtT5 node features X and edges E.
    """
    def __init__(
        self,
        fasta_path: Path,
        extractor: ProtT5FeatureExtractor = None,
        max_len: int = 1000,
    ):
        self.fasta_path = Path(fasta_path)
        self.extractor = extractor
        self.records = []
        self.desc_Lookup = []

        for record in SeqIO.parse(str(self.fasta_path), "fasta"):
            seq_str = str(record.seq).strip()
            if 0 < len(seq_str) <= max_len:
                label = 1.0 if record.id.startswith("allergen") else 0.0
                self.records.append((record.id, seq_str, label))
                self.desc_Lookup.append(extract_combined_phys_evo_vector(seq_str))

        print(
            f"Loaded {len(self.records)} sequences (<= {max_len} aa) & precomputed 227-dim "
            f"Phys+Evo vectors from {self.fasta_path.name}"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec_id, seq_str, label = self.records[idx]
        x = self.extractor.extract_node_features(seq_str, rec_id=rec_id)
        edge_index = build_bidirectional_edges(len(seq_str))
        y = torch.tensor([label], dtype=torch.float32)
        desc = self.desc_Lookup[idx]  # [227] pre-computed tensor
        return x, edge_index, desc, y, rec_id