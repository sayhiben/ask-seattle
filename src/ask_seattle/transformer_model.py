from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ask_seattle import __version__
from ask_seattle.data import LabeledPost, post_text

TRANSFORMER_METADATA_FILE = "ask_seattle_model.json"
DEFAULT_BASE_MODEL = "distilbert/distilbert-base-uncased"
DEFAULT_TRANSFORMER_PRESET = "distilbert"
ID2LABEL = {0: "not_askseattle", 1: "askseattle"}
LABEL2ID = {"not_askseattle": 0, "askseattle": 1}


@dataclass(frozen=True)
class TransformerPreset:
    name: str
    base_model: str
    tier: str
    max_length: int
    description: str
    notes: str


TRANSFORMER_PRESETS: dict[str, TransformerPreset] = {
    "distilbert": TransformerPreset(
        name="distilbert",
        base_model="distilbert/distilbert-base-uncased",
        tier="default",
        max_length=256,
        description="Small, stable baseline transformer for local text classification.",
        notes="Good first transformer because it is well-supported and cheaper than base-size encoders.",
    ),
    "deberta-v3-small": TransformerPreset(
        name="deberta-v3-small",
        base_model="microsoft/deberta-v3-small",
        tier="benchmark",
        max_length=256,
        description="Higher-quality small encoder candidate for the main benchmark set.",
        notes="Often a strong NLU tradeoff; benchmark local CPU latency before making it active.",
    ),
    "roberta-base": TransformerPreset(
        name="roberta-base",
        base_model="FacebookAI/roberta-base",
        tier="benchmark",
        max_length=256,
        description="Stable base-size encoder for comparison against DistilBERT and DeBERTa.",
        notes="Likely heavier than DistilBERT; useful as a known-quality reference point.",
    ),
    "electra-small": TransformerPreset(
        name="electra-small",
        base_model="google/electra-small-discriminator",
        tier="benchmark",
        max_length=256,
        description="Cheap transformer candidate for efficient local inference.",
        notes="Worth testing when throughput matters and DistilBERT recall is not enough.",
    ),
    "bertweet-base": TransformerPreset(
        name="bertweet-base",
        base_model="vinai/bertweet-base",
        tier="social-text",
        max_length=256,
        description="Social-media-pretrained encoder candidate for informal post phrasing.",
        notes="Twitter is not Reddit; include only after the main benchmark set has real labels.",
    ),
    "twitter-roberta-base": TransformerPreset(
        name="twitter-roberta-base",
        base_model="cardiffnlp/twitter-roberta-base",
        tier="social-text",
        max_length=256,
        description="RoBERTa variant pretrained on Twitter-like text.",
        notes="Use the base checkpoint, not a sentiment-finetuned checkpoint.",
    ),
    "deberta-v3-base": TransformerPreset(
        name="deberta-v3-base",
        base_model="microsoft/deberta-v3-base",
        tier="quality",
        max_length=256,
        description="Larger DeBERTa candidate when small models are close but insufficient.",
        notes="More expensive; do not use for default local shadow inference without latency data.",
    ),
    "modernbert-base": TransformerPreset(
        name="modernbert-base",
        base_model="answerdotai/ModernBERT-base",
        tier="long-context",
        max_length=512,
        description="Modern encoder candidate for longer text classification.",
        notes="Requires transformers>=4.48.0; raise max_length only after proving truncation hurts.",
    ),
    "bigbird-roberta-base": TransformerPreset(
        name="bigbird-roberta-base",
        base_model="google/bigbird-roberta-base",
        tier="long-context",
        max_length=1024,
        description="Long-sequence candidate for posts where body truncation is a real issue.",
        notes="Specialized and heavier; not part of the default benchmark group.",
    ),
}

DEFAULT_TRANSFORMER_BENCHMARK_PRESETS = (
    "distilbert",
    "deberta-v3-small",
    "roberta-base",
    "electra-small",
)


def resolve_transformer_preset(name: str) -> TransformerPreset:
    try:
        return TRANSFORMER_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(TRANSFORMER_PRESETS))
        raise ValueError(f"Unknown transformer preset {name!r}; available presets: {available}") from exc


def transformer_preset_rows() -> list[dict[str, Any]]:
    return [asdict(preset) for preset in TRANSFORMER_PRESETS.values()]


def train_transformer_model(
    train_posts: list[LabeledPost],
    validation_posts: list[LabeledPost],
    output_dir: str | Path,
    *,
    threshold: float,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    preset_name: str | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise SystemExit(
            "Install transformer support with: python -m pip install -e '.[transformer]'"
        ) from exc

    class PostDataset(Dataset):
        def __init__(self, posts: list[LabeledPost]) -> None:
            self.texts = [post_text(post.title, post.selftext) for post in posts]
            self.labels = [post.label for post in posts]

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {"text": self.texts[index], "label": self.labels[index]}

    device = torch.device(device_name or select_torch_device(torch))
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [item["text"] for item in batch],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return {key: value.to(device) for key, value in encoded.items()}

    train_loader = DataLoader(
        PostDataset(train_posts),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            optimizer.step()

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(artifact_dir, safe_serialization=False)
    tokenizer.save_pretrained(artifact_dir)

    metadata = {
        "model_type": "transformer",
        "model_name": "transformer_sequence_classifier",
        "model_version": __version__,
        "preset": preset_name,
        "base_model": base_model,
        "threshold": threshold,
        "max_length": max_length,
        "train_examples": len(train_posts),
        "validation_examples": len(validation_posts),
        "device": str(device),
        "version": __version__,
    }
    (artifact_dir / TRANSFORMER_METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_transformer_bundle(path: str | Path) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise SystemExit(
            "Install transformer support with: python -m pip install -e '.[transformer]'"
        ) from exc

    model_dir = Path(path)
    metadata_path = model_dir / TRANSFORMER_METADATA_FILE
    if not metadata_path.exists():
        raise ValueError(f"{model_dir} is missing {TRANSFORMER_METADATA_FILE}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    device = torch.device(os.getenv("ASK_SEATTLE_TORCH_DEVICE") or select_torch_device(torch))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    return {
        **metadata,
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
    }


def update_transformer_metadata(path: str | Path, updates: dict[str, Any]) -> dict[str, Any]:
    metadata_path = Path(path) / TRANSFORMER_METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(updates)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def transformer_positive_probabilities(bundle: dict[str, Any], texts: list[str]) -> list[float]:
    try:
        import torch
    except Exception as exc:
        raise SystemExit(
            "Install transformer support with: python -m pip install -e '.[transformer]'"
        ) from exc

    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]
    max_length = int(bundle.get("max_length") or 256)
    probabilities: list[float] = []

    for start in range(0, len(texts), int(bundle.get("inference_batch_size") or 16)):
        batch_texts = texts[start : start + int(bundle.get("inference_batch_size") or 16)]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            batch_probabilities = torch.softmax(logits, dim=-1)[:, LABEL2ID["askseattle"]]
        probabilities.extend(float(value) for value in batch_probabilities.cpu().tolist())

    return probabilities


def select_torch_device(torch_module: Any) -> str:
    override = os.getenv("ASK_SEATTLE_TORCH_DEVICE")
    if override:
        return override
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"
