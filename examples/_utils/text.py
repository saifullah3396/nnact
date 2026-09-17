"""Text datasets used by activation examples."""

from typing import Any

from torch.utils.data import Dataset


class WikiTextSamples(Dataset[dict[str, Any]]):
    """WikiText passages tokenized to fixed-length model inputs."""

    def __init__(self, tokenizer: Any, n: int = 256, max_length: int = 64) -> None:
        from datasets import load_dataset

        raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        self.texts = [
            text.strip()
            for text in raw["text"]
            if len(text.strip()) > 120 and not text.strip().startswith("=")
        ][:n]
        self.encoded = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "input_ids": self.encoded["input_ids"][idx],
            "attention_mask": self.encoded["attention_mask"][idx],
        }
