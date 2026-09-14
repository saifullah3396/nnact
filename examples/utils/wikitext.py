"""Shared WikiText-2 loading for the text notebooks."""

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from nnact import Sample


class WikiTextSamples(Dataset[Sample]):
    """WikiText passages, tokenised to a fixed length.

    nnact stacks Sample.data across a batch, so every row must be the same
    shape: pad to a fixed max_length rather than per batch. Attention masks
    are kept on the instance for pooling later.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, n: int = 256, max_length: int = 64
    ) -> None:
        raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        # Drop blanks and the "= Heading =" lines the raw dump is full of.
        self.texts = [
            t.strip()
            for t in raw["text"]
            if len(t.strip()) > 120 and not t.strip().startswith("=")
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

    def __getitem__(self, idx: int) -> Sample:
        return Sample(id=f"wiki_{idx:04d}", data=self.encoded["input_ids"][idx])
