"""BERT on WikiText — encoder activations in memory.

Minimal script form of examples/text/01_distilbert_wikitext_memory.ipynb.
Downloads on first run: WikiText-2 (~5 MB) and BERT weights (~440 MB).

Run: python examples/scripts/01_distilbert_wikitext_memory.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from transformers import AutoModelForTokenClassification, AutoTokenizer

from examples._utils.data import activation_loader
from examples._utils.text import WikiTextSamples
from nnact import ActivationPipeline


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased")
    dataset = WikiTextSamples(tokenizer, n=32, max_length=64)
    print(f"{len(dataset)} passages | input_ids {tuple(dataset[0]['input_ids'].shape)}")

    # run_dir is required for every run: it holds run.log and run_metadata.json.
    pipeline = ActivationPipeline(
        model,
        ["bert.encoder.layer.5"],
        output_type="token",
        tokenizer=tokenizer,
        run_dir=Path("runs") / "01_distilbert_wikitext_memory",
    )
    loader = activation_loader(dataset, batch_size=16)
    activations = pipeline.run(loader)
    print(activations.summary())


if __name__ == "__main__":
    main()
