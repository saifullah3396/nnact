"""BERT on WikiText — encoder activations in memory.

Minimal script form of examples/text/01_distilbert_wikitext_memory.ipynb.
Downloads on first run: WikiText-2 (~5 MB) and BERT weights (~440 MB).

Run: python examples/scripts/01_distilbert_wikitext_memory.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from transformers import AutoModelForTokenClassification, AutoTokenizer

from examples._utils.text import WikiTextSamples
from nnact import ActivationPipeline


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased")
    dataset = WikiTextSamples(tokenizer=tokenizer, n=32, max_length=64)
    print(
        f"{len(dataset)} passages | input_ids "
        f"{tuple(dataset[0].model_input.input_ids.shape)}"
    )

    pipeline = ActivationPipeline(
        model=model,
        layer_names=["bert.encoder.layer.5"],
        output_type="token",
        tokenizer=tokenizer,
    )
    run_result = pipeline.run(dataset=dataset, batch_size=16)
    print(run_result.dataset.summary())


if __name__ == "__main__":
    main()
