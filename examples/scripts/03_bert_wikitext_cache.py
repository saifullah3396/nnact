"""BERT on WikiText — encoder activations, cached to HDF5.

Minimal script form of examples/text/03_bert_wikitext_cache.ipynb. Same run
as 01_distilbert_wikitext_memory.py, but with cache_outputs=True so
activations stream to an HDF5 file instead of staying in RAM.

Run: python examples/scripts/03_bert_wikitext_cache.py
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

    dataset = WikiTextSamples(tokenizer, n=32, max_length=64)
    print(
        f"{len(dataset)} passages | input_ids "
        f"{tuple(dataset[0].model_input.input_ids.shape)}"
    )

    pipeline = ActivationPipeline(
        model,
        ["bert.encoder.layer.5"],
        output_type="token",
        tokenizer=tokenizer,
        cache_outputs=True,
        run_dir=Path("runs") / "03_bert_wikitext_cache",
    )
    activations = pipeline.run(dataset, batch_size=16)
    print(activations.summary())
    activations.print_cache_info()


if __name__ == "__main__":
    main()
