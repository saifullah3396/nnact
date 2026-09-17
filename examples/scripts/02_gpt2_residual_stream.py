"""GPT-Neo residual stream — every block, in memory.

Minimal script form of examples/text/02_gpt2_residual_stream.ipynb.
Downloads on first run: WikiText-2 (~5 MB) and GPT-Neo 125M weights (~500 MB).

Run: python examples/scripts/02_gpt2_residual_stream.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from transformers import AutoModelForCausalLM, AutoTokenizer

from examples._utils.data import activation_loader
from examples._utils.text import WikiTextSamples
from nnact import ActivationPipeline


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-Neo ships without a pad token
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
    dataset = WikiTextSamples(tokenizer, n=32, max_length=64)
    num_layers = model.config.num_layers
    layers = [f"transformer.h.{i}" for i in range(num_layers)] + ["transformer.ln_f"]
    print(f"{len(dataset)} passages | {len(layers)} layers: {layers}")

    # run_dir is required for every run: it holds run.log and run_metadata.json.
    pipeline = ActivationPipeline(
        model,
        layers,
        output_type="token",
        run_dir=Path(__file__).parent / "runs" / "02_gpt2_residual_stream",
    )
    loader = activation_loader(dataset, batch_size=16)
    activations = pipeline.run(loader)
    print(activations.summary())


if __name__ == "__main__":
    main()
