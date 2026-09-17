"""GPT-Neo residual stream — cached to HDF5.

Minimal script form of examples/text/04_gpt2_residual_stream_cache.ipynb.
Same run as 02_gpt2_residual_stream.py, but with cache_outputs=True so
activations stream to an HDF5 file instead of staying in RAM.

Run: python examples/scripts/04_gpt2_residual_stream_cache.py
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

    pipeline = ActivationPipeline(
        model,
        layers,
        output_type="token",
        tokenizer=tokenizer,
        cache_outputs=True,
        run_dir=Path("runs") / "04_gpt2_residual_stream_cache",
    )
    loader = activation_loader(dataset, batch_size=16)
    activations = pipeline.run(loader)
    print(activations.summary())
    activations.print_cache_info()


if __name__ == "__main__":
    main()
