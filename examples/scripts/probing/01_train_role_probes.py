"""Train role probes on a Qwen model's activations.

Minimal, model-agnostic replacement for the reference 01_train_role_probes.py
(rcv2/atria_core-based): loads the pre-built fake_dataset.jsonl at the repo
root as-is (no conversation generation), runs it through nnact's own
ActivationPipeline against a real Qwen model, then fits a probe per layer
with ProbeTrainer to see how well each layer's activations predict a
token's conversational role (user/assistant/system/tool/cot).

Run: python examples/scripts/probing/01_train_role_probes.py
"""

import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[3]))

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples._utils.data import activation_loader
from nnact import ActivationPipeline, ProbeConfig, ProbeTrainer

MODEL = "Qwen/Qwen3-1.7B"
FAKE_DATASET_PATH = Path(__file__).parents[3] / "fake_dataset.jsonl"
ROLES = ["user", "assistant", "system", "tool", "cot"]
ROLE_TO_ID = {role: index for index, role in enumerate(ROLES)}
NO_ROLE_LABEL = -1
LAYERS_TO_PROBE = 16
NUM_SAMPLES = 5  # cap for a quick test run, e.g. 100; None uses the full dataset
SAMPLE_SEED = 0


class RoleConversationSamples(Dataset[dict[str, Any]]):
    """Pre-tokenized fake conversations with a per-token role label.

    Each line of fake_dataset.jsonl already carries `token_ids`,
    `attention_mask`, and `token_roles` (one of ROLES, or None for tokens
    that aren't part of a labeled turn) at a fixed sequence length, so no
    tokenization happens here -- this just loads and encodes the roles.
    """

    def __init__(self, path: Path, n: int | None = None, seed: int = 0) -> None:
        with path.open() as f:
            records = [json.loads(line) for line in f]

        if n is not None and n < len(records):
            records = random.Random(seed).sample(records, n)

        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self._records[idx]
        labels = [ROLE_TO_ID.get(role, NO_ROLE_LABEL) for role in record["token_roles"]]
        return {
            "input_ids": torch.tensor(record["token_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(record["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def drop_unlabeled_tokens(
    *, labels: np.ndarray, sample_of_row: np.ndarray, token_ids: np.ndarray | None
) -> np.ndarray:
    """Keep only tokens whose role label is one of ROLES, dropping padding
    and any token that fell outside a labeled turn (encoded as NO_ROLE_LABEL).
    """
    return labels != NO_ROLE_LABEL


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    dataset = RoleConversationSamples(
        FAKE_DATASET_PATH, n=NUM_SAMPLES, seed=SAMPLE_SEED
    )
    print(f"{len(dataset)} conversations | roles: {ROLES}")

    layer_name = f"model.layers.{LAYERS_TO_PROBE}"
    pipeline = ActivationPipeline(
        model,
        [layer_name],
        output_type="token",
        tokenizer=tokenizer,
        cache_outputs=False,
        run_dir=Path("runs") / "01_train_role_probes",
    )
    loader = activation_loader(dataset, batch_size=1)
    activations = pipeline.run(loader)
    print(activations.summary())

    trainer = ProbeTrainer(ProbeConfig())
    result = trainer.fit(activations, layer_name, filter_fn=drop_unlabeled_tokens)
    print(
        f"layer {layer_name}: accuracy={result.accuracy:.4f} "
        f"(train={result.num_train_rows}, test={result.num_test_rows})"
    )


if __name__ == "__main__":
    main()
