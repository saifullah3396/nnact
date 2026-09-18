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
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples._utils.data import activation_loader
from nnact import ActivationPipeline, ProbeConfig, ProbePipeline, ProbeTrainer

MODEL = "Qwen/Qwen3-1.7B"
FAKE_DATASET_PATH = Path(__file__).parents[3] / "fake_dataset.jsonl"
ROLES = ["user", "assistant", "system", "tool", "cot"]
ROLE_TO_ID = {role: index for index, role in enumerate(ROLES)}
NO_ROLE_LABEL = -1
LAYERS_TO_PROBE = 16
NUM_SAMPLES = 50  # cap for a quick test run, e.g. 100; None uses the full dataset
SAMPLE_SEED = 0
PROBE_CACHE_DIR = Path("runs") / "01_train_role_probes" / "probes"

# Each combination gets its own probe, trained only on tokens whose role is
# in that combination -- mirrors the reference script's role_combinations.
ROLE_COMBINATIONS = [
    ["user", "assistant"],
    ["user", "assistant", "tool"],
    ["user", "cot", "assistant"],
    ["user", "cot", "assistant", "tool"],
]


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


def drop_outside_role_space(
    *, labels: np.ndarray, sample_of_row: np.ndarray, token_ids: np.ndarray | None
) -> np.ndarray:
    """Keep only tokens RoleSpaceView could map into its role space."""
    return labels != NO_ROLE_LABEL


class RoleSpaceView:
    """Presents a dataset's global role IDs as a dense 0..k-1 index.

    ProbeTrainer fits directly on whatever ``labels`` a dataset exposes,
    so probability columns from ``predict_proba`` end up ordered by the
    classifier's own sorted unique training labels -- not by the label's
    raw value. Fitting on global IDs like {user: 0, tool: 3} would leave a
    "tool" prediction's confidence at probabilities[..., 2] (its position
    among {0, 1, 3}), not probabilities[..., 3]. Remapping labels to
    0..k-1 up front, exactly as the reference role_combinations loop does
    with its ``roles_map``, keeps a label's value and column position the
    same. Everything else -- activations, sample_of_token, token_ids -- is
    passed through unchanged.
    """

    def __init__(self, dataset: object, role_space: list[str]) -> None:
        self._dataset = dataset
        self._role_to_local_id = {role: index for index, role in enumerate(role_space)}
        self._global_to_local = {
            ROLE_TO_ID[role]: index for role, index in self._role_to_local_id.items()
        }

    @property
    def labels(self) -> np.ndarray:
        global_labels = self._dataset.labels
        local_labels = np.full_like(global_labels, NO_ROLE_LABEL)
        for global_id, local_id in self._global_to_local.items():
            local_labels[global_labels == global_id] = local_id
        return local_labels

    def __getattr__(self, name: str) -> object:
        return getattr(self._dataset, name)

    def __len__(self) -> int:
        return len(self._dataset)


def main() -> None:
    layer_name = f"model.layers.{LAYERS_TO_PROBE}"
    role_space_cache_paths = {
        ",".join(role_space): PROBE_CACHE_DIR / ("-".join(role_space) + ".npz")
        for role_space in ROLE_COMBINATIONS
    }
    all_probes_cached = all(
        path.exists() for path in role_space_cache_paths.values()
    )

    activations = None
    if not all_probes_cached:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL)

        dataset = RoleConversationSamples(
            FAKE_DATASET_PATH, n=NUM_SAMPLES, seed=SAMPLE_SEED
        )
        print(f"{len(dataset)} conversations | roles: {ROLES}")

        activation_pipeline = ActivationPipeline(
            model,
            [layer_name],
            output_type="token",
            tokenizer=tokenizer,
            cache_outputs=False,
            run_dir=Path("runs") / "01_train_role_probes",
        )
        loader = activation_loader(dataset, batch_size=1)
        activations = activation_pipeline.run(loader)
        print(activations.summary())
    else:
        print("All role-space probes already cached; skipping activation extraction.")

    probe_pipeline = ProbePipeline(ProbeTrainer(ProbeConfig()))
    rows: list[dict[str, float | str]] = []
    for role_space in ROLE_COMBINATIONS:
        role_space_key = ",".join(role[0] for role in role_space)
        cache_path = role_space_cache_paths[",".join(role_space)]
        dataset_for_role_space = (
            RoleSpaceView(activations, role_space) if activations is not None else None
        )
        result = probe_pipeline.train(
            dataset_for_role_space,
            layer_name,
            cache_path=cache_path,
            filter_fn=drop_outside_role_space,
        )
        metrics = result.metrics
        confusion_table = pd.DataFrame(
            metrics.confusion_matrix,
            index=pd.Index(role_space, name="true"),
            columns=pd.Index(role_space, name="predicted"),
        )
        print(f"\nconfusion matrix [{role_space_key}]:\n{confusion_table}\n")
        rows.append(
            {
                "layer": LAYERS_TO_PROBE,
                "role_space": role_space_key,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
            }
        )

    results_path = PROBE_CACHE_DIR / "results.json"
    json_rows = [
        {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    results_path.write_text(json.dumps(json_rows, indent=2))

    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
