import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from nnact import ActivationPipeline, ProbeConfig, ProbePipeline, ProbeTrainer
from nnact._outputs._protocols import ActivationSample, SequenceModelInput
from nnact._probing._trainer import FilterFn

MODEL = "Qwen/Qwen3-1.7B"
FAKE_DATASET_PATH = Path(__file__).parents[3] / "fake_dataset.jsonl"
ROLES = ["user", "assistant", "system", "tool", "cot"]
NO_ROLE_LABEL = "none"
LAYERS_TO_PROBE = 16
NUM_SAMPLES = None  # cap for a quick test run, e.g. 100; None uses the full dataset
SAMPLE_SEED = 0
PROBE_CACHE_DIR = Path("runs") / "01_train_role_probes" / "probes"
# Reference's TensorProbeTrainerConfig.skip_first_n for nested_reasoning=True --
# drops each labeled turn's first 32 tokens (near a role-switch boundary).
SKIP_FIRST_N = 32

# Each combination gets its own probe, trained only on tokens whose role is
# in that combination -- mirrors the reference script's role_combinations.
ROLE_COMBINATIONS = [
    ["user", "assistant"],
    ["user", "assistant", "tool"],
    ["user", "cot", "assistant"],
    # ["user", "cot", "assistant", "tool"],
]


class RoleConversationSamples(Dataset[ActivationSample]):
    def __init__(self, path: Path, n: int | None = None, seed: int = 0) -> None:
        with path.open() as f:
            records = [json.loads(line) for line in f]

        if n is not None and n < len(records):
            records = random.Random(seed).sample(records, n)

        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> ActivationSample:
        record = self._records[idx]
        target_role = record["metadata"]["target_role"]
        if target_role == "thinking":
            target_role = "cot"
        labels = [
            role if role == target_role else NO_ROLE_LABEL
            for role in record["token_roles"]
        ]
        return ActivationSample(
            model_input=SequenceModelInput(
                input_ids=torch.tensor(record["token_ids"], dtype=torch.long),
                attention_mask=torch.tensor(record["attention_mask"], dtype=torch.long),
            ),
            activation_labels=labels,
        )

    def turn_positions(self) -> np.ndarray:
        return np.concatenate(
            [
                np.array(
                    [
                        p if p is not None else -1
                        for p, kept in zip(
                            record["token_idx_in_turn"],
                            record["attention_mask"],
                            strict=True,
                        )
                        if kept
                    ],
                    dtype=np.int64,
                )
                for record in self._records
            ]
        )


def make_drop_outside_role_space(
    role_space: list[str], turn_positions: np.ndarray, skip_first_n: int
) -> FilterFn:
    def drop_outside_role_space(
        *, labels: np.ndarray, sample_of_row: np.ndarray, token_ids: np.ndarray | None
    ) -> np.ndarray:
        return np.isin(labels, role_space) & (turn_positions >= skip_first_n)

    return drop_outside_role_space


def main() -> None:
    # The reference probes `all_pre_mlp_hidden_states` -- the layernormed
    # hidden state going into the MLP, not the decoder block's full output.
    # `post_attention_layernorm` is that exact tensor's producing submodule.
    layer_name = f"model.layers.{LAYERS_TO_PROBE}.post_attention_layernorm"
    role_space_cache_paths = {
        ",".join(role_space): PROBE_CACHE_DIR / ("-".join(role_space) + ".npz")
        for role_space in ROLE_COMBINATIONS
    }
    all_probes_cached = all(path.exists() for path in role_space_cache_paths.values())

    activations = None
    turn_positions = None
    if not all_probes_cached:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto")

        dataset = RoleConversationSamples(
            FAKE_DATASET_PATH, n=NUM_SAMPLES, seed=SAMPLE_SEED
        )
        print(f"{len(dataset)} conversations | roles: {ROLES}")
        turn_positions = dataset.turn_positions()

        activation_pipeline = ActivationPipeline(
            model,
            [layer_name],
            output_type="token",
            tokenizer=tokenizer,
            cache_outputs=True,
            run_dir=Path("runs") / "01_train_role_probes",
        )
        activations = activation_pipeline.run(dataset, batch_size=1)
        print(activations.summary())
    else:
        print("All role-space probes already cached; skipping activation extraction.")

    probe_pipeline = ProbePipeline(
        ProbeTrainer(ProbeConfig(C=1.0e-1, add_scaling=False))
    )
    rows: list[dict[str, float | str]] = []
    for role_space in ROLE_COMBINATIONS:
        role_space_key = ",".join(role[0] for role in role_space)
        cache_path = role_space_cache_paths[",".join(role_space)]
        filter_fn = None
        if activations is not None:
            filter_fn = make_drop_outside_role_space(
                role_space, turn_positions, SKIP_FIRST_N
            )
            eligible = filter_fn(
                labels=activations.labels,
                sample_of_row=activations.sample_of_token,
                token_ids=getattr(activations, "token_ids", None),
            )
            print(
                f"DEBUG role_space={role_space} eligible_tokens={int(eligible.sum())} "
                f"skip_first_n={SKIP_FIRST_N}"
            )
        result = probe_pipeline.train(
            activations,
            layer_name,
            cache_path=cache_path,
            filter_fn=filter_fn,
        )
        metrics = result.metrics
        confusion_table = pd.DataFrame(
            metrics.confusion_matrix,
            index=pd.Index(result.classes_, name="true"),
            columns=pd.Index(result.classes_, name="predicted"),
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
