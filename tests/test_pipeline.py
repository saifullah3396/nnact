from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from nnact import (
    ActivationPipeline,
    SequenceActivationSample,
    TokenActivationSample,
)
from tests.conftest import FakeModel, make_model_input


class TokenDataset(Dataset):
    def __init__(self, seqs: list[list[int]], metas: list[list[str]]) -> None:
        self.seqs = seqs
        self.metas = metas

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> TokenActivationSample:
        return TokenActivationSample(
            model_input=make_model_input(ids=self.seqs[idx]),
            metadata={"label": self.metas[idx]},
        )


class SequenceDataset(Dataset):
    def __init__(self, seqs: list[list[int]], metas: list[str]) -> None:
        self.seqs = seqs
        self.metas = metas

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> SequenceActivationSample:
        return SequenceActivationSample(
            model_input=make_model_input(ids=self.seqs[idx]),
            metadata={"label": self.metas[idx]},
        )


def test_token_pipeline_in_memory() -> None:
    dataset = TokenDataset(
        seqs=[[1, 2, 3], [4, 5, 3]], metas=[["a", "b", "c"], ["d", "e", "c"]]
    )
    pipeline = ActivationPipeline(
        model=FakeModel(),
        layer_names=["linear"],
        output_type="token",
        show_progress=False,
    )
    result = pipeline.run(dataset=dataset, batch_size=2)

    assert len(result.dataset) == 2
    assert result.dataset.metadata is not None
    assert result.dataset.metadata["label"].tolist() == ["a", "b", "c", "d", "e", "c"]
    assert result.metadata["model"] == "FakeModel"
    assert result.metadata["output_type"] == "token"
    assert "duration_seconds" in result.metadata


def test_sequence_pipeline_in_memory() -> None:
    dataset = SequenceDataset(seqs=[[1, 2], [3, 4]], metas=["user", "assistant"])
    pipeline = ActivationPipeline(
        model=FakeModel(),
        layer_names=["linear"],
        output_type="sequence",
        show_progress=False,
    )
    result = pipeline.run(dataset=dataset, batch_size=2)

    assert len(result.dataset) == 2
    assert result.dataset.metadata is not None
    assert result.dataset.metadata["label"].tolist() == ["user", "assistant"]


def test_cached_pipeline_resumes_without_rerunning(tmp_path: Path) -> None:
    dataset = SequenceDataset(seqs=[[1, 2], [3, 4]], metas=["user", "assistant"])
    pipeline = ActivationPipeline(
        model=FakeModel(),
        layer_names=["linear"],
        output_type="sequence",
        cache_dir=tmp_path,
        cache_outputs=True,
        show_progress=False,
    )
    first = pipeline.run(dataset=dataset, batch_size=2)
    assert "duration_seconds" in first.metadata

    second = pipeline.run(dataset=dataset, batch_size=2)
    assert "duration_seconds" not in second.metadata
    assert len(second.dataset) == 2
    assert second.dataset.metadata is not None
    assert second.dataset.metadata["label"].tolist() == ["user", "assistant"]
