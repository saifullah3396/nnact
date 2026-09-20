from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final, override

import numpy as np
from torch.utils.data import Dataset

from nnact._outputs._sequence import SequenceActivationOutput
from nnact._outputs._token import TokenActivationOutput

if TYPE_CHECKING:
    import pandas as pd


class ActivationDataset(Dataset, ABC):
    """Base class for positional, read-only access to accumulated activations.

    Concrete subclasses hold either :class:`SequenceActivationOutput` or
    :class:`TokenActivationOutput` samples, built up in memory by an
    :class:`~nnact._steps._accumulator.ActivationAccumulator` as an
    :class:`~nnact._pipeline.ActivationPipeline` run completes.
    """

    @property
    @abstractmethod
    def layer_names(self) -> list[str]:
        """Names of the layers held, in first-batch order."""

    @abstractmethod
    def activations(self, layer_name: str) -> np.ndarray:
        """One layer's accumulated activations, stacked across samples."""

    @property
    @abstractmethod
    def metadata(self) -> dict[str, np.ndarray] | None:
        """Caller-attached row-aligned metadata, when present."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples held."""

    def exists(self) -> bool:
        """Whether this dataset already holds data from a previous run.

        ``False`` unless overridden, e.g. by a file-backed dataset that can
        be pointed at an existing cache.
        """
        return False

    def layer_shape(self, layer_name: str) -> tuple[int, ...]:
        """Per-sample activation shape for one layer, excluding the sample axis."""
        return tuple(self.activations(layer_name).shape[1:])

    def summary(self) -> pd.DataFrame:
        """Tabulate the accumulated layers as a :class:`pandas.DataFrame`.

        Returns:
            One row per layer, indexed by layer name, with columns ``samples``
            (the number held, same for every row), ``shape`` (per sample), and
            ``bytes`` (for every sample held, as float32).

        Raises:
            ImportError: If pandas is not installed. It is not a dependency of
                ``nnact``; use :attr:`layer_names` and :meth:`activations`
                instead.
        """
        import pandas as pd

        names = self.layer_names
        shapes = [self.layer_shape(name) for name in names]
        elements = [int(np.prod(shape, dtype=np.int64)) for shape in shapes]
        num_samples = len(self)

        return pd.DataFrame(
            {
                "samples": [num_samples] * len(names),
                "shape": shapes,
                "bytes": [count * num_samples * 4 for count in elements],
            },
            index=pd.Index(names, name="layer"),
        )


@final
class InMemorySequenceActivationDataset(ActivationDataset):
    """Activations accumulated in memory from :class:`SequenceActivationOutput` batches.

    Every batch's arrays are held as-is and concatenated lazily. The result
    is cached per layer, so repeated reads return the same array without
    concatenating again.
    """

    def __init__(self) -> None:
        self._predictions: list[np.ndarray] = []
        self._top_probabilities: list[np.ndarray] = []
        self._metadata: dict[str, list[np.ndarray]] = {}
        self._activations: dict[str, list[np.ndarray]] = {}
        self._activation_cache: dict[str, np.ndarray] = {}

    def _add_batch(self, output: SequenceActivationOutput) -> None:
        """Append one batch's fields to this dataset's accumulated lists."""
        self._predictions.append(output.prediction)
        self._top_probabilities.append(output.top_probability)
        if output.metadata is not None:
            for key, value in output.metadata.items():
                self._metadata.setdefault(key, []).append(value)
        for name, tensor in output.activations.items():
            self._activations.setdefault(name, []).append(tensor)
            self._activation_cache.pop(name, None)

    @property
    @override
    def layer_names(self) -> list[str]:
        return list(self._activations)

    @property
    def prediction(self) -> np.ndarray:
        """The model's own argmax prediction, one per sample."""
        return np.concatenate(self._predictions, axis=0)

    @property
    def top_probability(self) -> np.ndarray:
        """Softmax probability of :attr:`prediction`, one per sample."""
        return np.concatenate(self._top_probabilities, axis=0)

    @property
    def metadata(self) -> dict[str, np.ndarray] | None:
        """Caller-attached per-sample metadata, or ``None`` if none were provided."""
        if not self._metadata:
            return None
        return {
            key: np.concatenate(values, axis=0)
            for key, values in self._metadata.items()
        }

    @override
    def activations(self, layer_name: str) -> np.ndarray:
        if layer_name not in self._activation_cache:
            self._activation_cache[layer_name] = np.concatenate(
                self._activations[layer_name], axis=0
            )
        return self._activation_cache[layer_name]

    @override
    def layer_shape(self, layer_name: str) -> tuple[int, ...]:
        return tuple(self._activations[layer_name][0].shape[1:])

    @override
    def __len__(self) -> int:
        return sum(prediction.shape[0] for prediction in self._predictions)


@final
class InMemoryTokenActivationDataset(ActivationDataset):
    """Activations accumulated in memory from :class:`TokenActivationOutput` batches.

    Every batch's arrays are held as-is and concatenated lazily. The result
    is cached per layer, so repeated reads return the same array without
    concatenating again.
    """

    def __init__(self) -> None:
        self._offsets: list[np.ndarray] = [np.zeros(1, dtype=np.int64)]
        self._predictions: list[np.ndarray] = []
        self._top_probabilities: list[np.ndarray] = []
        self._metadata: dict[str, list[np.ndarray]] = {}
        self._activations: dict[str, list[np.ndarray]] = {}
        self._activation_cache: dict[str, np.ndarray] = {}
        self._token_ids: list[np.ndarray] = []
        self._tokens: list[np.ndarray] = []

    def _add_batch(self, output: TokenActivationOutput) -> None:
        """Append one batch's fields, rebasing its offsets onto the running total."""
        base = self._offsets[-1][-1]
        self._offsets.append(output.offsets[1:] + base)
        self._predictions.append(output.prediction)
        self._top_probabilities.append(output.top_probability)
        if output.metadata is not None:
            for key, value in output.metadata.items():
                self._metadata.setdefault(key, []).append(value)
        for name, tensor in output.activations.items():
            self._activations.setdefault(name, []).append(tensor)
            self._activation_cache.pop(name, None)
        if output.token_ids is not None:
            self._token_ids.append(output.token_ids)
        if output.tokens is not None:
            self._tokens.append(output.tokens)

    @property
    @override
    def layer_names(self) -> list[str]:
        return list(self._activations)

    @property
    def offsets(self) -> np.ndarray:
        """Sample boundaries into the flat token layout, shape ``(len(self) + 1,)``."""
        return np.concatenate(self._offsets, axis=0)

    @property
    def prediction(self) -> np.ndarray:
        """The model's own argmax prediction, one per real token."""
        return np.concatenate(self._predictions, axis=0)

    @property
    def top_probability(self) -> np.ndarray:
        """Softmax probability of :attr:`prediction`, one per real token."""
        return np.concatenate(self._top_probabilities, axis=0)

    @property
    def metadata(self) -> dict[str, np.ndarray] | None:
        """Caller-attached per-token metadata, or ``None`` if none were provided."""
        if not self._metadata:
            return None
        return {
            key: np.concatenate(values, axis=0)
            for key, values in self._metadata.items()
        }

    @override
    def activations(self, layer_name: str) -> np.ndarray:
        if layer_name not in self._activation_cache:
            self._activation_cache[layer_name] = np.concatenate(
                self._activations[layer_name], axis=0
            )
        return self._activation_cache[layer_name]

    @override
    def layer_shape(self, layer_name: str) -> tuple[int, ...]:
        return tuple(self._activations[layer_name][0].shape[1:])

    @property
    def token_ids(self) -> np.ndarray | None:
        """Input token id per real token, or ``None`` if none were provided."""
        return np.concatenate(self._token_ids, axis=0) if self._token_ids else None

    @property
    def tokens(self) -> np.ndarray | None:
        """Decoded token string per real token, or ``None`` if none were provided."""
        return np.concatenate(self._tokens, axis=0) if self._tokens else None

    @property
    def sequence_lengths(self) -> np.ndarray:
        """Real token count per sample, shape ``(len(self),)``."""
        offsets = self.offsets
        return offsets[1:] - offsets[:-1]

    @property
    def sample_of_token(self) -> np.ndarray:
        """Which accumulated sample each flat token index belongs to."""
        return np.repeat(np.arange(len(self)), self.sequence_lengths)

    @override
    def __len__(self) -> int:
        return max(self.offsets.size - 1, 0)

    @override
    def summary(self) -> pd.DataFrame:
        """Tabulate every real token held, one row per token.

        Returns:
            A :class:`pandas.DataFrame` indexed by flat token position, with
            columns ``sample`` (which accumulated sample the token belongs
            to), ``token_id`` and ``token`` (present only when a tokenizer was
            given to the pipeline), ``predicted_id`` (the model's own argmax
            prediction for that token), ``predicted_probability``, one column
            per caller-attached metadata key.

        Raises:
            ImportError: If pandas is not installed. It is not a dependency of
                ``nnact``; use :attr:`token_ids`, :attr:`tokens`, and
                :meth:`activations` instead.
        """
        import pandas as pd

        offsets = self.offsets
        columns: dict[str, object] = {"sample": self.sample_of_token.tolist()}

        token_ids = self.token_ids
        if token_ids is not None:
            columns["token_id"] = token_ids.tolist()
        tokens = self.tokens
        if tokens is not None:
            columns["token"] = tokens.tolist()

        columns["predicted_id"] = self.prediction.tolist()
        columns["predicted_probability"] = self.top_probability.tolist()

        metadata = self.metadata
        if metadata is not None:
            for key, values in metadata.items():
                columns[key] = values.tolist()

        return pd.DataFrame(
            columns, index=pd.RangeIndex(int(offsets[-1]), name="token")
        )
