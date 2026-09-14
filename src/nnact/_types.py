from dataclasses import asdict, dataclass, field
from typing import Any, Self

import torch


@dataclass(frozen=True, slots=True)
class LayerActivation:
    layer_name: str
    tensor: torch.Tensor


@dataclass(frozen=True, slots=True)
class Sample:
    """A raw input sample for activation collection."""

    id: str
    data: Any


@dataclass(frozen=True, slots=True)
class ActivatedSample:
    """Activations captured from one or more layers for a single sample."""

    activations: list[LayerActivation]


@dataclass(frozen=True, slots=True)
class ModelOutput:
    logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """How an activation store was produced.

    Recorded by :class:`~nnact.ActivationMapper` and carried by the resulting
    store, so a cache found on disk months later can still say where it came
    from. Persisted as JSON in the HDF5 backend.

    Attributes:
        model: Class name of the model the activations came from.
        parameters: Total parameter count of that model.
        layers: Layers captured, in the order the store exposes them.
        samples: Number of samples written.
        batch_size: Samples per forward pass.
        device: Device batches were moved to, or ``"cpu"``.
        seconds: Wall-clock duration of the extraction.
        created: UTC ISO-8601 timestamp taken when the run finished.
        extra: Anything else the caller chose to record.
    """

    model: str = ""
    parameters: int = 0
    layers: list[str] = field(default_factory=list)
    samples: int = 0
    batch_size: int = 0
    device: str = ""
    seconds: float = 0.0
    created: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def samples_per_second(self) -> float:
        """Throughput of the run, or ``0.0`` if it was too fast to measure."""
        return self.samples / self.seconds if self.seconds else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of every field."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict`, tolerating unknown or missing keys.

        Fields the current version does not define are kept in :attr:`extra`,
        so metadata written by a newer version survives a round trip instead of
        raising.

        Args:
            data: A mapping produced by :meth:`to_dict`, or read back from a
                stored JSON attribute.

        Returns:
            The reconstructed metadata.
        """
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        extra = dict(data.get("extra") or {})
        extra.update({k: v for k, v in data.items() if k not in known and k != "extra"})
        return cls(**{k: v for k, v in data.items() if k in known}, extra=extra)
