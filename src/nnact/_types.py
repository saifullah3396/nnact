from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Self

import torch


@dataclass(frozen=True, slots=True)
class LayerActivation:
    layer_name: str
    tensor: torch.Tensor


class SampleLike(Protocol):
    """Optional per-sample convention for datasets used by a DataLoader.

    The mapper itself consumes batches from a caller-owned DataLoader. A
    dataset may use this convenient ``id``/``data`` convention, but its data
    can be any object and the caller's collate function decides how it becomes
    model keyword inputs.
    """

    id: str
    data: Any


@dataclass(frozen=True, slots=True)
class Sample:
    """Convenient ``id``/``data`` sample convention for caller datasets."""

    id: str
    data: Any


@dataclass(frozen=True, slots=True)
class ActivatedSample:
    """Activations captured from one or more layers for a single sample."""

    id: str
    activations: list[LayerActivation]


@dataclass(frozen=True, slots=True)
class ModelOutput:
    logits: torch.Tensor


def _format_count(value: int) -> str:
    """Abbreviate a large count, e.g. ``11689512`` to ``11.7M``."""
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= limit:
            return f"{value / limit:.1f}{suffix}"
    return str(value)


def _format_duration(seconds: float) -> str:
    """Render a duration at a readable scale, from microseconds to hours."""
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m{seconds % 60:04.1f}s"
    if seconds >= 1:
        return f"{seconds:.2f}s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.1f}ms"
    return f"{seconds * 1e6:.0f}us"


@dataclass(frozen=True, slots=True, repr=False)
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
        device: Hardware the forward passes ran on — ``"cpu"``, or the GPU's
            model name such as ``"NVIDIA GeForce RTX 3060 (cuda:0)"``. Not
            where the activations were stored: those are always float32 on
            CPU, whatever the model ran on.
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

    def __repr__(self) -> str:
        """Render one field per line, in the style of a pydantic model.

        Counts are abbreviated and the duration is scaled, since the exact
        parameter count and a full float of seconds are rarely what you want
        to read. Every field remains available as an attribute.
        """
        fields: list[tuple[str, str]] = [
            ("model", self.model),
            ("parameters", _format_count(self.parameters)),
            ("layers", ", ".join(self.layers)),
            ("samples", f"{self.samples:,}"),
            ("batch_size", str(self.batch_size)),
            ("device", self.device),
            ("seconds", _format_duration(self.seconds)),
            ("throughput", f"{_format_count(int(self.samples_per_second))}/s"),
            ("created", self.created),
        ]
        if self.extra:
            fields.append(("extra", repr(self.extra)))

        width = max(len(name) for name, _ in fields)
        lines = [f"{type(self).__name__}("]
        lines.extend(f"    {name:<{width}} = {value}" for name, value in fields)
        lines.append(")")
        return "\n".join(lines)

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
