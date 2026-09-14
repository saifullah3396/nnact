import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, final, overload

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nnact._model import HookedModel
from nnact._types import RunMetadata, Sample
from nnact.store import (
    ActivationStore,
    ActivationWriter,
    H5ActivationStore,
    H5ActivationWriter,
    MemoryActivationStore,
    MemoryActivationWriter,
)

if TYPE_CHECKING:
    import pandas as pd


def _as_list(layer_names: str | list[str]) -> list[str]:
    """Accept a single layer name or a list of them."""
    return [layer_names] if isinstance(layer_names, str) else list(layer_names)


def _with_progress[T](
    batches: Iterable[T], *, enabled: bool, label: str
) -> Iterable[T]:
    """Wrap an iterable in a tqdm bar, falling back to it unchanged.

    tqdm is not a dependency of ``nnact``, and its notebook variant is picked
    automatically when running under Jupyter.

    Args:
        batches: The iterable to wrap.
        enabled: Pass ``False`` to skip the bar entirely.
        label: Description shown beside the bar.

    Returns:
        ``batches``, wrapped if tqdm is available and ``enabled`` is set.
    """
    if not enabled:
        return batches
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return batches
    return tqdm(batches, desc=f"{label} activations", unit="batch")


def _describe_device(device: torch.device) -> str:
    """Name the hardware behind a device, not just its index.

    ``cuda:0`` says nothing about which card ran a job, which is what makes a
    recorded duration interpretable later.

    Args:
        device: The device to describe.

    Returns:
        The GPU's model name with its index, or the plain device type for CPU
        and for accelerators exposing no name.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        return f"{torch.cuda.get_device_name(index)} (cuda:{index})"
    return str(device)


def _model_device(model: nn.Module, override: torch.device | str | None) -> str:
    """Report the device the model's forward passes actually ran on.

    Prefers the model's own parameters over the ``device`` argument, since a
    model already living on a GPU runs there whether or not a device was
    passed. Falls back to the argument for parameterless models.

    Args:
        model: The model that was run.
        override: The ``device`` argument given to :meth:`ActivationMapper.map`.

    Returns:
        A human-readable device description.
    """
    try:
        return _describe_device(next(model.parameters()).device)
    except StopIteration:
        return _describe_device(torch.device(override)) if override else "cpu"


def _collate_samples(batch: list[Sample]) -> tuple[list[str], torch.Tensor]:
    ids = [s.id for s in batch]
    stacked = torch.stack([s.data for s in batch])  # type: ignore[arg-type]
    return ids, stacked


@final
class ActivationMapper:
    def __init__(self, model: nn.Module) -> None:
        self._model = model

    def available_layers(self, depth: int | None = None) -> list[str]:
        """List the layer names the model accepts, in definition order.

        Args:
            depth: Keep only names at most this many levels deep, counting
                dots — ``1`` gives top-level blocks such as ``layer4``, ``2``
                descends one level into them. ``None`` lists every module.

        Returns:
            Hookable module names. The model itself is excluded.
        """
        names = [name for name, _ in self._model.named_modules() if name]
        if depth is None:
            return names
        return [name for name in names if name.count(".") < depth]

    def parameter_count(self) -> int:
        """Total number of parameters in the wrapped model."""
        return sum(p.numel() for p in self._model.parameters())

    def summary(
        self, layer_names: str | list[str] | None = None, depth: int = 3
    ) -> "pd.DataFrame":
        """Tabulate the hookable layers as a :class:`pandas.DataFrame`.

        Useful before extracting: it shows which names are hookable, what kind
        of module each one is, and how many parameters each holds. Being a
        DataFrame, it renders as a table in a notebook and can be filtered or
        sorted like any other.

        Args:
            layer_names: Mark these layers in a ``selected`` column. Omit to
                list every layer without marking.
            depth: Nesting depth to list, as in :meth:`available_layers`.

        Returns:
            One row per hookable layer, indexed by layer name, with columns
            ``module`` (the class name), ``parameters``, and — when
            ``layer_names`` is given — ``selected``.

        Raises:
            ImportError: If pandas is not installed. It is not a dependency of
                ``nnact``; use :meth:`available_layers` instead.
        """
        import pandas as pd

        modules = dict(self._model.named_modules())
        names = self.available_layers(depth=depth)

        frame = pd.DataFrame(
            {
                "module": [type(modules[name]).__name__ for name in names],
                "parameters": [
                    sum(p.numel() for p in modules[name].parameters()) for name in names
                ],
            },
            index=pd.Index(names, name="layer"),
        )

        if layer_names is not None:
            selected = set(_as_list(layer_names))
            frame["selected"] = [name in selected for name in names]
        return frame

    def _check_layers(self, layer_names: list[str]) -> None:
        """Fail before the first forward pass if a requested layer is absent.

        Args:
            layer_names: The layers about to be captured.

        Raises:
            ValueError: If any is not a module of the model. The message lists
                the closest available names, since the usual cause is a typo or
                the wrong nesting depth.
        """
        available = self.available_layers()
        missing = [name for name in layer_names if name not in available]
        if not missing:
            return

        headline = (
            f"Layer(s) not found in {type(self._model).__name__}: {', '.join(missing)}"
        )
        lines = [headline]
        for name in missing:
            parent = name.rsplit(".", 1)[0] if "." in name else ""
            siblings = [
                candidate
                for candidate in available
                if (candidate.rsplit(".", 1)[0] if "." in candidate else "") == parent
            ]
            if siblings:
                where = f"under '{parent}'" if parent else "at the top level"
                lines.append(f"  did you mean, {where}: {', '.join(siblings[:8])}")
        lines.append(
            f"Call summary() or available_layers() to list all "
            f"{len(available)} hookable layers."
        )
        raise ValueError("\n".join(lines))

    @overload
    def map(
        self,
        dataset: Dataset[Sample],
        layer_names: str | list[str],
        path: None = None,
        writer: None = None,
        batch_size: int = 256,
        num_workers: int = 0,
        device: torch.device | str | None = None,
        progress: bool = True,
    ) -> MemoryActivationStore: ...

    @overload
    def map(
        self,
        dataset: Dataset[Sample],
        layer_names: str | list[str],
        path: Path,
        writer: None = None,
        batch_size: int = 256,
        num_workers: int = 0,
        device: torch.device | str | None = None,
        progress: bool = True,
    ) -> H5ActivationStore: ...

    @overload
    def map(
        self,
        dataset: Dataset[Sample],
        layer_names: str | list[str],
        path: Path | None = None,
        *,
        writer: ActivationWriter,
        batch_size: int = 256,
        num_workers: int = 0,
        device: torch.device | str | None = None,
        progress: bool = True,
    ) -> ActivationStore: ...

    def map(
        self,
        dataset: Dataset[Sample],
        layer_names: str | list[str],
        path: Path | None = None,
        writer: ActivationWriter | None = None,
        batch_size: int = 256,
        num_workers: int = 0,
        device: torch.device | str | None = None,
        progress: bool = True,
    ) -> ActivationStore:
        """Run the model over ``dataset`` and store the captured activations.

        Args:
            dataset: Yields :class:`~nnact._types.Sample` items. Consumed in
                order, so position ``i`` of the store is the ``i``-th sample.
            layer_names: Layer or layers to capture, named as in
                :meth:`available_layers`.
            path: Destination for an HDF5 cache. When omitted, activations are
                kept in memory.
            writer: Supply a writer directly, overriding ``path``.
            batch_size: Samples per forward pass.
            num_workers: DataLoader worker processes. Leave at ``0`` when
                writing to HDF5, since an open file handle does not survive
                ``fork``.
            device: Device to move batches to. The model is not moved; do that
                yourself beforehand.
            progress: Show a tqdm progress bar over the batches. Silently
                skipped if tqdm is not installed.

        Returns:
            A store over the captured activations. The concrete type is
            narrowed for type checkers: omitting ``path`` and ``writer`` gives a
            :class:`~nnact.store.MemoryActivationStore`, so ``.activations`` is
            available without a cast; passing ``path`` gives a
            :class:`~nnact.store.H5ActivationStore`.

        Raises:
            ValueError: If a requested layer is not a module of the model. The
                check runs before the first forward pass, so a typo fails
                immediately rather than after a long extraction.
        """
        names = _as_list(layer_names)
        self._check_layers(names)

        loader: DataLoader[Sample] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_samples,  # type: ignore[arg-type]
        )

        if writer is None:
            writer = (
                MemoryActivationWriter() if path is None else H5ActivationWriter(path)
            )

        hooked = HookedModel(self._model)
        hooked.eval()

        batches = _with_progress(
            loader, enabled=progress, label=type(self._model).__name__
        )
        started = time.perf_counter()
        with torch.no_grad():
            for ids, data in batches:  # type: ignore[misc]
                if device is not None:
                    data = data.to(device)
                with hooked.capture(names):
                    hooked(data)
                    activations = {name: hooked.get_activation(name) for name in names}
                    writer.write(ids, activations)
        elapsed = time.perf_counter() - started

        writer.metadata = RunMetadata(
            model=type(self._model).__name__,
            parameters=self.parameter_count(),
            layers=names,
            samples=len(writer.sample_ids),
            batch_size=batch_size,
            device=_model_device(self._model, device),
            seconds=elapsed,
            created=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        return writer.close()
