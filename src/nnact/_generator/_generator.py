from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, final, overload

import pandas as pd
import torch
from ignite.engine import Engine
from ignite.handlers import Timer
from torch import nn


@final
class ActivationGenerator:
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
    ) -> pd.DataFrame:
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
            selected = set(
                [layer_names] if isinstance(layer_names, str) else list(layer_names)
            )
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

    def _create_engine(
        self, step: ActivationStep, writer: ActivationWriter, *, progress: bool
    ) -> tuple[Engine, Timer]:
        """Build the Ignite engine and attach run-level handlers."""
        engine = Engine(step)
        timer = Timer(average=False).attach(engine)
        _attach_writer(engine, writer)

        if progress:
            from ignite.contrib.handlers import ProgressBar

            ProgressBar(desc=f"{type(self._model).__name__} activations").attach(engine)

        return engine, timer

    def _create_writer(
        self, path: Path | None, writer: ActivationWriter | None
    ) -> ActivationWriter:
        """Use the explicit writer or create one from the optional path."""
        if writer is not None:
            return writer
        return MemoryActivationWriter() if path is None else H5ActivationWriter(path)

    @overload
    def map(
        self,
        loader: Iterable[Mapping[str, Any]],
        layer_names: str | list[str],
        path: None = None,
        writer: None = None,
        device: torch.device | str | None = None,
        progress: bool = True,
    ) -> MemoryActivationStore: ...

    @overload
    def map(
        self,
        loader: Iterable[Mapping[str, Any]],
        layer_names: str | list[str],
        path: Path,
        writer: None = None,
        device: torch.device | str | None = None,
        progress: bool = True,
    ) -> H5ActivationStore: ...

    @overload
    def map(
        self,
        loader: Iterable[Mapping[str, Any]],
        layer_names: str | list[str],
        path: Path | None = None,
        *,
        writer: ActivationWriter,
        device: torch.device | str | None = None,
        progress: bool = True,
    ) -> ActivationStore: ...

    def map(
        self,
        loader: Iterable[Mapping[str, Any]],
        layer_names: str | list[str],
        path: Path | None = None,
        writer: ActivationWriter | None = None,
        device: torch.device | str | None = None,
        progress: bool = True,
    ) -> ActivationStore:
        """Run the model over pre-batched inputs and store its activations.

        Args:
            loader: A caller-configured batch iterable, usually a DataLoader,
                yielding mappings with an ``"id"`` field and model input
                fields. IDs must be a batch of strings. All other fields are
                passed to the model as keyword arguments, so the model's
                ``forward`` parameter names define their meaning. The loader
                owns dataset access and collation.
            layer_names: Layer or layers to capture, named as in
                :meth:`available_layers`.
            path: Destination for an HDF5 cache. When omitted, activations are
                kept in memory.
            writer: Supply a writer directly, overriding ``path``.
            device: Device to move tensor leaves to. Other batch values are
                preserved. The model is not moved; do that yourself beforehand.
            progress: Show Ignite's progress bar over the batches.

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

        writer = self._create_writer(path, writer)
        step = ActivationStep(self._model, names, device)
        engine, timer = self._create_engine(step, writer, progress=progress)
        engine.run(loader, max_epochs=1)
        metadata = RunMetadata(
            model=type(self._model).__name__,
            parameters=self.parameter_count(),
            layers=names,
            samples=writer.sample_count,
            batch_size=getattr(loader, "batch_size", None) or 0,
            device=_model_device(self._model, device),
            seconds=timer.value(),
            created=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        return writer.close(metadata=metadata)
