from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast, final

import torch
from torch import nn

from nnact._model import HookedModel
from nnact._types import SampleActivations
from nnact._utils import _as_list, _move_tensors

if TYPE_CHECKING:
    import pandas as pd


@final
class ActivationMapper:
    """Capture sample-level activations: one dense tensor per layer.

    For a token-level probe over a sequence model, see
    :class:`~nnact.TokenActivationMapper` instead, which trims padding and
    returns real tokens only.
    """

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

    def map(
        self,
        loader: Iterable[Mapping[str, Any]],
        layer_names: str | list[str],
        output_transform: Callable[[object, dict[str, Any]], dict[str, torch.Tensor]]
        | None = None,
        device: torch.device | str | None = None,
        progress: bool = True,
    ) -> SampleActivations:
        """Run the model over pre-batched inputs and collect its activations.

        Args:
            loader: A caller-configured batch iterable, usually a DataLoader,
                yielding mappings of model keyword inputs. Every key is passed
                to the model as a keyword argument, so the model's ``forward``
                parameter names define their meaning. The loader owns dataset
                access, batching, and collation.
            layer_names: Layer or layers to capture, named as in
                :meth:`available_layers`.
            output_transform: Derive named per-sample tensors from the
                model's raw output and that same call's ``model_inputs``
                (after any ``device`` move), e.g. a
                :class:`~nnact.adapters.GenerativeLMAdapter`. Field names and
                shapes are entirely up to the transform — nothing here is
                specific to any task. Runs inside the same hooked forward
                pass as the layer captures, so getting both costs one pass
                over the data, not two. Omit to skip collecting output.
            device: Device to move tensor leaves to. Other batch values are
                preserved. The model is not moved; do that yourself beforehand.
            progress: Show a progress bar over the batches.

        Returns:
            A :class:`~nnact.SampleActivations` holding one stacked tensor per
            layer, of shape ``(n_samples, *act_shape)``, in loader order, plus
            one stacked tensor per output field if ``output_transform`` was
            given. All tensors are detached, moved to CPU, and cast to
            ``float16``.

        Raises:
            ValueError: If a requested layer is not a module of the model. The
                check runs before the first forward pass, so a typo fails
                immediately rather than after a long extraction.
        """
        names = _as_list(layer_names)
        self._check_layers(names)

        hooked = HookedModel(self._model)
        hooked.eval()

        act_batches: list[dict[str, torch.Tensor]] = []
        output_batches: list[dict[str, torch.Tensor]] = []
        if progress:
            from tqdm.auto import tqdm

            loader = tqdm(loader, desc=f"{type(self._model).__name__} activations")

        with torch.no_grad():
            for batch in loader:
                model_inputs = dict(batch)
                if device is not None:
                    model_inputs = cast(
                        "dict[str, Any]", _move_tensors(model_inputs, device)
                    )

                with hooked.capture(names):
                    output = hooked(**model_inputs)
                    act_batches.append(
                        {
                            name: hooked.get_activation(name)
                            .detach()
                            .to(device="cpu", dtype=torch.float16)
                            for name in names
                        }
                    )
                    if output_transform is not None:
                        transformed = output_transform(output, model_inputs)
                        output_batches.append(
                            {
                                field: tensor.detach().to(
                                    device="cpu", dtype=torch.float16
                                )
                                for field, tensor in transformed.items()
                            }
                        )

        activations = {
            name: torch.cat([batch[name] for batch in act_batches], dim=0)
            for name in names
        }
        collected_output = (
            {
                field: torch.cat([batch[field] for batch in output_batches], dim=0)
                for field in output_batches[0]
            }
            if output_batches
            else {}
        )
        return SampleActivations(activations=activations, output=collected_output)
