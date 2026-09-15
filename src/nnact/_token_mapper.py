"""A minimal, padding-aware companion to :class:`~nnact.ActivationMapper`.

:class:`~nnact.ActivationMapper` is deliberately general: it doesn't assume a
sequence dimension, an attention mask, or any particular downstream use, so it
keeps one dense ``(batch, *act_shape)`` tensor per layer -- every padding
position included.

For a token-level probe over a sequence model, that padding is pure waste:
with left-padded, fixed-``max_length`` batches, most of what gets hooked,
cast, and collected is never a real token. :class:`TokenActivationMapper` is a
narrower tool for exactly that case -- it knows every batch carries an
attention mask, trims each sample down to its real tokens before the
activation ever reaches host memory, and returns one flat, ungrouped
``(total_real_tokens, hidden)`` tensor per layer rather than a per-sample
grouping. Downstream label/grouping arrays are the caller's responsibility to
keep aligned to this same row order (concatenation in loader order, real
tokens only, per sample in the order `attention_mask` marks them).
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast, final

import torch
from torch import nn

from nnact._model import HookedModel
from nnact._types import TokenActivations
from nnact._utils import _as_list, _move_tensors


@final
class TokenActivationMapper:
    """Collect real-token activations, in memory, as flat tensors per layer."""

    def __init__(self, model: nn.Module) -> None:
        self._model = model

    def map(
        self,
        loader: Iterable[Mapping[str, Any]],
        layer_names: str | list[str],
        output_transform: Callable[[object, dict[str, Any]], dict[str, torch.Tensor]]
        | None = None,
        attention_mask_key: str = "attention_mask",
        device: torch.device | str | None = None,
        progress: bool = True,
    ) -> TokenActivations:
        """Run the model over `loader`, keeping only each sample's real tokens.

        Args:
            loader: A batch iterable (usually a `DataLoader`) yielding
                mappings of model keyword inputs. Every batch must include
                `attention_mask_key`, a `(batch, seq_len)` tensor with `1` at
                real-token positions and `0` at padding, and its keys are
                otherwise passed straight to `model.forward`.
            layer_names: Layer or layers to hook, as in
                `ActivationMapper.available_layers`.
            output_transform: Derive named per-token tensors from the model's
                raw output and that same call's `model_inputs` (after any
                `device` move), e.g. a
                :class:`~nnact.adapters.GenerativeLMAdapter`. Each returned
                field must have shape `(batch, seq_len, ...)`, broadcasting
                against `attention_mask` on its first two axes. Field names
                and further shape are entirely up to the transform — nothing
                here is specific to any task. Runs inside the same hooked
                forward pass as the layer captures, so getting both costs one
                pass over the data, not two, and each field is trimmed to
                real tokens the same way as layer activations, so it stays
                aligned with the token-level activation rows on axis 0. Omit
                to skip collecting output.
            attention_mask_key: The batch key holding the attention mask.
            device: Device to move tensor leaves to before the forward pass.
                The model is not moved; do that yourself beforehand.
            progress: Show a progress bar over the batches.

        Returns:
            A :class:`~nnact.TokenActivations` holding, per layer, one tensor
            of shape `(total_real_tokens, hidden)`: every sample's real-token
            rows (in `attention_mask` order), concatenated across samples and
            batches in loader order. Detached, moved to CPU, and cast to
            float16. Includes the same real-token rows of each output field
            if `output_transform` was given.

        Raises:
            ValueError: If a batch is missing `attention_mask_key`, or a
                hooked layer's or output field's shape doesn't broadcast
                against `(batch, seq_len)`.
        """
        names = _as_list(layer_names)
        hooked = HookedModel(self._model)
        hooked.eval()

        act_rows: dict[str, list[torch.Tensor]] = {name: [] for name in names}
        output_rows: dict[str, list[torch.Tensor]] = {}
        if progress:
            from tqdm.auto import tqdm

            loader = tqdm(loader, desc=f"{type(self._model).__name__} token activations")

        with torch.no_grad():
            for batch in loader:
                if attention_mask_key not in batch:
                    raise ValueError(
                        f"Batch is missing the attention mask key {attention_mask_key!r}."
                    )
                model_inputs = dict(batch)
                if device is not None:
                    model_inputs = cast(
                        "dict[str, Any]", _move_tensors(model_inputs, device)
                    )
                attention_mask = model_inputs[attention_mask_key]

                with hooked.capture(names):
                    output = hooked(**model_inputs)

                    for name in names:
                        activation = hooked.get_activation(name)
                        if activation.shape[:2] != attention_mask.shape:
                            raise ValueError(
                                f"Layer '{name}' output shape "
                                f"{tuple(activation.shape)} doesn't match "
                                f"attention_mask shape {tuple(attention_mask.shape)}."
                            )
                        mask = attention_mask.to(activation.device).bool()
                        for sample_index in range(activation.shape[0]):
                            act_rows[name].append(
                                activation[sample_index][mask[sample_index]]
                                .detach()
                                .to(device="cpu", dtype=torch.float16)
                            )

                    if output_transform is not None:
                        transformed = output_transform(output, model_inputs)
                        for field, tensor in transformed.items():
                            if tensor.shape[:2] != attention_mask.shape:
                                raise ValueError(
                                    f"Output field '{field}' shape "
                                    f"{tuple(tensor.shape)} doesn't match "
                                    f"attention_mask shape {tuple(attention_mask.shape)}."
                                )
                            mask = attention_mask.to(tensor.device).bool()
                            rows = output_rows.setdefault(field, [])
                            for sample_index in range(tensor.shape[0]):
                                rows.append(
                                    tensor[sample_index][mask[sample_index]]
                                    .detach()
                                    .to(device="cpu", dtype=torch.float16)
                                )

        activations = {
            name: torch.cat(rows, dim=0) if rows else torch.empty(0)
            for name, rows in act_rows.items()
        }
        collected_output = {
            field: torch.cat(rows, dim=0) for field, rows in output_rows.items()
        }
        return TokenActivations(activations=activations, output=collected_output)
