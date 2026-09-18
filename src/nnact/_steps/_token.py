from __future__ import annotations

import numpy as np
import torch
from ignite.engine import Engine
from transformers import PreTrainedTokenizerBase

from nnact._model._hooked import HookedModel, tensor_to_numpy
from nnact._outputs._protocols import ActivationBatch, ModelOutput
from nnact._outputs._sequence import SequenceActivationOutput
from nnact._outputs._token import TokenActivationOutput
from nnact._steps._utils import _move_tensors, _top_prediction


class TokenActivationStep:
    """Ignite step function: one forward pass, one row of output per real token.

    Callable as an Ignite ``process_function`` -- ``Engine(step)`` invokes
    it once per batch, and its return value becomes ``engine.state.output``
    for that iteration's handlers (e.g. an
    :class:`~nnact._steps._accumulator.ActivationAccumulator`). Padding
    positions (``attention_mask == 0``) are dropped before returning, so a
    padded batch never contributes rows for its padding.
    """

    def __init__(
        self,
        hooked_model: HookedModel,
        layer_names: list[str],
        device: torch.device | str = "cpu",
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        """Bind this step to one model, one set of layers, and an optional tokenizer.

        Args:
            hooked_model: The model to run, already wrapped for activation
                capture.
            layer_names: Layers to capture on every call. Validated
                immediately via :meth:`~nnact._model._hooked.HookedModel.check_layers`,
                so a typo fails at construction rather than on first batch.
            device: Device the model and every batch are moved to before
                the forward pass.
            tokenizer: If given, each output row also carries the decoded
                token string. Omit to skip decoding (and the ``tokens``
                column) entirely.
        """
        self._hooked_model = hooked_model
        self._layer_names = layer_names
        self._device = device
        self._tokenizer = tokenizer
        self._hooked_model.check_layers(layer_names=self._layer_names)

    def _prepare_model(self):
        """Move the wrapped model to its target device and set eval mode.

        Called once before the Ignite engine starts iterating, not per
        batch -- see :meth:`~nnact._steps._runner.ActivationStepRunner.run`.
        """
        self._hooked_model = self._hooked_model.to(self._device).eval()

    @torch.no_grad()
    def __call__(
        self,
        engine: Engine,
        batch: ActivationBatch,
    ) -> TokenActivationOutput:
        """Run one batch through the model and capture its per-token activations.

        Args:
            engine: Unused directly, but required by Ignite's
                ``process_function`` calling convention.
            batch: The batch to run. Every tensor field is moved to
                :attr:`_device` first.

        Returns:
            One row per real (non-padding) token across every sample in
            ``batch``, holding the model's own prediction, top softmax
            probability, optional loss and ground-truth labels, the
            captured layers' activations, and -- when this step was built
            with a tokenizer -- the decoded token string.

        Raises:
            AssertionError: If ``batch`` is not an ``ActivationBatch``.
        """
        assert isinstance(batch, ActivationBatch), (
            "batch passed to the generator must be an ActivationBatch."
        )

        if self._device is not None:
            batch = _move_tensors(value=batch, device=self._device)

        token_ids = batch.model_input.input_ids
        attention_mask = batch.model_input.attention_mask

        with self._hooked_model.capture(layer_names=self._layer_names):
            raw_output: ModelOutput = self._hooked_model(
                **batch.model_input.as_model_kwargs()
            )

            activations = {
                name: self._hooked_model.get_activation(layer_name=name)
                for name in self._layer_names
            }

        prediction, top_probability = _top_prediction(logits=raw_output.logits)

        sequence_output = SequenceActivationOutput(
            prediction=prediction,
            top_probability=top_probability,
            loss=(
                tensor_to_numpy(tensor=raw_output.loss)
                if raw_output.loss is not None
                else None
            ),
            activations=activations,
        )

        tokens = None
        if self._tokenizer is not None:
            tokens = [
                self._tokenizer.convert_ids_to_tokens(ids) for ids in token_ids.tolist()
            ]
            assert isinstance(tokens, list), f"List[str] expected, got {type(tokens)}"

        return TokenActivationOutput.from_sequence(
            output=sequence_output,
            mask=tensor_to_numpy(tensor=attention_mask),
            labels=(
                np.asarray(batch.activation_labels)
                if batch.activation_labels is not None
                else None
            ),
            token_ids=tensor_to_numpy(tensor=token_ids),
            tokens=np.asarray(tokens) if tokens is not None else None,
        )
