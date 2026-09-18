from __future__ import annotations

import numpy as np
import torch
from ignite.engine import Engine

from nnact._model._hooked import HookedModel, tensor_to_numpy
from nnact._outputs._protocols import ActivationBatch, ModelOutput
from nnact._outputs._sequence import SequenceActivationOutput
from nnact._steps._utils import _move_tensors, _top_prediction


class SequenceActivationStep:
    """Ignite step function: one forward pass, one row of output per sample.

    Callable as an Ignite ``process_function`` -- ``Engine(step)`` invokes
    it once per batch, and its return value becomes ``engine.state.output``
    for that iteration's handlers (e.g. an
    :class:`~nnact._steps._accumulator.ActivationAccumulator`).
    """

    def __init__(
        self,
        hooked_model: HookedModel,
        layer_names: list[str],
        device: torch.device | str | None,
    ) -> None:
        """Bind this step to one model and one set of layers.

        Args:
            hooked_model: The model to run, already wrapped for activation
                capture.
            layer_names: Layers to capture on every call. Validated
                immediately via :meth:`~nnact._model._hooked.HookedModel.check_layers`,
                so a typo fails at construction rather than on first batch.
            device: Device the model and every batch are moved to before
                the forward pass. ``None`` leaves both where they already
                are.
        """
        self._hooked_model = hooked_model
        self._layer_names = layer_names
        self._device = device
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
    ) -> SequenceActivationOutput:
        """Run one batch through the model and capture its activations.

        Args:
            engine: Unused directly, but required by Ignite's
                ``process_function`` calling convention.
            batch: The batch to run. Every tensor field is moved to
                :attr:`_device` first.

        Returns:
            One row per sample in ``batch``, holding the model's own
            prediction, top softmax probability, optional loss and
            ground-truth labels, and the captured layers' activations.

        Raises:
            AssertionError: If ``batch`` is not an ``ActivationBatch``.
        """
        assert isinstance(batch, ActivationBatch), (
            "batch passed to the generator must be an ActivationBatch."
        )

        if self._device is not None:
            batch = _move_tensors(value=batch, device=self._device)

        with self._hooked_model.capture(layer_names=self._layer_names):
            raw_output: ModelOutput = self._hooked_model(
                **batch.model_input.as_model_kwargs()
            )

            activations = {
                name: self._hooked_model.get_activation(layer_name=name)
                for name in self._layer_names
            }

        prediction, top_probability = _top_prediction(logits=raw_output.logits)

        return SequenceActivationOutput(
            prediction=prediction,
            top_probability=top_probability,
            loss=(
                tensor_to_numpy(tensor=raw_output.loss)
                if raw_output.loss is not None
                else None
            ),
            labels=(
                np.asarray(batch.activation_labels)
                if batch.activation_labels is not None
                else None
            ),
            activations=activations,
        )
