from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class LayerActivation:
    layer_name: str
    tensor: torch.Tensor


@dataclass(frozen=True, slots=True)
class SampleActivations:
    """Result of :meth:`~nnact.ActivationMapper.map`.

    One dense ``(n_samples, *act_shape)`` tensor per captured layer, plus
    whatever per-sample fields ``output_transform`` produced from the model's
    own output for the same forward passes, keyed by field name.
    """

    activations: dict[str, torch.Tensor] = field(default_factory=dict)
    output: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenActivations:
    """Result of :meth:`~nnact.TokenActivationMapper.map`.

    One flat ``(total_real_tokens, hidden)`` tensor per captured layer —
    padding positions dropped, real tokens concatenated in loader order —
    plus whatever per-token fields ``output_transform`` produced from the
    model's own output, keyed by field name and trimmed to real tokens the
    same way as the activations.
    """

    activations: dict[str, torch.Tensor] = field(default_factory=dict)
    output: dict[str, torch.Tensor] = field(default_factory=dict)
