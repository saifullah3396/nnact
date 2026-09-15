"""Standardized output fields for generative (causal) language models.

`ActivationMapper`/`TokenActivationMapper` accept an `output_transform` that
turns a model's raw forward output into named tensors, collected in the same
pass as layer activations. This module is the first *model adapter*: a
reusable `output_transform` for the one model family in scope for now --
generative LLMs, whose forward output exposes `.logits` of shape
`(batch, seq_len, vocab)`.

It derives the same three per-token quantities the original probing pipeline
computed from raw logits (`ForwardPredictions.forward_predictions` in the
prior tensor-based trainer): the argmax next-token id, that token's
probability, and the log-probability the model actually assigned to the real
next token (which `perplexity()` below turns into per-token perplexity).
Nothing here is specific to any one architecture -- any causal LM exposing
`.logits` works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class GenerativeLMAdapter:
    """`output_transform` for causal LMs: derives next-token prediction fields.

    Example:
        >>> adapter = GenerativeLMAdapter(input_ids_key="input_ids")
        >>> mapper.map(loader, layer_names, output_transform=adapter)  # doctest: +SKIP
    """

    #: Batch key holding the input ids, used to find each position's actual
    #: next token (shifted by one) for `next_token_log_probability`.
    input_ids_key: str = "input_ids"

    def __call__(self, model_output: object, model_inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Derive per-token prediction fields from one forward pass's output.

        Args:
            model_output: The model's raw forward return value; must expose
                `.logits` of shape `(batch, seq_len, vocab)` (as every HF
                `*ForCausalLM` output does).
            model_inputs: That same call's keyword inputs, used to read
                `input_ids_key` for the actual-next-token lookup.

        Returns:
            A dict with three `(batch, seq_len)` tensors:

            - `next_token_id`: the model's argmax prediction at each position.
            - `next_token_probability`: softmax probability of that argmax.
            - `next_token_log_probability`: log-probability the model gave the
              token that actually follows each position (shifted by one; NaN
              at each sequence's last position, where nothing follows). Feed
              this to `perplexity` for a per-token perplexity tensor.

        Raises:
            AttributeError: If `model_output` has no `.logits`.
            KeyError: If `model_inputs` has no `input_ids_key`.
        """
        logits = model_output.logits.to(torch.float32)  # type: ignore[union-attr]
        input_ids = model_inputs[self.input_ids_key]

        log_normalizer = torch.logsumexp(logits, dim=-1)
        top_logit, top_id = logits.max(dim=-1)

        actual_next_ids = input_ids.to(logits.device)[:, 1:]
        actual_next_logit = (
            logits[:, :-1].gather(-1, actual_next_ids.unsqueeze(-1)).squeeze(-1)
        )
        next_token_log_probability = torch.full_like(log_normalizer, float("nan"))
        next_token_log_probability[:, :-1] = actual_next_logit - log_normalizer[:, :-1]

        return {
            "next_token_id": top_id.to(torch.float32),
            "next_token_probability": (top_logit - log_normalizer).exp(),
            "next_token_log_probability": next_token_log_probability,
        }


def perplexity(next_token_log_probability: torch.Tensor) -> torch.Tensor:
    """Per-token perplexity: ``exp`` of each position's negative log-likelihood.

    Args:
        next_token_log_probability: A `GenerativeLMAdapter` output field of
            the same name, or any tensor shaped like it.

    Returns:
        A tensor of the same shape, NaN wherever the input is NaN (each
        sequence's last position).
    """
    return (-next_token_log_probability).exp()
