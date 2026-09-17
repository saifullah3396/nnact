"""Sanity check: do nnact's forward hooks match the manual forward-pass unroll?

`run_qwen3_return_topk` below is copied as-is: it reimplements Qwen3's
forward pass by hand (causal mask, RoPE, per-layer attention/MLP) just to
grab two tensors per layer: `pre_mlp` (output of `post_attention_layernorm`,
before `mlp`) and the decoder layer's own output (`hidden_states` after the
residual add). Both are just the output of an existing named submodule, so
`nnact.HookedModel` should be able to capture the same tensors with a plain
forward hook and no reimplementation at all. This script proves that on
every layer of a real model, on real example text, before anything is built
on top of the assumption.

Run: python examples/scripts/probing/00_hook_sanity_check.py
"""

import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nnact import HookedModel
from nnact._logging import get_logger

logger = get_logger(__name__)

MODEL = "Qwen/Qwen3-1.7B"


@torch.no_grad()
def run_qwen3_return_topk(
    model, input_ids, attention_mask, return_hidden_states: bool = False
):
    """
    Reverse-engineered forward pass for dense Qwen3 (HF v4.57.3).

    Params:
        @model: Qwen3ForCausalLM
        @input_ids: (B, N)
        @attention_mask: (B, N) OR already-prepared dict with keys {"full_attention", "sliding_attention"} (4D masks)
        @return_hidden_states: if True, collects per-layer (BN, D) tensors

    Returns:
        Same MoE-compatible dictionary format you've been using.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Qwen3Model
    base = model.model

    # Move attention_mask to device (support either 2D or dict of masks)
    if isinstance(attention_mask, dict):
        causal_mask_mapping = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in attention_mask.items()
        }
        attention_mask_2d = None
    else:
        attention_mask_2d = (
            attention_mask.to(device) if attention_mask is not None else None
        )
        causal_mask_mapping = None

    # Token embeddings
    inputs_embeds = base.embed_tokens(input_ids)  # (B, N, D)
    B, N, D = inputs_embeds.shape  # noqa: RUF059

    # No-cache path
    past_key_values = None
    cache_position = torch.arange(0, N, device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)

    # Build the same causal_mask_mapping as Qwen3Model.forward does (if not already provided)
    if causal_mask_mapping is None:
        mask_kwargs = dict(  # noqa: C408
            config=base.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask_2d,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        has_sliding_layers = getattr(
            base, "has_sliding_layers", ("sliding_attention" in base.config.layer_types)
        )
        if has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = (
                create_sliding_window_causal_mask(**mask_kwargs)
            )

    # Shared RoPE
    position_embeddings = base.rotary_emb(inputs_embeds, position_ids)  # (cos, sin)

    hidden_states = inputs_embeds

    # API-compat: dense => empty MoE fields
    all_topk_experts = []
    all_topk_weights = []
    all_router_logits = []
    all_expert_outputs = []
    all_pre_mlp_hidden_states = []
    all_hidden_states = []

    # Unroll decoder layers (matches Qwen3DecoderLayer.forward)
    for layer in base.layers[: base.config.num_hidden_layers]:
        attn_mask = causal_mask_mapping[layer.attention_type]

        # ---- Attention ----
        residual = hidden_states
        hs_norm = layer.input_layernorm(hidden_states)

        attn_out, _ = layer.self_attn(
            hidden_states=hs_norm,
            position_embeddings=position_embeddings,
            attention_mask=attn_mask,
            past_key_values=None,
            cache_position=cache_position,
        )
        hidden_states = residual + attn_out

        # ---- MLP ----
        residual = hidden_states
        pre_mlp = layer.post_attention_layernorm(hidden_states)
        if return_hidden_states:
            all_pre_mlp_hidden_states.append(pre_mlp.reshape(-1, D).detach().cpu())

        mlp_out = layer.mlp(pre_mlp)
        hidden_states = residual + mlp_out

        if return_hidden_states:
            all_hidden_states.append(hidden_states.reshape(-1, D).detach().cpu())

    # Final norm + LM head
    hidden_states = base.norm(hidden_states)
    logits = model.lm_head(hidden_states)  # (B, N, V)

    return {
        "logits": logits,
        "all_topk_experts": all_topk_experts,
        "all_topk_weights": all_topk_weights,
        "all_pre_mlp_hidden_states": all_pre_mlp_hidden_states,
        "all_router_logits": all_router_logits,
        "all_hidden_states": all_hidden_states,
        "all_expert_outputs": all_expert_outputs,
    }


@torch.no_grad()
def run_hooked_forward(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_layers: int,
) -> dict[str, torch.Tensor]:
    """`HookedModel`-based stand-in for `run_qwen3_return_topk`, same return shape.

    Hooks every layer's `post_attention_layernorm` and the layer itself in a
    single forward pass, rather than re-running the model once per layer.
    """
    hooked = HookedModel(model)
    layer_names = [
        name
        for i in range(num_layers)
        for name in (f"model.layers.{i}.post_attention_layernorm", f"model.layers.{i}")
    ]
    with hooked.capture(layer_names):
        output = hooked(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )

    return {
        "logits": output.logits,
        "all_pre_mlp_hidden_states": {
            i: torch.from_numpy(
                hooked.get_activation(f"model.layers.{i}.post_attention_layernorm")
            )
            for i in range(num_layers)
        },
        "all_hidden_states": {
            i: torch.from_numpy(hooked.get_activation(f"model.layers.{i}"))
            for i in range(num_layers)
        },
    }


@torch.no_grad()
def _verify_hooked_forward_pass(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    run_forward_with_hidden_states: Callable[..., dict[str, torch.Tensor]],
) -> None:
    """Assert that `HookedModel` produces identical logits to the real model."""
    inputs = tokenizer(
        ["Hi! I am a dog and I like to bark", "Vegetables are good for"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=640,
    ).to(model.device)

    num_layers = model.config.num_hidden_layers

    original_results = model(**inputs, use_cache=False)
    manual_results = run_qwen3_return_topk(
        model,
        inputs["input_ids"],
        inputs["attention_mask"],
        return_hidden_states=True,
    )
    hooked_results = run_forward_with_hidden_states(
        model, inputs["input_ids"], inputs["attention_mask"], num_layers
    )

    assert torch.equal(original_results.logits, manual_results["logits"]), (
        "Error in manual forward"
    )
    assert torch.equal(original_results.logits, hooked_results["logits"]), (
        "Error in hooked forward"
    )

    for layer in range(num_layers):
        manual_pre_mlp = manual_results["all_pre_mlp_hidden_states"][layer]
        manual_hidden = manual_results["all_hidden_states"][layer].reshape(
            -1, manual_pre_mlp.shape[-1]
        )
        hooked_pre_mlp = hooked_results["all_pre_mlp_hidden_states"][layer].reshape(
            -1, manual_pre_mlp.shape[-1]
        )
        hooked_hidden = hooked_results["all_hidden_states"][layer].reshape(
            -1, manual_pre_mlp.shape[-1]
        )

        assert torch.equal(manual_pre_mlp, hooked_pre_mlp), (
            f"pre_mlp mismatch at layer {layer}: hooked capture does not match "
            "manual forward"
        )
        assert torch.equal(manual_hidden, hooked_hidden), (
            f"hidden_states mismatch at layer {layer}: hooked capture does not "
            "match manual forward"
        )
        logger.info(
            "[layer %2d] pre_mlp %s and hidden_states %s match exactly",
            layer,
            tuple(manual_pre_mlp.shape),
            tuple(manual_hidden.shape),
        )

    loss = (
        ForCausalLMLoss(
            hooked_results["logits"],
            torch.where(
                inputs["input_ids"] == tokenizer.pad_token_id,
                torch.tensor(-100),
                inputs["input_ids"],
            ),
            hooked_results["logits"].size(-1),
        )
        .detach()
        .cpu()
        .item()
    )

    logger.info("LM loss: %s", loss)
    logger.info("Layers verified: %s", num_layers)
    logger.info(
        "Verified hooked forward pass exactly matches the manual forward and the "
        "original model output on every layer!"
    )


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()

    num_layers = model.config.num_hidden_layers
    logger.info("model=%s num_hidden_layers=%s", MODEL, num_layers)

    _verify_hooked_forward_pass(model, tokenizer, run_hooked_forward)


if __name__ == "__main__":
    main()
