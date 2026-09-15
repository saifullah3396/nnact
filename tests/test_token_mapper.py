"""Token-level activation capture, trimmed to real tokens only."""

import torch

from nnact._token_mapper import TokenActivationMapper


class _Echo(torch.nn.Module):
    """Identity model so activations equal the input exactly."""

    def __init__(self) -> None:
        super().__init__()
        self.identity = torch.nn.Identity()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.identity(input_ids.float())


def test_drops_padding_and_concatenates_real_tokens() -> None:
    """Only real-token rows survive, concatenated across samples and batches."""
    batch = {
        "input_ids": torch.tensor(
            [
                [[1.0], [2.0], [0.0]],
                [[3.0], [0.0], [0.0]],
            ]
        ),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
    }

    result = TokenActivationMapper(_Echo()).map([batch], "identity", progress=False)

    assert torch.equal(
        result.activations["identity"], torch.tensor([[1.0], [2.0], [3.0]])
    )


def test_output_transform_aligned_with_real_tokens() -> None:
    """The transformed output is trimmed the same way as layer activations."""
    batch = {
        "input_ids": torch.tensor([[[1.0], [2.0]]]),
        "attention_mask": torch.tensor([[1, 0]]),
    }

    def transform(output: object) -> dict[str, torch.Tensor]:
        assert isinstance(output, torch.Tensor)
        return {"logits": output}

    result = TokenActivationMapper(_Echo()).map(
        [batch], "identity", output_transform=transform, progress=False
    )

    assert torch.equal(result.output["logits"], torch.tensor([[1.0]]))


def test_output_transform_supports_multiple_arbitrary_fields() -> None:
    """output_transform may return any number of task-specific fields.

    Each field is trimmed and concatenated independently by name, so a
    caller can pull out e.g. next-token id and probability without nnact
    knowing anything about that task.
    """
    batch = {
        "input_ids": torch.tensor(
            [
                [[1.0], [2.0], [0.0]],
                [[3.0], [0.0], [0.0]],
            ]
        ),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
    }

    def transform(output: object) -> dict[str, torch.Tensor]:
        assert isinstance(output, torch.Tensor)
        batch_size, seq_len, _ = output.shape
        return {
            "next_token_id": torch.arange(batch_size * seq_len).reshape(
                batch_size, seq_len
            ),
            "next_token_probability": torch.full((batch_size, seq_len), 0.5),
        }

    result = TokenActivationMapper(_Echo()).map(
        [batch], "identity", output_transform=transform, progress=False
    )

    assert torch.equal(result.output["next_token_id"], torch.tensor([0, 1, 3]))
    assert torch.equal(
        result.output["next_token_probability"], torch.tensor([0.5, 0.5, 0.5])
    )
