from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from nnact._outputs._sequence import SequenceActivationOutput
from nnact._outputs._utils import _assert_leading_shape, _assert_shape


@dataclass(frozen=True, kw_only=True)
class TokenActivationOutput:
    """One batch's worth of activations, one row per real (non-padding) token.

    Returned by :class:`~nnact._steps._token.TokenActivationStep` for each
    batch, built via :meth:`from_sequence` from a
    :class:`~nnact._outputs._sequence.SequenceActivationOutput` by dropping
    every padding position. Multiple samples' tokens are concatenated into
    one flat ``(num_tokens, ...)`` layout; :attr:`offsets` records where
    each sample's tokens begin and end within it.

    Attributes:
        prediction: The model's own argmax prediction, one per real token,
            shape ``(num_tokens,)``.
        top_probability: Softmax probability of ``prediction``, same shape.
        offsets: Sample boundaries into the flat token layout, shape
            ``(batch_size + 1,)``. Sample ``i``'s tokens are
            ``[offsets[i], offsets[i + 1])``.
        metadata: Named per-token values a caller attached to the batch's
            samples (see :class:`~nnact._outputs._protocols.TokenActivationSample`),
            each masked down to real tokens the same way ``token_ids`` is.
            ``nnact`` never interprets the keys or values.
        activations: Captured layer outputs, keyed by layer name, each
            shaped ``(num_tokens, *feature)``.
        token_ids: The input token id per real token, if provided.
        tokens: The decoded token string per real token, if provided.
    """

    prediction: np.ndarray
    top_probability: np.ndarray
    offsets: np.ndarray
    metadata: dict[str, np.ndarray] | None = None
    activations: dict[str, np.ndarray] = field(default_factory=dict)
    token_ids: np.ndarray | None = None
    tokens: np.ndarray | None = None

    def __post_init__(self):
        """Validate every field's shape and ``offsets``'s invariants.

        Raises:
            AssertionError: If any field's shape doesn't match
                ``(num_tokens,)`` (or a layer's or metadata value's leading
                shape doesn't), if ``tokens`` has the wrong length, or if
                ``offsets`` isn't a nondecreasing integer array starting at
                ``0`` and ending at ``num_tokens``.
        """
        _assert_shape(name="prediction", tensor=self.prediction, shape=(None,))

        num_tokens = self.prediction.shape[0]

        _assert_shape(
            name="top_probability", tensor=self.top_probability, shape=(num_tokens,)
        )
        _assert_shape(name="offsets", tensor=self.offsets, shape=(None,))
        _assert_shape(name="token_ids", tensor=self.token_ids, shape=(num_tokens,))

        if self.tokens is not None:
            assert len(self.tokens) == num_tokens, (
                f"tokens must have {num_tokens} entries, got {len(self.tokens)}"
            )

        if self.metadata is not None:
            for key, value in self.metadata.items():
                _assert_shape(name=f"metadata[{key!r}]", tensor=value, shape=(num_tokens,))

        for name, tensor in self.activations.items():
            _assert_leading_shape(
                name=f"activation '{name}'",
                tensor=tensor,
                shape=(num_tokens,),
            )

        # Offsets must contain integer indices.
        assert self.offsets.dtype.kind in "iu"

        # Offsets must contain at least the initial boundary.
        assert self.offsets.size >= 1

        # The first sequence must start at token index 0.
        assert self.offsets[0].item() == 0

        # Sequence boundaries must be nondecreasing.
        assert np.all(self.offsets[1:] >= self.offsets[:-1])

        # The final boundary must equal the total number of tokens.
        assert self.offsets[-1].item() == num_tokens

    @classmethod
    def from_sequence(
        cls,
        output: SequenceActivationOutput,
        mask: np.ndarray,
        metadata: dict[str, np.ndarray] | None = None,
        token_ids: np.ndarray | None = None,
        tokens: np.ndarray | None = None,
    ) -> TokenActivationOutput:
        """Build a token-level output by dropping ``output``'s padding.

        Args:
            output: The sequence-level batch to flatten.
            mask: Boolean (or 0/1) array, shape
                ``(output.batch_size, output.sequence_length)`` -- ``True``
                (or nonzero) at every real token to keep.
            metadata: Named per-token values, each shaped like ``mask``,
                masked the same way -- ``nnact`` never interprets the keys
                or values.
            token_ids: Input token id per position, same shape as ``mask``,
                masked the same way.
            tokens: Decoded token string per position, same shape as
                ``mask``, masked the same way.

        Returns:
            A new instance holding only the positions ``mask`` keeps,
            concatenated across samples in order, with :attr:`offsets`
            recording each sample's boundaries in the result.
        """
        _assert_shape(
            name="mask",
            tensor=mask,
            shape=(output.batch_size, output.sequence_length),
        )
        if metadata is not None:
            for key, value in metadata.items():
                _assert_shape(
                    name=f"metadata[{key!r}]",
                    tensor=value,
                    shape=(output.batch_size, output.sequence_length),
                )
        _assert_shape(
            name="token_ids",
            tensor=token_ids,
            shape=(output.batch_size, output.sequence_length),
        )

        mask = mask.astype(bool)
        lengths = mask.sum(axis=-1)
        offsets = np.concatenate([np.zeros(1, dtype=lengths.dtype), lengths.cumsum()])

        return cls(
            prediction=output.prediction[mask],
            top_probability=output.top_probability[mask],
            offsets=offsets,
            metadata=(
                {key: value[mask] for key, value in metadata.items()}
                if metadata is not None
                else None
            ),
            activations={
                name: tensor[mask] for name, tensor in output.activations.items()
            },
            token_ids=token_ids[mask] if token_ids is not None else None,
            tokens=tokens[mask] if tokens is not None else None,
        )

    @cached_property
    def batch_size(self) -> int:
        """Number of samples this output was built from."""
        return self.offsets.size - 1

    @cached_property
    def num_tokens(self) -> int:
        """Total real tokens across every sample."""
        return self.prediction.shape[0]

    @cached_property
    def sequence_lengths(self) -> np.ndarray:
        """Real token count per sample, shape ``(batch_size,)``."""
        return self.offsets[1:] - self.offsets[:-1]

    def __len__(self) -> int:
        """Number of samples this output was built from -- see :attr:`batch_size`."""
        return self.batch_size
