from __future__ import annotations

from typing import Literal

import torch


class AbstractTokenizer:
    """Character- or token-level encoder with padding id 0."""

    PAD_ID = 0

    def __init__(self, vocab: list[str], eos: str):
        if not vocab:
            raise ValueError("vocab must be non-empty")
        if len(set(vocab)) != len(vocab):
            raise ValueError("vocab tokens must be unique")
        if eos not in vocab:
            raise ValueError(f"eos token {eos!r} must be included in vocab")

        self.token_to_idx = {token: idx + 1 for idx, token in enumerate(vocab)}
        self.idx_to_token = {idx + 1: token for idx, token in enumerate(vocab)}
        self.eos_token = eos
        self.eos = self.token_to_idx[eos]

    @property
    def vocab_size(self) -> int:
        """Embedding vocabulary size (pad id 0 plus token ids 1..len(vocab))."""
        return len(self.token_to_idx) + 1

    def tokenize(self, prompt: str) -> list[str]:
        """Split a string into a list of token strings."""
        raise NotImplementedError()

    def decode(self, tokens: list[int]) -> str:
        """Decode token ids back to text, skipping padding and EOS."""
        raise NotImplementedError()

    def batch_encode(
        self, prompts: list[str], padding: bool = False
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Encode a batch of strings; optionally pad to the longest sequence."""
        if not prompts:
            return [], []

        batch_tokens = [self.tokenize(prompt) for prompt in prompts]
        max_len = max(len(tokens) for tokens in batch_tokens)

        input_ids: list[list[int]] = []
        attention_mask: list[list[float]] = []

        for tokens in batch_tokens:
            token_ids = [self.token_to_idx[token] for token in tokens]
            seq_len = len(token_ids)
            ids = list(token_ids)
            mask = [1.0] * seq_len
            if padding and seq_len < max_len:
                pad_len = max_len - seq_len
                ids.extend([self.PAD_ID] * pad_len)
                mask.extend([0.0] * pad_len)
            input_ids.append(ids)
            attention_mask.append(mask)

        return input_ids, attention_mask

    def batch_decode(self, token_ids_batch: list[list[int]]) -> list[str]:
        return [self.decode(token_ids) for token_ids in token_ids_batch]

    def __call__(
        self,
        texts: list[str],
        padding: bool = False,
        return_tensor: Literal["pt"] | None = None,
    ):
        input_ids, attention_mask = self.batch_encode(texts, padding=padding)
        if return_tensor == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class AbstractTask:
    tokenizer: AbstractTokenizer | None = None

    @classmethod
    def get_tokenizer(cls) -> AbstractTokenizer:
        if cls.tokenizer is None:
            raise ValueError(f"Task {cls.__name__} does not define a tokenizer")
        return cls.tokenizer

    def get_task(self):
        """Return a task dict with at least a ``prompt`` field."""
        raise NotImplementedError

    def compute_reward(self, prompt: str, completion: str) -> float:
        """Return 1.0 if balanced, else 0.0."""
        raise NotImplementedError
