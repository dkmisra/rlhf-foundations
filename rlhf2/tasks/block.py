import random

from rlhf2.tasks.abstract import AbstractTokenizer, AbstractTask

_COLORS = ["Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Black"]
MARKER = "White"
EOS_TOKEN = "."


class BlockTokenizer(AbstractTokenizer):
    """Word-level tokenizer over color blocks plus a marker and EOS.

    Tokens are matched greedily (longest first) while skipping whitespace, so
    directly concatenated strings such as ``prompt + completion`` tokenize the
    same way regardless of whether a space separates the two pieces.
    """

    def __init__(self):
        super().__init__(vocab=list(_COLORS) + [MARKER, EOS_TOKEN], eos=EOS_TOKEN)
        # Longest token first so greedy matching never stops on a shorter prefix.
        self._match_order = sorted(self.token_to_idx, key=len, reverse=True)

    def tokenize(self, prompt: str) -> list[str]:
        tokens: list[str] = []
        i, n = 0, len(prompt)
        while i < n:
            if prompt[i].isspace():
                i += 1
                continue
            for token in self._match_order:
                if prompt.startswith(token, i):
                    tokens.append(token)
                    i += len(token)
                    break
            else:
                raise ValueError(
                    f"Unknown token at position {i} in prompt {prompt!r}"
                )
        return tokens

    def decode(self, tokens: list[int]) -> str:
        words = []
        for token_id in tokens:
            if token_id == self.PAD_ID:
                continue
            if token_id == self.eos:
                break
            words.append(self.idx_to_token[token_id])
        return " ".join(words)


class Block(AbstractTask):
    """Output the mirror image of a sequence of colored blocks.

    The prompt is a sequence of color blocks terminated by a special
    ``MARKER`` (``White``). The target completion is the reversed sequence of
    colors (excluding the marker).

    Example::

        prompt:            Red Blue Blue Red Green White
        target_completion: Green Red Blue Blue Red
    """

    COLORS = _COLORS
    MARKER = MARKER

    tokenizer = BlockTokenizer()

    def __init__(
        self,
        mean_prompt_length: int = 5,
        prompt_length_min: int | None = None,
        prompt_length_max: int | None = None,
        max_completion_length: int = 16,
        rng: random.Random | None = None,
    ):
        if mean_prompt_length < 1:
            raise ValueError("mean_prompt_length must be at least 1")
        if max_completion_length < 1:
            raise ValueError("max_completion_length must be at least 1")

        self.mean_prompt_length = mean_prompt_length
        self.max_completion_length = max_completion_length
        self.rng = rng or random.Random()

        lo = prompt_length_min if prompt_length_min is not None else max(
            1, int(round(mean_prompt_length * 0.5))
        )
        hi = prompt_length_max if prompt_length_max is not None else max(
            lo, int(round(mean_prompt_length * 1.5))
        )
        if lo > hi:
            raise ValueError("prompt_length_min must be <= prompt_length_max")
        self.prompt_length_min = lo
        self.prompt_length_max = hi

    def generate_sample(self) -> dict:
        """Sample a block sequence and its mirror image."""
        length = self.rng.randint(self.prompt_length_min, self.prompt_length_max)
        blocks = [self.rng.choice(self.COLORS) for _ in range(length)]
        prompt = " ".join(blocks + [self.MARKER])
        target_completion = " ".join(reversed(blocks))
        return {
            "prompt": prompt,
            "blocks": blocks,
            "target_completion": target_completion,
            "max_completion_length": self.max_completion_length,
        }

    def compute_reward(self, prompt: str, completion: str) -> float:
        expected = self.mirror(prompt)
        if expected is None:
            return 0.0
        return 1.0 if self.tokenizer.tokenize(completion) == expected else 0.0

    @classmethod
    def mirror(cls, prompt: str) -> list[str] | None:
        """Return the reversed color blocks before the marker, or None if invalid.

        A valid prompt is a (possibly empty) run of colors followed by exactly
        one marker at the end.
        """
        tokens = cls.tokenizer.tokenize(prompt)
        if not tokens or tokens[-1] != cls.MARKER:
            return None
        blocks = tokens[:-1]
        if any(block not in cls.COLORS for block in blocks):
            return None
        return list(reversed(blocks))
