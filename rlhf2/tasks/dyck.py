import random

from .abstract import AbstractTokenizer, AbstractTask

_OPEN = "([{"
_CLOSE = ")]}"
_BRACKETS = _OPEN + _CLOSE


class DyckTokenizer(AbstractTokenizer):

    def __init__(self):
        super().__init__(vocab=list(_BRACKETS))

    def tokenize(self, prompt: str) -> list[str]:
        tokens = list(prompt)
        for token in tokens:
            if token not in self.token_to_idx:
                raise ValueError(f"Unknown token {token!r} in prompt {prompt!r}")
        return tokens

    def decode(self, tokens: list[int]) -> str:
        chars = []
        for token_id in tokens:
            if token_id == self.PAD_ID:
                continue
            if self.eos is not None and token_id == self.eos:
                break
            chars.append(self.idx_to_token[token_id])
        return "".join(chars)


class Dyck(AbstractTask):
    """Complete a partial bracket string with valid closings.

    Bracket types: ``( )``, ``[ ]``, ``{ }``.
    """

    OPEN = _OPEN
    CLOSE = _CLOSE
    OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
    CLOSE_TO_OPEN = {")": "(", "]": "[", "}": "{"}
    BRACKETS = _BRACKETS

    tokenizer = DyckTokenizer()

    def __init__(
        self,
        mean_prompt_length: int = 8,
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

    def get_task(self) -> dict:
        """Sample a prompt and return metadata for balancing brackets."""
        for _ in range(100):
            length = self.rng.randint(self.prompt_length_min, self.prompt_length_max)
            prompt, open_stack = self._sample_prefix(length)
            if open_stack:
                return {
                    "prompt": prompt,
                    "open_stack": list(open_stack),
                    "target_completion": "".join(
                        self.OPEN_TO_CLOSE[b] for b in reversed(open_stack)
                    ),
                    "max_completion_length": self.max_completion_length,
                }

        open_bracket = self.rng.choice(self.OPEN)
        return {
            "prompt": open_bracket,
            "open_stack": [open_bracket],
            "target_completion": self.OPEN_TO_CLOSE[open_bracket],
            "max_completion_length": self.max_completion_length,
        }

    def compute_reward(self, prompt: str, completion: str) -> float:
        return 1.0 if self.is_balanced(prompt + completion) else 0.0

    @classmethod
    def is_balanced(cls, text: str) -> bool:
        stack: list[str] = []
        for char in text:
            if char in cls.OPEN:
                stack.append(char)
            elif char in cls.CLOSE_TO_OPEN:
                if not stack or stack.pop() != cls.CLOSE_TO_OPEN[char]:
                    return False
            else:
                return False
        return len(stack) == 0

    def _sample_prefix(self, length: int) -> tuple[str, list[str]]:
        """Random walk over bracket ops; returns prefix and unmatched opens."""
        stack: list[str] = []
        chars: list[str] = []
        for _ in range(length):
            can_close = bool(stack)
            if can_close and self.rng.random() < 0.5:
                open_bracket = stack.pop()
                chars.append(self.OPEN_TO_CLOSE[open_bracket])
            else:
                open_bracket = self.rng.choice(self.OPEN)
                stack.append(open_bracket)
                chars.append(open_bracket)
        return "".join(chars), stack
