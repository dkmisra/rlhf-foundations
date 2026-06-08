import random

import pytest

from rlhf2.tasks.block import Block, BlockTokenizer


@pytest.mark.parametrize(
    "prompt, mirror",
    [
        ("Red Blue Blue Red Green White", ["Green", "Red", "Blue", "Blue", "Red"]),
        ("Red White", ["Red"]),
        ("White", []),
        ("Red Blue Green", None),  # no trailing marker
        ("Red White Blue", None),  # marker not at the end
        ("White White", None),  # marker used as a block
    ],
)
def test_mirror(prompt, mirror):
    assert Block.mirror(prompt) == mirror


def test_tokenize_handles_concatenation_without_spaces():
    tok = BlockTokenizer()
    # SFT/RL concatenate "prompt + completion" with no separating space.
    concatenated = "Red Blue White" + "Blue Red" + tok.eos_token
    assert tok.tokenize(concatenated) == ["Red", "Blue", "White", "Blue", "Red", "."]


def test_tokenize_decode_round_trip():
    tok = BlockTokenizer()
    prompt = "Red Blue Green White"
    ids = [tok.token_to_idx[t] for t in tok.tokenize(prompt)]
    assert tok.decode(ids) == prompt


def test_prompt_token_prefix_is_stable_under_concatenation():
    tok = BlockTokenizer()
    prompt = "Red Blue Green White"
    completion = "Green Blue Red"
    prompt_tokens = tok.tokenize(prompt)
    joint_tokens = tok.tokenize(prompt + completion)
    assert joint_tokens[: len(prompt_tokens)] == prompt_tokens


def test_get_task_target_completion_is_the_mirror():
    task = Block(rng=random.Random(0))
    for _ in range(20):
        item = task.generate_sample()
        expected = " ".join(Block.mirror(item["prompt"]))
        assert item["target_completion"] == expected


def test_compute_reward_rewards_only_correct_mirror():
    task = Block(rng=random.Random(0))
    assert task.compute_reward("Red Blue White", "Blue Red") == 1.0
    assert task.compute_reward("Red Blue White", "Red Blue") == 0.0
    assert task.compute_reward("Red Blue White", "Blue") == 0.0
