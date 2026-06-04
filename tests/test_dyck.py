import random

import pytest

from tasks.dyck import Dyck


@pytest.mark.parametrize(
    "text, balanced",
    [
        ("", True),
        ("()", True),
        ("([{}])", True),
        ("(]", False),
        ("(", False),
        (")(", False),
        ("a", False),
    ],
)
def test_is_balanced(text, balanced):
    assert Dyck.is_balanced(text) is balanced


def test_get_task_target_completion_balances_prompt():
    task = Dyck(rng=random.Random(0))
    for _ in range(20):
        item = task.get_task()
        assert Dyck.is_balanced(item["prompt"] + item["target_completion"])
        assert item["open_stack"], "a Dyck task should have unmatched open brackets"


def test_compute_reward_rewards_only_balanced_completions():
    task = Dyck(rng=random.Random(0))
    assert task.compute_reward("([", "])") == 1.0
    assert task.compute_reward("([", ")") == 0.0
