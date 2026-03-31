import random


class RollResult:
    def __init__(self, successes: int, rolls: list[int]) -> None:
        self.rolls = rolls
        self.sux = successes


def roll_d10s(
    pool: int, *, double: list[int] | None = None, target: int = 7
) -> RollResult:
    if pool <= 0:
        return RollResult(0, [])

    double_values = [10] if double is None else double
    rolls = []
    successes = 0
    for _ in range(pool):
        result = random.randint(1, 10)
        rolls.append(result)

        if result >= target:
            successes += 2 if result in double_values else 1

    return RollResult(successes, rolls)
