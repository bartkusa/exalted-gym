from enum import IntEnum


class CombatActions(IntEnum):
    SURRENDER = -1
    FULL_DEFENSE = 0
    WITHERING_ATTACK = 1
    DECISIVE_ATTACK = 2
