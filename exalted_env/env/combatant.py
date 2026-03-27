from enum import Enum

from armor import Armor, noArmor
from character import Character
from weapon import Weapon, fists


class CombatState(Enum):
    ACTIVE = 1
    SURRENDERED = 0
    DEAD = -1


class Combatant:
    def __init__(
        self,
        character: Character,
        *,
        armor: Armor | None = None,
        weapon: Weapon = fists
    ) -> None:
        self.character = character

        # ephemeral combat state
        self.damage: int = 0
        self.initiative: int = 0
        self.onslaught_penalty: int = 0
        self.took_turn: bool = False
        self.state: CombatState = CombatState.ACTIVE

        # equipment
        self.armor = armor or noArmor
        self.weapon1 = weapon
        self.weapon2: Weapon | None = fists
