from enum import Enum
import math

from exalted_env.env.models.armor import Armor, noArmor
from exalted_env.env.models.character import Character
from exalted_env.env.models.weapon import Weapon, fists


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
        # Defense Modifier is added to DV. Onslaught Penalties decrease it; Full Defense increases it.
        # This resets to zero on the combatant's turn.
        self.defense_modifier: int = 0
        self.took_turn: bool = False
        self.state: CombatState = CombatState.ACTIVE

        # equipment
        self.armor = armor or noArmor
        self.weapon1 = weapon
        self.weapon2: Weapon | None = fists

    @property
    def dv_parry(self) -> int:
        base_pool = self.character.dexterity + self.character.melee
        return max(
            0,
            (
                math.ceil(base_pool / 2)
                + self.weapon1.defense
                + self.wound_penalty
                + self.defense_modifier
            ),
        )

    @property
    def dv_evasion(self) -> int:
        base_pool = self.character.dexterity + self.character.dodge
        return max(
            0,
            (
                math.ceil(base_pool / 2)
                + self.weapon1.defense
                + self.armor.mobility
                + self.wound_penalty
                + self.defense_modifier
            ),
        )

    @property
    def dv(self) -> int:
        return max(self.dv_evasion, self.dv_parry)

    @property
    def wound_penalty(self) -> int:
        assert self.damage >= 0

        count_health_levels = len(self.character.health_levels)
        assert len(count_health_levels) > 0

        index = self.damage if (self.damage < count_health_levels) else -1
        return self.character.health_levels[index]
