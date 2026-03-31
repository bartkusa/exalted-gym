from enum import Enum
import math

from exalted_env.env.models.armor import Armor, noArmor
from exalted_env.env.models.character import Character
from exalted_env.env.models.weapon import Weapon, fists
from exalted_env.env.types import PZAgentId


class CombatState(Enum):
    ACTIVE = 1
    SURRENDERED = 0
    DEAD = -1


class Combatant:
    """
    Represents a `Character` in combat, with:
    - volatile and ephemeral combat stats.
    - mutable equipment
    """

    def __init__(
        self,
        agent: PZAgentId,
        character: Character,
        *,
        armor: Armor | None = None,
        weapon: Weapon = fists,
    ) -> None:
        self.agent: PZAgentId = agent

        self.character = character

        # region ==== Equipment ====

        self.armor = armor or noArmor
        """What armor is the combatant currently wearing?"""

        self.weapon1 = weapon
        """What main weapon is the combatant currently wielding?"""

        self.weapon2: Weapon | None = fists
        """
        What secondary weapon is the combatant currently wielding?
        
        (even someone with a spear or sword can punch/kick)
        """

        # endregion ==== Equipment ====

        # region ==== Ephemeral combat state ====

        self.damage: int = 0
        """
        How many health levels of damage have been taken. Can exceed the number of actual health levels.

        To get the combatant's current wound penalty, use the Combatant's `wound_penalty` property.
        """

        self.initiative: int = 0
        """How much initiative the character currently has. An abstract representation of advantage in combat."""

        self.defense_modifier: int = 0
        """
        Defense Modifier is _added_ to DV.
        - Full Defense increases it
        - Onslaught penalties decrease it

        This always resets to zero on the combatant's turn. Modifiers that live longer (or shorter) than that should be tracked elsewhere.
        """

        self.state: CombatState = CombatState.ACTIVE
        """Highest-level game state: are they still fighting, or not?"""

        self.took_turn: bool = False
        """Did the combatant take their turn this round, or not?"""

        self.is_crashed: bool = False
        """Whether this combatant is currently in Initiative Crash state."""

        self.crashed_by: PZAgentId | None = None
        """Agent id of the opponent who caused the current crash, if any."""

        self.crash_turns_remaining: int = 0
        """How many of this combatant's own turns remain before automatic crash recovery."""

        self.crash_recovered_round: int | None = None
        """Round number when this combatant most recently recovered from crash."""

        self.extra_turn_pending: bool = False
        """Whether this combatant should immediately act again due to Initiative Shift."""

        self.forced_attack_target: PZAgentId | None = None
        """During shift extra turn, attack actions may only target this opponent."""

        # endregion ==== Ephemeral combat state ====

    @property
    def dv_parry(self) -> int:
        """Returns the combatant's current Parry defense value, including all modifiers. Cannot be less than 0."""
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
        """Returns the combatant's current Evasion defense value, including all modifiers. Cannot be less than 0."""
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
        """Returns the combatant's Parry or Evasion DV, whichever is higher, including all modifiers. Cannot be less than 0."""
        return max(self.dv_evasion, self.dv_parry)

    @property
    def wound_penalty(self) -> int:
        """Returns the combatant's current wound penalty, based on their `damage` and the character's base `health_levels`."""
        assert self.damage >= 0

        count_health_levels = len(self.character.health_levels)
        assert count_health_levels > 0

        index = self.damage if (self.damage < count_health_levels) else -1
        return self.character.health_levels[index]
