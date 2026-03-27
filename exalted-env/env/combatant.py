from armor import Armor, noArmor
from character import Character
from weapon import Weapon, fists


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
        self.initiative: int = 3
        self.onslaught_penalty: int = 0
        self.took_turn: bool = False
        self.state: "Active" | "Dead" | "Surrendered" = "Active"

        # equipment
        self.armor = armor
        self.weapon1 = weapon
        self.weapon2: Weapon | None = fists
