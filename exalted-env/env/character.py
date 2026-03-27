class Character:
    def __init__(
        self,
        *,
        name: str,
        # attributes
        strength: int,
        dexterity: int,
        stamina: int,
        wits: int,
        # abilities
        awareness: int,
        dodge: int,
        melee: int,
        # misc
        health_levels: list[int] | None = None
    ) -> None:
        self.name = name

        self.strength = strength
        self.dexterity = dexterity
        self.stamina = stamina
        self.wits = wits

        self.awareness = awareness
        self.dodge = dodge
        self.melee = melee

        self.health_levels = health_levels or [0, 0, -1, -1, -2, -2, -4]
