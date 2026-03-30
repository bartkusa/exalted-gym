class Armor:
    """(Ex3 page 591)"""

    def __init__(
        self, *, hardness: int = 0, mobility_penalty: int, name: str, soak: int
    ) -> None:
        self.hardness = hardness
        """Represents how low-initiative decisive attacks cannot harm the wearer. Usually 0 for non-magical armor."""

        self.mobility = mobility_penalty
        """This represents how clumsy the armor is, and is _added_ to the wearer's Evasion."""

        self.name = name

        self.soak = soak
        """This is subtracted from the raw damage of a withering attack (in addition to the bearer's Stamina)."""


noArmor = Armor(soak=0, mobility_penalty=0, name="No Armor")
ltArmor = Armor(soak=3, mobility_penalty=0, name="Light Armor")
mdArmor = Armor(soak=5, mobility_penalty=-1, name="Medium Armor")
hvArmor = Armor(soak=7, mobility_penalty=-2, name="Heavy Armor")
