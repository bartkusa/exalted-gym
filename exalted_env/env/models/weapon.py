class Weapon:
    """(Ex3 page 580)"""

    def __init__(
        self, *, acc: int, dfn: int, dmg: int, name: str, overwhelming: int = 1
    ) -> None:
        self.accuracy = acc
        """Adds to withering attacks' die pools."""

        self.defense = dfn
        """Adds directly to user's Parry. Only one weapon can parry an attack."""

        self.damage = dmg
        """After a successful withering attack, adds directly to the raw damage pool."""

        self.name = name

        self.overwhelming = overwhelming
        """The minimum damage pool for a withering attack, after subtracting soak."""

        # TO ADD: tags, skills


fists = Weapon(acc=4, dfn=0, dmg=7, name="Fists")
ltWeapon = Weapon(acc=4, dfn=0, dmg=7, name="Light Weapon")
mdWeapon = Weapon(acc=2, dfn=1, dmg=9, name="Medium Weapon")
hvWeapon = Weapon(acc=0, dfn=-1, dmg=11, name="Heavy Weapon")
