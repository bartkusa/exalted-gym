class Weapon:
    def __init__(
        self, *, acc: int, dfn: int, dmg: int, name: str, overwhelming: int = 1
    ) -> None:
        self.accuracy = acc
        self.defense = dfn
        self.damage = dmg
        self.name = name
        self.overwhelming = overwhelming
        # TO ADD: tags, skills


fists = Weapon(acc=4, dfn=0, dmg=7, name="Fists")
ltWeapon = Weapon(acc=4, dfn=0, dmg=7, name="Light Weapon")
mdWeapon = Weapon(acc=2, dfn=1, dmg=9, name="Medium Weapon")
hvWeapon = Weapon(acc=0, dfn=-1, dmg=11, name="Heavy Weapon")
