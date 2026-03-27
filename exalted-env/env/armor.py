class Armor:
    def __init__(
        self, *, hardness: int = 0, mobility_penalty: int, name: str, soak: int
    ) -> None:
        self.hardness = hardness
        self.mobility = mobility_penalty
        self.name = name
        self.soak = soak


ltArmor = Armor(soak=3, mobility_penalty=0, name="Light Armor")
mdArmor = Armor(soak=5, mobility_penalty=1, name="Medium Armor")
hvArmor = Armor(soak=7, mobility_penalty=1, name="Heavy Armor")
