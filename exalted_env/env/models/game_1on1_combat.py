from exalted_env.env.models.combatant import Combatant


class Game1On1Combat:
    """
    Represents the model of an Exalted combat. Mostly just a list of combatants, and the current round.
    """

    def __init__(self, combatant1: Combatant, combatant2: Combatant) -> None:
        self.round: int = 1

        self.combatants: list[Combatant] = [combatant1, combatant2]
        self.combatant1: Combatant = combatant1
        self.combatant2: Combatant = combatant2

    # We'll uncomment this when we have bigger teams.
    # def is_game_active(self) -> bool:
    #     return (
    #         self.combatant1.state is CombatState.ACTIVE
    #         and self.combatant2.state is CombatState.ACTIVE
    #     )
