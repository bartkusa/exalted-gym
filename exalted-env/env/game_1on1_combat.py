try:
    from .combatant import Combatant, CombatState
except ImportError:  # pragma: no cover - fallback for direct execution
    from combatant import Combatant, CombatState


class Game1On1Combat:
    def __init__(self, combatant1: Combatant, combatant2: Combatant) -> None:
        self.combatant1 = combatant1
        self.combatant2 = combatant2

        self.round: int = 1

    def isGameActive(self) -> bool:
        return (
            self.combatant1.state is CombatState.ACTIVE
            and self.combatant2.state is CombatState.ACTIVE
        )

    def getNextActor(self) -> Combatant | None:
        next_actor = None

        actors = [self.combatant1, self.combatant2]
        for a in actors:
            if a.took_turn:
                continue
            if next_actor is None or a.initiative > next_actor.initiative:
                next_actor = a

        if next_actor is not None:
            return next_actor

        # Both actors already took their turns; start next round.
        self.round += 1
        for a in actors:
            a.took_turn = False

        if self.combatant1.initiative == self.combatant2.initiative:
            return self.combatant1

        return self.combatant1 if self.combatant1.initiative > self.combatant2.initiative else self.combatant2
