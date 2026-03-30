from exalted_env.env.models.combatant import Combatant, CombatState


def action_surrender(combatant: Combatant) -> None:
    combatant.state = CombatState.SURRENDERED
