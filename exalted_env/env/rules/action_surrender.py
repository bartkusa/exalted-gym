from exalted_env.env.models.combatant import Combatant, CombatState


def action_surrender(combatant: Combatant) -> None:
    """
    Not technically an action in Exalted.

    In this model, after surrender, the combatant will NEVER re-engage or take another action.
    """
    combatant.state = CombatState.SURRENDERED
