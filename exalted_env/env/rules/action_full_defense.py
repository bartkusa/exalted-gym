from exalted_env.env.models.combatant import Combatant


def action_full_defense(combatant: Combatant) -> int:
    """
    (Ex3 page 196) Increases the combatant's `defense_modifier` by 2.

    Returns:
        the combatant's new `defense_modifier`
    """
    combatant.defense_modifier += 2
    return combatant.defense_modifier
