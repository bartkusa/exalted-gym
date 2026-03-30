from exalted_env.env.models.combatant import Combatant


def action_full_defense(combatant: Combatant) -> int:
    combatant.defense_modifier += 2
    return combatant.defense_modifier
