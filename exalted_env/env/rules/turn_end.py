from exalted_env.env.models.combatant import Combatant
from exalted_env.env.rules.initiative_crash import recover_from_crash


def turn_end(combatant: Combatant, current_round: int) -> None:
    if combatant.is_crashed:
        combatant.crash_turns_remaining -= 1
        if combatant.crash_turns_remaining <= 0:
            recover_from_crash(combatant, current_round=current_round)
            combatant.initiative = 3
