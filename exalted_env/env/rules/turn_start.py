from exalted_env.env.models.combatant import Combatant


def turn_start(combatant: Combatant) -> None:
    combatant.took_turn = True

    # Reset some ephemeral combat states
    combatant.defense_modifier = 0
