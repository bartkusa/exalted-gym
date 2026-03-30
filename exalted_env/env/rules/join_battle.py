from exalted_env.env.models.combatant import Combatant
import exalted_env.env.rules.dice as dice


def join_battle(combatants: list[Combatant]) -> None:
    """
    (Ex3 page 192)

    When combatants first enter combat (no matter what round), this sets their's initial Initiative to
    `3 + [Wits+Awareness]`.
    """
    for c in combatants:
        jb_pool = c.character.wits + c.character.awareness + c.wound_penalty
        c.initiative = 3 + dice.roll_d10s(jb_pool).sux
