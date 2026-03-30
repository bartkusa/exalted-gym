from exalted_env.env.models.combatant import Combatant
import exalted_env.env.rules.dice as dice


def join_battle(combatants: list[Combatant]) -> None:
    for c in combatants:
        jb_pool = c.character.wits + c.character.awareness + c.wound_penalty
        c.initiative = 3 + dice.roll_d10s(jb_pool).sux
