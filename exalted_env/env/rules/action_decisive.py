from exalted_env.env.models.combatant import CombatState, Combatant
import exalted_env.env.rules.dice as dice


def action_decisive_attack(attacker: Combatant, defender: Combatant) -> int:
    # TODO: allow weapons, skills

    attack_pool = (
        attacker.character.dexterity + attacker.character.melee + attacker.wound_penalty
    )
    attack_roll = dice.roll_d10s(attack_pool)
    attack_margin = attack_roll.sux - defender.dv

    # *After* attack roll, apply onslaught penalty to defender
    defender.defense_modifier -= 1

    if attack_margin < 0:
        match attacker.initiative:
            case atk_init if 1 <= atk_init <= 10:
                attacker.initiative -= 2
            case atk_init if 10 < atk_init:
                attacker.initiative -= 3

        # TODO: self-crash?

        return 0
    else:
        # TODO: compare defender's hardness

        damage_roll = dice.roll_d10s(attacker.initiative, double=[])

        # Reset to base init
        attacker.initiative = 3

        defender.damage += damage_roll.sux
        if defender.damage > len(defender.character.health_levels):
            defender.state = CombatState.DEAD

        return damage_roll.sux
