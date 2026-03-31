from exalted_env.env.models.combatant import Combatant
import exalted_env.env.rules.dice as dice
from exalted_env.env.rules.initiative_crash import apply_crash_from_opponent


def action_withering_attack(
    attacker: Combatant, defender: Combatant, current_round: int
) -> int:
    """
    (Ex3 page 191) A move that represents one combatant trying to gain advantage over the other, and steal their
    Initiative. NEVER causes any damage to the defender.

    Returns:
        the amount of initiative that the attacker gained; 0 if attack failed
    """
    # TODO: allow weapons, skills

    weapon = attacker.weapon1
    attack_pool = (
        attacker.character.dexterity
        + attacker.character.melee
        + weapon.accuracy
        + attacker.wound_penalty
    )
    attack_roll = dice.roll_d10s(attack_pool)
    attack_margin = attack_roll.sux - defender.dv

    # *After* attack roll, apply onslaught penalty to defender
    defender.defense_modifier -= 1

    if attack_margin < 0:
        return 0  # Miss

    raw_dmg_pool = attack_margin + attacker.character.strength + weapon.damage
    post_soak_dmg_pool = max(
        weapon.overwhelming,
        raw_dmg_pool - (defender.character.stamina + defender.armor.soak),
    )
    damage_roll = dice.roll_d10s(post_soak_dmg_pool)

    old_attacker_init = attacker.initiative

    attacker.initiative += damage_roll.sux + 1
    defender.initiative -= damage_roll.sux
    apply_crash_from_opponent(attacker, defender, current_round=current_round)

    return attacker.initiative - old_attacker_init
