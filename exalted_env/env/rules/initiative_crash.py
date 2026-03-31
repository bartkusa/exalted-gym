from exalted_env.env.models.combatant import Combatant
import exalted_env.env.rules.dice as dice


def is_recovery_lockout_round(target: Combatant, current_round: int) -> bool:
    if target.crash_recovered_round is None:
        return False
    return (current_round - target.crash_recovered_round) <= 1


def recover_from_crash(combatant: Combatant, current_round: int) -> None:
    combatant.is_crashed = False
    combatant.crashed_by = None
    combatant.crash_turns_remaining = 0
    combatant.crash_recovered_round = current_round
    combatant.forced_attack_target = None


def _enter_crash(victim: Combatant, caused_by: Combatant) -> None:
    victim.is_crashed = True
    victim.crashed_by = caused_by.agent
    victim.crash_turns_remaining = 3


def _apply_initiative_shift(
    newly_crashed: Combatant, crasher: Combatant, current_round: int
) -> None:
    if not crasher.is_crashed:
        return
    if crasher.crashed_by != newly_crashed.agent:
        return

    recover_from_crash(crasher, current_round=current_round)
    if crasher.initiative < 3:
        crasher.initiative = 3

    join_battle_pool = (
        crasher.character.wits + crasher.character.awareness + crasher.wound_penalty
    )
    crasher.initiative += dice.roll_d10s(join_battle_pool).sux
    crasher.extra_turn_pending = True
    crasher.forced_attack_target = newly_crashed.agent


def apply_crash_from_opponent(
    attacker: Combatant, defender: Combatant, current_round: int
) -> bool:
    if defender.initiative >= 0 or defender.is_crashed:
        return False

    _enter_crash(defender, caused_by=attacker)
    if not is_recovery_lockout_round(defender, current_round=current_round):
        attacker.initiative += 5

    _apply_initiative_shift(
        newly_crashed=defender, crasher=attacker, current_round=current_round
    )
    return True


def apply_self_crash(combatant: Combatant) -> bool:
    if combatant.initiative >= 0 or combatant.is_crashed:
        return False

    _enter_crash(combatant, caused_by=combatant)
    combatant.initiative -= 5
    return True
