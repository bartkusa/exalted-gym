from exalted_env.env.models.combatant import Combatant
from exalted_env.env.models.game_1on1_combat import Game1On1Combat

from exalted_env.env.rules.round_start import round_start


def who_is_next(game: Game1On1Combat) -> Combatant | None:
    """
    Determine the next combatant to act.

    Pseudocode:
    - find the combatant who has not gone this round, with the highest initiative
    - if everyone has gone, run round_start(game) and retry
    """

    def pick_highest_init_not_gone() -> Combatant | None:
        next_actor: Combatant | None = None
        for combatant in game.combatants:
            if combatant.took_turn:
                continue
            if next_actor is None or combatant.initiative > next_actor.initiative:
                next_actor = combatant
        return next_actor

    next_actor = pick_highest_init_not_gone()
    if next_actor is not None:
        return next_actor

    # Everyone has gone this round. Start a new round and pick again.
    round_start(game)
    return pick_highest_init_not_gone()
