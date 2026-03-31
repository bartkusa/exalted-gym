from random import shuffle

from exalted_env.env.models.combatant import Combatant
from exalted_env.env.models.game_1on1_combat import Game1On1Combat
from exalted_env.env.rules.round_start import round_start


def who_is_next(game: Game1On1Combat) -> Combatant | None:
    """
    Determine the next combatant to act.

    Pseudocode:
    - find the combatant who has not gone this round, with the highest initiative
      - if there's a tie, randomly pick someone tied for most init
    - if everyone has gone, run round_start(game) and retry
    """

    def pick_highest_init_not_gone() -> Combatant | None:
        next_actors: list[Combatant] | None = None

        for combatant in game.combatants:
            if combatant.took_turn:
                continue
            elif next_actors is None:
                next_actors = [combatant]
            elif combatant.initiative < next_actors[0].initiative:
                continue
            elif combatant.initiative == next_actors[0].initiative:
                next_actors.append(combatant)
            else:
                next_actors = [combatant]

        if next_actors:
            if len(next_actors) > 1:
                shuffle(next_actors)
            return next_actors[0]
        else:
            return None

    next_actor = pick_highest_init_not_gone()
    if next_actor is not None:
        return next_actor

    # Everyone has gone this round. Start a new round and pick again.
    round_start(game)
    return pick_highest_init_not_gone()
