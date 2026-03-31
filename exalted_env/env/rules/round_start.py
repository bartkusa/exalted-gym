from exalted_env.env.models.game_1on1_combat import Game1On1Combat


def round_start(game: Game1On1Combat) -> None:
    """Increment `game.round`, and reset `took_turn` for every Combatant."""
    game.round += 1

    for combatant in game.combatants:
        combatant.took_turn = False
