import random

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import AECEnv

from exalted_env.env.combat_actions import CombatActions
from exalted_env.env.models.character import Character
from exalted_env.env.models.combatant import CombatState, Combatant
from exalted_env.env.models.game_1on1_combat import Game1On1Combat
import exalted_env.env.rules as rules
from exalted_env.env.types import (
    PZAgentId,
    PZObsType,
    PZActionType,
    agent_blue_1,
    agent_red_1,
)


class ExaltedEnv(AECEnv[PZAgentId, PZObsType, PZActionType]):
    """
    Encapsulates an environment for Exalted 3e combat.
    """

    metadata = {
        "name": "exalted_env_v0",
        "render_modes": ["human"],
    }

    ACTIONS = [
        CombatActions.SURRENDER,
        CombatActions.FULL_DEFENSE,
        CombatActions.WITHERING_ATTACK,
        CombatActions.DECISIVE_ATTACK,
    ]

    # region ---- reward constants ----

    I_WIN_REWARD = 1.000
    DRAW_REWARD = -0.200
    I_SURRENDER_REWARD = -0.800
    I_DIED_REWARD = -1.000

    WOUND_REWARD_DIVISOR = 20
    """
    Only applies at game end, as a penalty in win or draw, but NOT to surrenderer.

    Eg: if I have a -2 wound penalty, that's worth `-2/20 = -0.1` points.
    """

    PENALTY_FOR_INVALID_ACTION = -0.400
    """Because I haven't figured out how to mask actions yet."""

    TICKING_CLOCK_TURN_REWARD = -0.005
    """Create pressure to end match."""

    IM_CRASHED_TURN_REWARD = 4 * TICKING_CLOCK_TURN_REWARD
    """Signal that crash is a dangerous state."""

    CRASHED_THEM_REWARD = 0.1
    """Dunno if we should reward withering, but CRASHING is especially valuable."""

    DAMAGE_DONE_REWARD_MULT = 0.050
    """Incentivize decisive attacks, instead of high-init-as-a-security-blanket."""

    # endregion ---- reward constants ----

    def __init__(self, max_rounds: int = 25):
        super().__init__()
        self.max_rounds = max_rounds

        self.possible_agents: list[PZAgentId] = [agent_red_1, agent_blue_1]

        self.agents: list[PZAgentId] = []
        """Agents make decisions. Agents are "players", and they control `Combatants`."""

        self.action_spaces: dict[PZAgentId, Discrete] = {
            agent: Discrete(len(self.ACTIONS)) for agent in self.possible_agents
        }
        """
        Each agent has a Gymnasium [`Space`](https://gymnasium.farama.org/api/spaces/), representing the agent's choices
        on their turn.
        
        This creates a mapping between `0,1,2,...` and `ACTIONS`.
        """

        self.observation_spaces: dict[PZAgentId, Box] = {
            agent: Box(low=-200, high=200, shape=(20,), dtype=np.int32)
            for agent in self.possible_agents
        }
        """
        Each agent has a Gymnasium [`Space`](https://gymnasium.farama.org/api/spaces/), representing what the agent can
        perceive.

        Here, each agent can perceive a vector of `12` numbers (ie, a 12-dimensional box), each in the range
        `[-200, 200]`.
        """

        self.rewards: dict[PZAgentId, float] = {}
        self._cumulative_rewards: dict[PZAgentId, float] = {}

        self.terminations: dict[PZAgentId, bool] = {}
        """
        For each agent, may be set to `True` when that agent's ending-condition happens (ie, materialized
        win/loss)
        """

        self.truncations: dict[PZAgentId, bool] = {}
        """For each agent, may be set to `True` when combat terminates without victory (eg, timeout)."""

        self.infos: dict[PZAgentId, dict] = {}
        """???"""

        self.game: Game1On1Combat | None = None

        self._combatants: dict[PZAgentId, Combatant] = {}
        """Mapping of agent-name to their Combatant"""

        self.agent_selection: PZAgentId | None = None
        """Exposes the currently active agent to the driver"""

    def reset(self, seed=None, options=None) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        options = options or {}
        self.agents = self.possible_agents[:]

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        char_red = options.get("character_red_1") or Character(
            name="character_red",
            strength=3,
            dexterity=4,
            stamina=2,
            wits=3,
            awareness=3,
            dodge=3,
            melee=4,
        )
        char_blue = options.get("character_blue_1") or Character(
            name="character_blue",
            strength=3,
            dexterity=4,
            stamina=2,
            wits=3,
            awareness=3,
            dodge=3,
            melee=4,
        )

        combatant_red = Combatant(agent_red_1, char_red)
        combatant_blue = Combatant(agent_blue_1, char_blue)
        self._combatants = {
            agent_red_1: combatant_red,
            agent_blue_1: combatant_blue,
        }

        rules.join_battle([combatant_red, combatant_blue])
        self.game = Game1On1Combat(combatant_red, combatant_blue)

        self.agent_selection = self._which_agent_is_next()

    def observe(self, agent) -> PZObsType:
        me = self._combatants[agent]
        other_agent = self._other_agent(agent)
        them = self._combatants[other_agent]
        round_num = self.game.round if self.game is not None else 1

        obs = np.array(
            [
                round_num,
                # Dynamic layer - what changes, and the model needs to see it change?
                me.damage,
                them.damage,
                me.wound_penalty,
                them.wound_penalty,
                me.initiative,
                them.initiative,
                me.crash_turns_remaining,
                them.crash_turns_remaining,
                me.defense_modifier,
                them.defense_modifier,
                # Strategic layer - what are the major combined values the model needs to weigh?
                # withering pool
                (
                    me.character.dexterity
                    + me.character.melee
                    + me.weapon1.accuracy
                    + me.wound_penalty
                ),
                (
                    them.character.dexterity
                    + them.character.melee
                    + them.weapon1.accuracy
                    + them.wound_penalty
                ),
                # decisive pool
                me.character.strength + me.weapon1.damage + me.wound_penalty,
                them.character.strength + them.weapon1.damage + them.wound_penalty,
                # DV
                me.dv,
                them.dv,
                # Soak
                me.character.stamina + me.armor.soak,
                them.character.stamina + them.armor.soak,
            ],
            dtype=np.int32,
        )
        return obs

    def step(self, action: PZActionType) -> None:
        # If no agent selected, or game is over, then skip ahead
        if self.agent_selection is None:
            return

        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        # OK, starting our turn...
        cur_agent = self.agent_selection
        other_agent = self._other_agent(cur_agent)
        cur_combatant = self._combatants[cur_agent]
        defender = self._combatants[other_agent]
        had_forced_attack_restriction = (
            cur_combatant.forced_attack_target is not None
            and not cur_combatant.extra_turn_pending
        )

        self._clear_rewards()
        rules.turn_start(cur_combatant)

        try:
            chosen_action = self.ACTIONS[int(action)]
        except (ValueError, TypeError, IndexError):
            # Punish invalid actions, instead of crashing?
            chosen_action = CombatActions.FULL_DEFENSE
            self._add_reward(cur_agent, self.PENALTY_FOR_INVALID_ACTION)

        if chosen_action == CombatActions.DECISIVE_ATTACK and cur_combatant.is_crashed:
            chosen_action = CombatActions.FULL_DEFENSE
            self._add_reward(cur_agent, self.PENALTY_FOR_INVALID_ACTION)
        elif (
            chosen_action
            in (CombatActions.WITHERING_ATTACK, CombatActions.DECISIVE_ATTACK)
            and cur_combatant.forced_attack_target is not None
            and cur_combatant.forced_attack_target != defender.agent
        ):
            chosen_action = CombatActions.FULL_DEFENSE
            self._add_reward(cur_agent, self.PENALTY_FOR_INVALID_ACTION)

        # Execute chosen action
        if chosen_action == CombatActions.SURRENDER:
            rules.action_surrender(cur_combatant)
            self._finish_episode(
                winner=other_agent, loser=cur_agent, loser_surrendered=True
            )
        elif chosen_action == CombatActions.FULL_DEFENSE:
            rules.action_full_defense(cur_combatant)
        elif chosen_action == CombatActions.WITHERING_ATTACK:
            defender_was_crashed = defender.is_crashed
            init_gained = rules.action_withering_attack(
                cur_combatant, defender, current_round=self.game.round
            )
            if defender.is_crashed and not defender_was_crashed:
                self._add_reward(cur_agent, self.CRASHED_THEM_REWARD)
        elif chosen_action == CombatActions.DECISIVE_ATTACK:
            defender_health_left = (
                len(defender.character.health_levels) - defender.damage
            )
            dmg_dealt = rules.action_decisive_attack(cur_combatant, defender)
            self._add_reward(
                cur_agent,
                min(dmg_dealt, defender_health_left) * self.DAMAGE_DONE_REWARD_MULT,
            )
            if defender.state == CombatState.DEAD:
                self._finish_episode(winner=cur_agent, loser=other_agent)

        rules.turn_end(cur_combatant, current_round=self.game.round)

        if not self._is_done():
            self._add_reward(cur_agent, self.TICKING_CLOCK_TURN_REWARD)
            if cur_combatant.is_crashed:
                self._add_reward(cur_agent, self.IM_CRASHED_TURN_REWARD)

        # Initiative Shift can grant an immediate extra turn.
        if cur_combatant.extra_turn_pending and not self._is_done():
            cur_combatant.extra_turn_pending = False
            self.agent_selection = cur_agent
        else:
            # End of turn; pick next agent. Round might increment.
            self.agent_selection = self._which_agent_is_next()

        if had_forced_attack_restriction:
            cur_combatant.forced_attack_target = None

        # If round incremented over max, truncate the game and call it a draw.
        if (
            self.game is not None
            and self.game.round > self.max_rounds
            and not self._is_done()
        ):
            self._declare_a_draw()

        if self._is_done():
            self.agents = []
            self.agent_selection = None

        self._accumulate_rewards()

    def render(self) -> None:
        if self.game is None:
            print("Environment not initialized. Call reset() first.")
            return
        red = self._combatants[agent_red_1]
        blu = self._combatants[agent_blue_1]
        red_emoji = (
            "🏳️"
            if red.state == CombatState.SURRENDERED
            else "💀" if red.state == CombatState.DEAD else ""
        )
        blu_emoji = (
            "🏳️"
            if blu.state == CombatState.SURRENDERED
            else "💀" if blu.state == CombatState.DEAD else ""
        )
        draw_emoji = "⚔️" if self.game.round > self.max_rounds else ""
        print(
            f"  Round {self.game.round}{draw_emoji} | "
            f"🔴{red_emoji}(dmg={red.damage}, init={red.initiative})  /  "
            f"🟦{blu_emoji}(dmg={blu.damage}, init={blu.initiative})"
        )

    def observation_space(self, agent: PZAgentId) -> Box:
        return self.observation_spaces[agent]

    def action_space(self, agent: PZAgentId) -> Discrete:
        return self.action_spaces[agent]

    def action_name(self, action_idx: int | None) -> str:
        if action_idx is None:
            return "NONE"
        try:
            action = self.ACTIONS[int(action_idx)]
        except (ValueError, TypeError, IndexError):
            return f"INVALID({action_idx})"
        return action.name

    def _other_agent(self, agent: PZAgentId) -> PZAgentId:
        return agent_blue_1 if agent == agent_red_1 else agent_red_1

    def _which_agent_is_next(self) -> PZAgentId | None:
        """
        May cause `game.round` to increment.

        Returns:
            The name of the next agent to go, using `rules.who_is_next(Game)`
        """
        if self.game is None:
            return None
        next_combatant = rules.who_is_next(self.game)
        return None if next_combatant is None else next_combatant.agent

    def _finish_episode(
        self,
        winner: PZAgentId,
        loser: PZAgentId,
        *,
        loser_surrendered: bool = False,
    ) -> None:
        self.terminations[winner] = True
        self.terminations[loser] = True

        winning_combatant = self._combatants[winner]
        # losing_combatant = self._combatants[loser]

        winner_reward = self.I_WIN_REWARD + (
            winning_combatant.wound_penalty / self.WOUND_REWARD_DIVISOR
        )
        self.rewards[winner] += float(winner_reward)

        loser_reward = (
            self.I_SURRENDER_REWARD if loser_surrendered else self.I_DIED_REWARD
        )
        self.rewards[loser] += float(loser_reward)

    def _declare_a_draw(self) -> None:
        """Set self.truncations, and apply rewards to everyone, based on relative damage."""
        self.truncations[agent_red_1] = True
        self.truncations[agent_blue_1] = True

        red = self._combatants[agent_red_1]
        blue = self._combatants[agent_blue_1]
        self.rewards[agent_red_1] += (
            self.DRAW_REWARD
            + (red.wound_penalty - blue.wound_penalty) / self.WOUND_REWARD_DIVISOR
        )
        self.rewards[agent_blue_1] += (
            self.DRAW_REWARD
            + (blue.wound_penalty - red.wound_penalty) / self.WOUND_REWARD_DIVISOR
        )

    def _is_done(self) -> bool:
        """Does any agent have a value in `self.terminations` or `self.truncations`?"""
        return any(self.terminations.values()) or any(self.truncations.values())

    def _add_reward(self, agent: PZAgentId, value: float) -> None:
        """Increment agent's `self.rewards` by `value`, and decrement the other agent's by the same amount"""
        other = self._other_agent(agent)
        self.rewards[agent] += value
        self.rewards[other] -= value
