import random

import numpy as np
from numpy.typing import NDArray
from gymnasium.spaces import Box, Discrete
from pettingzoo import AECEnv
from typing import NewType, TypeAlias

from exalted_env.env.combat_actions import CombatActions
from exalted_env.env.models.character import Character
from exalted_env.env.models.combatant import CombatState, Combatant
from exalted_env.env.models.game_1on1_combat import Game1On1Combat
import exalted_env.env.rules as rules


PZAgentId = NewType("PZAgentId", str)
PZObsType: TypeAlias = NDArray[np.int32]
PZActionType: TypeAlias = int | None

agent_red_1 = PZAgentId("agent_red_1")
agent_blue_1 = PZAgentId("agent_blue_1")


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

    def __init__(self, max_rounds: int = 50):
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
            agent: Box(low=-200, high=200, shape=(12,), dtype=np.int32)
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
            # TODO should observe weapon, too.
            # maybe should just observe DV, and not its constituent pieces?
            # should observe dodge _and_ parry.
            # should observe health. and soak.
            # observe damage... _and_ health? wound modifier?
            [
                round_num,
                me.initiative,
                them.initiative,
                me.damage,
                them.damage,
                me.defense_modifier,
                them.defense_modifier,
                me.character.dexterity,
                them.character.dexterity,
                me.character.melee,
                them.character.melee,
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

        self._clear_rewards()
        rules.turn_start(cur_combatant)

        try:
            chosen_action = self.ACTIONS[int(action)]
        except (ValueError, TypeError, IndexError):
            # Punish invalid actions, instead of crashing?
            chosen_action = CombatActions.FULL_DEFENSE
            self._add_reward(cur_agent, -0.1)

        # Execute chosen action
        if chosen_action == CombatActions.SURRENDER:
            rules.action_surrender(cur_combatant)
            self._finish_episode(winner=other_agent, loser=cur_agent)
        elif chosen_action == CombatActions.FULL_DEFENSE:
            rules.action_full_defense(cur_combatant)
            self._add_reward(cur_agent, 0.01)
        elif chosen_action == CombatActions.WITHERING_ATTACK:
            init_gained = rules.action_withering_attack(cur_combatant, defender)
            reward = -0.02 if init_gained <= 0 else (0.05 + 0.01 * init_gained)
            self._add_reward(cur_agent, reward)
        elif chosen_action == CombatActions.DECISIVE_ATTACK:
            dmg_dealt = rules.action_decisive_attack(cur_combatant, defender)
            reward = -0.02 if dmg_dealt <= 0 else (0.10 + 0.02 * dmg_dealt)
            self._add_reward(cur_agent, reward)
            if defender.state == CombatState.DEAD:
                self._finish_episode(winner=cur_agent, loser=other_agent)

        # End of turn; pick next agent. Round might increment.
        self.agent_selection = self._which_agent_is_next()

        # If round incremented over max, truncate the game and call it a draw.
        if (
            self.game is not None
            and self.game.round > self.max_rounds
            and not self._is_done()
        ):
            self.truncations[cur_agent] = True
            self.truncations[other_agent] = True

        if self._is_done():
            self.agents = []
            self.agent_selection = None

        self._accumulate_rewards()

    def render(self) -> None:
        if self.game is None:
            print("Environment not initialized. Call reset() first.")
            return
        p0 = self._combatants[agent_red_1]
        p1 = self._combatants[agent_blue_1]
        print(
            f"Round {self.game.round} | "
            f"P0(dmg={p0.damage}), init={p0.initiative} "
            f"P1(dmg={p1.damage}), init={p1.initiative}"
        )

    def observation_space(self, agent: PZAgentId) -> Box:
        return self.observation_spaces[agent]

    def action_space(self, agent: PZAgentId) -> Discrete:
        return self.action_spaces[agent]

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

    def _finish_episode(self, winner: PZAgentId, loser: PZAgentId) -> None:
        # TODO soften loss, if loser surrendered
        # TODO reduce victory, if winner wounded?
        self.terminations[winner] = True
        self.terminations[loser] = True
        self._add_reward(winner, 1.0)
        self._add_reward(loser, -1.0)

    def _is_done(self) -> bool:
        """Does any agent have a value in `self.terminations` or `self.truncations`?"""
        return any(self.terminations.values()) or any(self.truncations.values())

    def _add_reward(self, agent: PZAgentId, value: float) -> None:
        """Increment agent's `self.rewards` by `value`, and decrement the other agent's by the same amount"""
        other = self._other_agent(agent)
        self.rewards[agent] += value
        self.rewards[other] -= value
