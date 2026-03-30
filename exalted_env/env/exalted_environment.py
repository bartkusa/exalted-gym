from __future__ import annotations

import random

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector

from exalted_env.env.combat_actions import CombatActions
from exalted_env.env.models.character import Character
from exalted_env.env.models.combatant import CombatState, Combatant
from exalted_env.env.models.game_1on1_combat import Game1On1Combat
import exalted_env.env.rules as rules


class ExaltedEnv(AECEnv):
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
        self.possible_agents = ["player_0", "player_1"]
        self.agents: list[str] = []

        self.action_spaces = {
            agent: Discrete(len(self.ACTIONS)) for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: Box(low=-200, high=200, shape=(12,), dtype=np.int32)
            for agent in self.possible_agents
        }

        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict] = {}

        self.game: Game1On1Combat | None = None
        self._combatants: dict[str, Combatant] = {}
        self._agent_selector = None
        self.agent_selection: str | None = None

    def reset(self, seed=None, options=None):
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

        c0 = options.get("character_0") or Character(
            name="player_0",
            strength=3,
            dexterity=4,
            stamina=3,
            wits=3,
            awareness=3,
            dodge=3,
            melee=4,
        )
        c1 = options.get("character_1") or Character(
            name="player_1",
            strength=3,
            dexterity=4,
            stamina=3,
            wits=3,
            awareness=3,
            dodge=3,
            melee=4,
        )
        actor_0 = Combatant(c0)
        actor_1 = Combatant(c1)
        self._combatants = {"player_0": actor_0, "player_1": actor_1}

        rules.join_battle([actor_0, actor_1])
        self.game = Game1On1Combat(actor_0, actor_1)

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = (
            self._select_agent_from_game() or self._agent_selector.next()
        )

    def observe(self, agent):
        me = self._combatants[agent]
        other_agent = self._other(agent)
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

    def step(self, action):
        if self.agent_selection is None:
            return
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        other = self._other(agent)
        actor = self._combatants[agent]
        defender = self._combatants[other]

        self._clear_rewards()
        actor.took_turn = True
        actor.defense_modifier = 0

        try:
            chosen_action = self.ACTIONS[int(action)]
        except (ValueError, TypeError, IndexError):
            chosen_action = CombatActions.FULL_DEFENSE
            self._add_reward(agent, -0.05)

        if chosen_action == CombatActions.SURRENDER:
            rules.action_surrender(actor)
            self._finish_episode(winner=other, loser=agent)
        elif chosen_action == CombatActions.FULL_DEFENSE:
            rules.action_full_defense(actor)
            self._add_reward(agent, 0.01)
        elif chosen_action == CombatActions.WITHERING_ATTACK:
            init_gained = rules.action_withering_attack(actor, defender)

            reward = -0.02 if init_gained <= 0 else (0.05 + 0.01 * init_gained)
            self._add_reward(agent, reward)
        elif chosen_action == CombatActions.DECISIVE_ATTACK:
            dmg_dealt = rules.action_decisive_attack(actor, defender)

            reward = -0.02 if dmg_dealt <= 0 else (0.10 + 0.02 * dmg_dealt)
            self._add_reward(agent, reward)
            if defender.state == CombatState.DEAD:
                self._finish_episode(winner=agent, loser=other)

        if (
            self.game is not None
            and self.game.round > self.max_rounds
            and not self._is_done()
        ):
            self.truncations[agent] = True
            self.truncations[other] = True

        if self._is_done():
            self.agents = []
            self.agent_selection = None
        else:
            next_actor = self._select_agent_from_game()
            if next_actor is None:
                next_actor = self._agent_selector.next()
            self.agent_selection = next_actor

        self._accumulate_rewards()

    def render(self):
        if self.game is None:
            print("Environment not initialized. Call reset() first.")
            return
        p0 = self._combatants["player_0"]
        p1 = self._combatants["player_1"]
        print(
            f"Round {self.game.round} | "
            f"P0(init={p0.initiative}, dmg={p0.damage}) "
            f"P1(init={p1.initiative}, dmg={p1.damage})"
        )

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _other(self, agent: str) -> str:
        return "player_1" if agent == "player_0" else "player_0"

    def _select_agent_from_game(self) -> str | None:
        if self.game is None:
            return None
        actor = self.game.getNextActor()
        if actor is None:
            return None
        if actor is self._combatants["player_0"]:
            return "player_0"
        return "player_1"

    def _finish_episode(self, winner: str, loser: str):
        # TODO soften loss, if loser surrendered
        self.terminations[winner] = True
        self.terminations[loser] = True
        self._add_reward(winner, 1.0)
        self._add_reward(loser, -1.0)

    def _is_done(self) -> bool:
        return any(self.terminations.values()) or any(self.truncations.values())

    def _add_reward(self, agent: str, value: float):
        other = self._other(agent)
        self.rewards[agent] += value
        self.rewards[other] -= value
