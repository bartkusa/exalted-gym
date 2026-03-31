# from pathlib import Path
# import sys


# ROOT = Path(__file__).parent
# sys.path.append(str(ROOT / "exalted-env"))

from exalted_env.exalted_env_v0 import ExaltedEnv
from exalted_env.env.types import PZAgentId


def run_smoke_test(episodes: int = 5) -> None:
    env = ExaltedEnv(max_rounds=40)
    for ep in range(episodes):
        env.reset(seed=ep)
        episode_returns = {agent: 0.0 for agent in env.possible_agents}
        steps = 0

        while env.agents:
            agent = env.agent_selection
            action = 0 if agent is None else env.action_space(agent).sample()
            env.step(action)
            steps += 1
            for name in episode_returns:
                episode_returns[name] += env.rewards.get(name, 0.0)

        print(
            f"episode={ep} steps={steps} "
            f"return_p0={episode_returns[PZAgentId('agent_red_1')]:.3f} "
            f"return_p1={episode_returns[PZAgentId('agent_blue_1')]:.3f}"
        )
    env.close()


if __name__ == "__main__":
    run_smoke_test()
