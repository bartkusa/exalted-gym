import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from exalted_env.env.types import PZAgentId, agent_red_1, agent_blue_1
from exalted_env.exalted_env_v0 import ExaltedEnv


@dataclass
class DQNConfig:
    episodes: int = 2000
    max_rounds: int = 40
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    replay_size: int = 50_000
    learning_starts: int = 1_000
    train_freq: int = 4
    target_update_freq: int = 500
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 1500
    seed: int = 0
    log_every: int = 25
    save_path: str = "dqn_exalted.pt"
    load_path: str | None = None
    spy_episodes: int = 0
    spy_every: int = 0
    spy_log_path: str | None = None


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        obs_t = torch.tensor(np.array(obs), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(
            1
        )
        next_obs_t = torch.tensor(
            np.array(next_obs), dtype=torch.float32, device=device
        )
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        return obs_t, actions_t, rewards_t, next_obs_t, dones_t


def run_smoke_test(episodes: int = 5, render: bool = False) -> None:
    env = ExaltedEnv(max_rounds=40)
    for ep in range(episodes):
        env.reset(seed=ep)
        episode_returns = {agent: 0.0 for agent in env.possible_agents}
        steps = 0
        if render:
            env.render()

        while env.agents:
            agent = env.agent_selection
            action = 0 if agent is None else env.action_space(agent).sample()
            env.step(action)
            if render:
                env.render()
            steps += 1
            for name in episode_returns:
                episode_returns[name] += env.rewards.get(name, 0.0)

        print(
            f"episode={ep} steps={steps} "
            f"return_🔴={episode_returns[agent_red_1]:.3f} "
            f"return_🟦={episode_returns[agent_blue_1]:.3f}"
        )
    env.close()


def _epsilon_for_episode(cfg: DQNConfig, episode_idx: int) -> float:
    if episode_idx >= cfg.eps_decay_episodes:
        return cfg.eps_end
    progress = episode_idx / max(1, cfg.eps_decay_episodes)
    return cfg.eps_start + progress * (cfg.eps_end - cfg.eps_start)


def _select_action(
    policy_net: QNetwork,
    obs: np.ndarray,
    epsilon: float,
    action_dim: int,
    device: torch.device,
) -> tuple[int, bool]:
    if random.random() < epsilon:
        return random.randrange(action_dim), True
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(obs_t)
        return int(torch.argmax(q_values, dim=1).item()), False


def _load_checkpoint_if_exists(
    policy_net: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    load_path: str | None,
    device: torch.device,
) -> int:
    if load_path is None:
        return 0
    path = Path(load_path)
    if not path.exists():
        print(f"Checkpoint not found at {path.resolve()}, starting from scratch.")
        return 0
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
        policy_net.load_state_dict(ckpt["policy_state_dict"])
        target_net.load_state_dict(
            ckpt.get("target_state_dict", ckpt["policy_state_dict"])
        )
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_episode = int(ckpt.get("episode", 0))
    else:
        # Backward-compatible: old format was just the policy state_dict.
        policy_net.load_state_dict(ckpt)
        target_net.load_state_dict(ckpt)
        start_episode = 0
    print(
        f"Loaded checkpoint from {path.resolve()} (starting at episode {start_episode})."
    )
    return start_episode


def run_dqn_training(cfg: DQNConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = ExaltedEnv(max_rounds=cfg.max_rounds)
    env.reset(seed=cfg.seed)
    probe_agent = env.possible_agents[0]
    obs_dim = int(np.asarray(env.observe(probe_agent), dtype=np.float32).shape[0])
    action_dim = int(env.action_space(probe_agent).n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = QNetwork(obs_dim, action_dim).to(device)
    target_net = QNetwork(obs_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.replay_size)

    total_steps = 0
    losses: deque[float] = deque(maxlen=250)
    recent_avg_returns: deque[float] = deque(maxlen=100)
    best_avg_return = float("-inf")
    start_episode = _load_checkpoint_if_exists(
        policy_net=policy_net,
        target_net=target_net,
        optimizer=optimizer,
        load_path=cfg.load_path,
        device=device,
    )
    spy_log_file = None
    if cfg.spy_log_path:
        spy_path = Path(cfg.spy_log_path)
        spy_path.parent.mkdir(parents=True, exist_ok=True)
        spy_log_file = spy_path.open("a", encoding="utf-8", buffering=1)
        spy_log_file.write(
            f"\n===== spy session start seed={cfg.seed} episodes={cfg.episodes} =====\n"
        )
        spy_log_file.flush()

    def _emit_spy(line: str) -> None:
        print(line)
        if spy_log_file is not None:
            spy_log_file.write(line + "\n")
            spy_log_file.flush()

    try:
        for ep in range(start_episode, start_episode + cfg.episodes):
            env.reset(seed=cfg.seed + ep)
            episode_returns = {agent: 0.0 for agent in env.possible_agents}
            epsilon = _epsilon_for_episode(cfg, ep - start_episode)
            episode_steps = 0
            random_actions = 0
            should_spy = (ep - start_episode) < cfg.spy_episodes or (
                cfg.spy_every > 0 and (ep + 1) % cfg.spy_every == 0
            )

            if should_spy:
                _emit_spy(f"\n--- Spy episode {ep + 1} (epsilon={epsilon:.3f}) ---")

            while env.agents:
                agent = env.agent_selection
                if agent is None:
                    break

                obs = np.asarray(env.observe(agent), dtype=np.float32)
                action, was_exploration = _select_action(
                    policy_net, obs, epsilon, action_dim, device
                )
                if was_exploration:
                    random_actions += 1

                action_name = env.action_name(action)
                env.step(action)

                reward = float(env.rewards.get(agent, 0.0))
                done = bool(
                    env.terminations.get(agent, False)
                    or env.truncations.get(agent, False)
                    or agent not in env.agents
                )
                next_obs = (
                    np.zeros_like(obs)
                    if done
                    else np.asarray(env.observe(agent), dtype=np.float32)
                )

                replay.add(obs, action, reward, next_obs, done)

                for name in episode_returns:
                    episode_returns[name] += env.rewards.get(name, 0.0)

                total_steps += 1
                episode_steps += 1

                if should_spy:
                    _emit_spy(
                        f"step={episode_steps} agent={agent} action={action_name} "
                        f"reward={reward:+.3f} done={done} explore={was_exploration}"
                    )
                    _emit_spy(
                        f"  Round {env.game.round} | "
                        f"🔴1(dmg={env._combatants[agent_red_1].damage}, "
                        f"init={env._combatants[agent_red_1].initiative}) / "
                        f"🟦1(dmg={env._combatants[agent_blue_1].damage}, "
                        f"init={env._combatants[agent_blue_1].initiative})"
                    )

                if (
                    len(replay) >= cfg.learning_starts
                    and total_steps % cfg.train_freq == 0
                ):
                    obs_b, act_b, rew_b, next_obs_b, done_b = replay.sample(
                        cfg.batch_size, device
                    )
                    q_pred = policy_net(obs_b).gather(1, act_b)
                    with torch.no_grad():
                        q_next = target_net(next_obs_b).max(dim=1, keepdim=True).values
                        q_target = rew_b + cfg.gamma * (1.0 - done_b) * q_next
                    loss = F.smooth_l1_loss(q_pred, q_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))

                if total_steps % cfg.target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            mean_episode_return = float(np.mean(list(episode_returns.values())))
            recent_avg_returns.append(mean_episode_return)
            rolling_avg_return = float(np.mean(recent_avg_returns))
            best_avg_return = max(best_avg_return, rolling_avg_return)

            if (ep + 1) % cfg.log_every == 0:
                avg_loss = float(np.mean(losses)) if losses else 0.0
                explore_pct = 100.0 * random_actions / max(1, episode_steps)
                print(
                    f"ep={ep + 1}/{start_episode + cfg.episodes} "
                    f"steps={episode_steps} "
                    f"eps={epsilon:.3f} "
                    f"explore={explore_pct:.1f}% "
                    f"ret_🔴={episode_returns[agent_red_1]:.3f} "
                    f"ret_🟦={episode_returns[agent_blue_1]:.3f} "
                    f"avg_return_100={rolling_avg_return:.3f} "
                    f"best_avg_return_100={best_avg_return:.3f} "
                    f"replay={len(replay)} "
                    f"avg_loss={avg_loss:.4f}"
                )
            if should_spy:
                _emit_spy(
                    f"--- End spy ep {ep + 1} | "
                    f"ret_🔴={episode_returns[agent_red_1]:.3f} "
                    f"ret_🟦={episode_returns[agent_blue_1]:.3f} "
                    f"steps={episode_steps} ---\n"
                )
    finally:
        if spy_log_file is not None:
            spy_log_file.write("===== spy session end =====\n")
            spy_log_file.flush()
            spy_log_file.close()

    save_path = Path(cfg.save_path)
    torch.save(
        {
            "episode": start_episode + cfg.episodes,
            "policy_state_dict": policy_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
        },
        save_path,
    )
    print(f"Saved policy checkpoint to {save_path.resolve()}")
    env.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "train"], default="train")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-path", type=str, default="dqn_exalted.pt")
    parser.add_argument(
        "--spy-episodes",
        type=int,
        default=0,
        help="Spy on the first N training episodes with step-by-step output.",
    )
    parser.add_argument(
        "--spy-every",
        type=int,
        default=0,
        help="Also spy every Nth episode (0 disables).",
    )
    parser.add_argument(
        "--spy-log-path",
        type=str,
        default=None,
        help="Optional file path for spy play-by-play logs.",
    )
    parser.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Optional checkpoint path to continue training from.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.mode == "smoke":
        run_smoke_test(episodes=args.episodes, render=args.render)
    else:
        run_dqn_training(
            DQNConfig(
                episodes=args.episodes,
                seed=args.seed,
                save_path=args.save_path,
                load_path=args.load_path,
                spy_episodes=args.spy_episodes,
                spy_every=args.spy_every,
                spy_log_path=args.spy_log_path,
            )
        )
