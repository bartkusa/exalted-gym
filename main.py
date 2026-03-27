import gymnasium as gym
env = gym.make("CartPole-v1")
obs, info = env.reset(seed=42)
print("Initial observation:", obs)
env.close()
