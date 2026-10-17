import gymnasium as gym

env = gym.make("CliffWalking-v1")
obs, info = env.reset(seed=0)
print("start state:", obs)
print("observation space:", env.observation_space)
print("action space:", env.action_space)

obs, r, terminated, truncated, info = env.step(0)
print("after action 0 (up):", obs, r, terminated, truncated)