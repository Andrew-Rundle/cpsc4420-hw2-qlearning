import gymnasium as gym
from agent import QLearningAgent, train

env = gym.make("CliffWalking-v1")
agent = QLearningAgent(48, 4, seed=0)
rewards, lengths = train(env, agent, n_episodes=500, seed=0)
print("first 50 episodes mean:", rewards[:50].mean())
print("last 50 episodes mean: ", rewards[-50:].mean())