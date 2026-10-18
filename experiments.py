import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from agent import QLearningAgent, train

ENV_ID = "CliffWalking-v1"

BASELINE = dict(alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.99, n_episodes=500, max_steps=200)

def run_experiment(config, seeds=range(5)):
    all_rewards, all_lengths = [], []
    for seed in seeds:
        env = gym.make(ENV_ID)
        agent = QLearningAgent(48, 4, alpha=config["alpha"], gamma=config["gamma"], epsilon=config["epsilon"], epsilon_min=config["epsilon_min"], epsilon_decay=config["epsilon_decay"], seed=seed)
        rewards, lengths = train(env, agent, n_episodes=config["n_episodes"], max_steps=config["max_steps"], seed=seed)
        all_rewards.append(rewards)
        all_lengths.append(lengths)
    return np.array(all_rewards), np.array(all_lengths)

def moving_average(x, window=20):
    return np.convolve(x, np.ones(window) / window, mode="valid")

def plot_baseline(rewards, path="results/baseline.png"):
    mean_curve = rewards.mean(axis=0)
    plt.figure(figsize=(8,5))
    for r in rewards:
         plt.plot(moving_average(r), color="gray", alpha=0.3, linewidth=0.8)
    plt.plot(moving_average(mean_curve), color="C0", linewidth=2, label="mean over seeds (20-episode moving avg)")
    plt.axhline(-13, color="green", linestyle="--", label="optimal (-13)")
    plt.xlabel("Training episode")
    plt.ylabel("Total reward per episode")
    plt.title("Baseline Q-learning on CliffWalking")
    plt.ylim(-1600, 0)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

if __name__ == "__main__":
    rewards, lengths = run_experiment(BASELINE)
    plot_baseline(rewards)
    print("Baseline first 50 episodes mean:", rewards[:, :50].mean())
    print("Baseline last 50 episodes mean: ", rewards[:, -50:].mean())
    print("Baseline last 50 mean steps:    ", lengths[:, -50:].mean())

    plt.figure(figsize=(8, 5))
    for alpha in [0.1, 0.5, 0.9]:
        rewards, lengths = run_experiment({**BASELINE, "alpha": alpha})
        plt.plot(moving_average(rewards.mean(axis=0)), label=f"alpha = {alpha}")
        print(f"alpha={alpha}: last-100 mean = {rewards[:, -100:].mean():.2f}, "
              f"last-100 steps = {lengths[:, -100:].mean():.1f}")
    plt.axhline(-13, color="green", linestyle="--", label="optimal (-13)")
    plt.xlabel("Training episode")
    plt.ylabel("Total reward per episode")
    plt.title("Effect of learning rate on CliffWalking")
    plt.ylim(-600, 0)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/alpha_sweep.png", dpi=150)
    plt.close()