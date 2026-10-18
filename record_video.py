import numpy as np
import gymnasium as gym
import imageio

Q = np.load("results/Q_best.npy")

env = gym.make("CliffWalking-v1", render_mode="rgb_array")
s, _ = env.reset(seed=0)
frames = [env.render()]
path = [s]

for _ in range(200):
    a = int(np.argmax(Q[s]))              # greedy action, no exploration
    s, r, terminated, truncated, _ = env.step(a)
    frames.append(env.render())
    path.append(s)
    if terminated or truncated:
        break
env.close()

frames += [frames[-1]] * 4                # hold the final frame briefly
imageio.mimsave("results/agent.mp4", frames, fps=2)
print("path:", path)
print("steps:", len(path) - 1)