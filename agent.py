import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.99, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions))

    def act(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())   
        return int(self.rng.choice(best))

    def update(self, s, a, r, s2, terminated):
        if terminated:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q[s2])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train(env, agent, n_episodes=500, max_steps=200, seed=None):
    rewards, lengths = [], []
    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed if ep == 0 else None)
        total, steps = 0.0, 0
        for _ in range(max_steps):
            a = agent.act(s)
            s2, r, terminated, truncated, _ = env.step(a)
            agent.update(s, a, r, s2, terminated)
            total += r
            steps += 1
            s = s2
            if terminated or truncated:
                break
        agent.decay_epsilon()
        rewards.append(total)
        lengths.append(steps)
    return np.array(rewards), np.array(lengths)

 