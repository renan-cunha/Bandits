from src.contextual_bandits.epsilon_greedy import EpsilonGreedy
from src.contextual_bandits.contextual_environment import ContextualEnvironment
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


X = np.load("src/contextual_bandits/Bibtex/bibtex_x.npy")
y = np.load("src/contextual_bandits/Bibtex/bibtex_y.npy")

c_env = ContextualEnvironment(X, y)
epsilon_greedy = EpsilonGreedy(0.2)
num_runs = 1
num_steps = 10000
training_freq = 50
total_rewards = np.zeros((num_runs, num_steps))

for run in tqdm(range(num_runs)):
    rewards = epsilon_greedy.simulate(c_env, num_steps, progress_bar=True,
                                      training_freq=50)
    total_rewards[run] = rewards

rewards = np.mean(total_rewards, axis=0)
rewards = rewards.flatten()
mean_rewards = np.zeros_like(rewards, dtype=float)
for index in range(rewards.shape[0]):
    mean_rewards[index] = np.mean(rewards[:index+1])

plt.subplot(211)
plt.plot(rewards)
plt.title("Raw Rewards")

plt.subplot(212)
plt.plot(mean_rewards)
plt.xlabel("Steps")
plt.title("Cumulative Mean Rewards")
plt.grid()
plt.show()
