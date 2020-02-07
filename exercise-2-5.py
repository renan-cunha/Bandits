import matplotlib.pyplot as plt
from src.arm import Arm
from src.bandit import Bandit
import numpy as np
from tqdm import tqdm
from src.testbed import Testbed

mean = 0
stddev = 1
num_arms = 10
num_runs = 2*10**3
num_steps = 10**4
epsilon = 0.1
step_size = 0.1
random_walk_stddev = 0.01


def sample_average_function(step: int) -> float:
    return 1.0/(step+1)


def parameter_function(step: int) -> float:
    return step_size


rewards = np.zeros((num_runs, num_steps, 2))
is_optimal = np.zeros((num_runs, num_steps, 2), dtype=bool)

for run in tqdm(range(num_runs)):
    unit_arms = [Arm(mean, stddev) for x in range(num_arms)]
    testbed = Testbed(unit_arms)
    bandit = Bandit(testbed)
    for function_index, step_function in enumerate([sample_average_function,
                                                    parameter_function]):
        run_rewards, run_is_optimal = bandit.run(num_steps, epsilon, 
                                                 step_function, 
                                                 random_walk_stddev)
        rewards[run, :, function_index] = run_rewards
        is_optimal[run, :, function_index] = run_is_optimal

np.save("data/rewards.npy", rewards)
np.save("data/is_optimal.npy", is_optimal)

mean_sample_rewards = np.mean(rewards[:, :, 0], axis=0)
mean_step_rewards = np.mean(rewards[:, :, 1], axis=0)
plt.subplot(211)
plt.plot(mean_sample_rewards, label="Sample Average")
plt.plot(mean_step_rewards, label="Alpha=0.1")
plt.legend()
plt.grid()
plt.ylabel("Rewards")
plt.xlabel("Step")
plt.title("Average of Rewards in 2000 runs")

plt.subplot(212)
is_optimal = is_optimal.astype(float)
mean_sample_actions = np.mean(is_optimal[:, :, 0], axis=0)
mean_step_actions = np.mean(is_optimal[:, :, 1], axis=0)
plt.plot(mean_sample_actions, label="Sample Average")
plt.plot(mean_step_actions, label="Alpha=0.1")
plt.legend()
plt.grid()
plt.ylabel("Fraction of Optimal Action")
plt.xlabel("Step")
plt.show()
