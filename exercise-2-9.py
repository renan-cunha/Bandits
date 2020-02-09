import random
from src.arm import Arm
from src.testbed import Testbed
from src.bandit import Bandit
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


parameter_settings = [1.0/128, 1.0/64, 1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0/2, 
                      1.0, 2.0, 4.0]
exp = 5
num_steps = 2*10**exp
average_last_rewards = 10**exp
random_walk_stddev = 0.01
num_runs = 200


total_rewards = np.zeros((num_steps, num_runs, len(parameter_settings), 4), 
                         dtype="float16")
#total_best_actions = np.zeros_like(total_rewards)
pbar = tqdm(total=num_runs*len(parameter_settings))
for run in range(num_runs):
    arm_means = [random.gauss(0, 1) for x in range(10)]
    arms = [Arm(x, 1) for x in arm_means]

    testbed = Testbed(arms)
    bandit = Bandit(testbed)
    for parameter_index, parameter in enumerate(parameter_settings):
        pbar.update(1)
        results = [bandit.run(num_steps, epsilon=parameter, 
                              sample_average=True, 
                              random_walk_stddev=random_walk_stddev),
                   bandit.run(num_steps, gradient=True, step_size=parameter,
                              random_walk_stddev=random_walk_stddev),
                   bandit.run(num_steps, ucb=True, c=parameter, 
                              sample_average=True, 
                              random_walk_stddev=random_walk_stddev),
                   bandit.run(num_steps, initial_estimate=parameter, 
                              step_size=0.1,
                              random_walk_stddev=random_walk_stddev)]
        results = [x[0] for x in results]
        results_array = np.array(results)
        results_array = results_array.transpose(1, 0)
        total_rewards[:, run, parameter_index, :] = results_array
pbar.close()

total_rewards = total_rewards[-average_last_rewards:]
average_rewards = np.mean(total_rewards, axis=1)
average_rewards = np.mean(average_rewards, axis=0)
for label_index, label in enumerate(["Epsilon-Greedy (Epsilon)", 
                                     "Gradient Bandit (Alpha)", 
                                     "UCB (c)", 
                                     "Greedy with Optimistic Initialization (Q0)"]):
    plt.plot(average_rewards[:, label_index], label=label)
    plt.legend()
plt.xticks(ticks=list(range(len(parameter_settings))), 
           labels=[str(x) for x in parameter_settings])
plt.xlabel("Epsilon, Alpha, c, Q0")
plt.ylabel(f"Average Reward of last {average_last_rewards} steps")
plt.title(f"{num_runs} runs")
plt.show()


