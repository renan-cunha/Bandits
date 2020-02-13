import random
from src.arm import Arm
from src.testbed import Testbed
from src.bandit import Bandit
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


exp = 5
num_steps = 2*10**exp
average_last_rewards = 10**exp
random_walk_stddev = 0.01
num_runs = 200
parameter_settings = [1.0/128, 1.0/64, 1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0/2, 
                      1.0, 2.0, 4.0]

epsilon_greedy_params = parameter_settings[:6]
gradient_params = parameter_settings[2:]
ucb_params = parameter_settings[3:]
optimistic_params = parameter_settings[-5:]

params = [epsilon_greedy_params, gradient_params, ucb_params, 
          optimistic_params]

total_rewards = [np.zeros((num_steps, num_runs, len(x)),
                          dtype="float16") for x in params]

ep_gred_rewards, grad_rewards, ucb_rewards, optimistic_rewards = total_rewards

common_args = {"num_steps": num_steps,
               "random_walk_stddev": random_walk_stddev}
ep_gred_args = {"sample_average": True}
grad_args = {"gradient": True}
ucb_args = {"ucb": True, **ep_gred_args}
optimistic_args = {"step_size": 0.1}
total_args = [ep_gred_args, grad_args, ucb_args, optimistic_args]

param_names = ["epsilon", "step_size", "c", "initial_estimate"]

#total_best_actions = np.zeros_like(total_rewards)
pbar = tqdm(total=num_runs*sum(len(x) for x in params))
for run in range(num_runs):
    arm_means = [random.gauss(0, 1) for x in range(10)]
    arms = [Arm(x, 1) for x in arm_means]

    testbed = Testbed(arms)
    bandit = Bandit(testbed)

    for rewards, args, param_name, algo_params in zip(total_rewards, total_args, 
                                                      param_names, params):
        for param_index, param in enumerate(algo_params):
            args = {**args, **common_args, param_name: param}
            rewards[:, run, param_index], _ = bandit.run(**args)
            pbar.update(1)
    
pbar.close()

labels = ["Epsilon-Greedy (Epsilon)", "Gradient Bandit (Alpha)", "UCB (c)", 
          "Greedy with Optimistic Initialization (Q0)"]

for index, data in enumerate(zip(labels, params)):
    label, algo_params = data 
    
    algorithm_rewards = total_rewards[index]
    last_rewards = algorithm_rewards[-average_last_rewards:]
    average_rewards = np.mean(last_rewards, axis=1)
    average_rewards = np.mean(average_rewards, axis=0)
    x = np.log2(np.array(algo_params))
    plt.plot(x, average_rewards, label=label)
    plt.legend()

ticks = np.linspace(-7, 2, 10)
plt.xticks(ticks=ticks, labels=[str(x) for x in parameter_settings])
plt.xlabel("Epsilon, Alpha, c, Q0")
plt.ylabel(f"Average Reward of last {average_last_rewards} steps")
plt.title(f"{num_runs} runs")
plt.show()


