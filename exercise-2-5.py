import matplotlib.pyplot as plt
from src.arm import Arm
from src.bandit import Bandit
from copy import deepcopy
import numpy as np
import random
from scipy.signal import savgol_filter


mean = 0
stddev = 1
num_arms = 10
num_steps = 10**4
epsilon = 0.1
step_size = 0.1

unit_arm = Arm(mean, stddev)
unit_arms = [Arm(x, 1) for x in [0.1, -0.1, 2, -2, 0.5, 0.5, 1, -1, 1.5, -1.5]]
bandit = Bandit(unit_arms)


rewards = []
actions = []
action_estimates = np.full(bandit.get_num_actions(), fill_value=0.0)
for step in range(1, num_steps+1):
    coin = random.uniform(0, 1)
    if coin > epsilon:
        action = np.argmax(action_estimates)
    else:
        action = bandit.random_action()
    actions.append(action)
    reward = bandit.act(action)
    rewards.append(reward)
    action_estimate = action_estimates[action]
    action_estimates[action] = action_estimate + (1.0/step)*(reward-action_estimate)
print(action_estimates)
print(actions.count(2))
