from typing import List, Callable, Tuple
from src.arm import Arm
from src.testbed import Testbed
import random
import numpy as np
from copy import deepcopy


class Bandit:

    def __init__(self, testbed: Testbed):
        self.testbed = testbed
        self.step_size = 0.1
        self.action_counts = np.zeros(testbed.get_num_actions())
        self.sample_average = False
    
    def run(self, num_steps: int, epsilon: float, sample_average: bool,
            step_size: float = 0.1,
            random_walk_stddev: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a tuple containing:
            1) an array with rewards per step
            2) a boolean array inidicating if the action is optimal per step"""
        self.action_counts = np.zeros(self.testbed.get_num_actions())
        self.sample_average = sample_average
        self.step_size = step_size

        testbed = deepcopy(self.testbed)
        num_actions = testbed.get_num_actions()
        rewards = np.zeros(num_steps, dtype=float)
        is_optimal = np.zeros(num_steps, dtype=bool)

        self.action_estimates = np.zeros(num_actions, dtype=float)
        for step in range(num_steps):
            
            action = self.action_policy(epsilon)
            reward = testbed.act(action)
            rewards[step] = reward
            is_optimal[step] = (action == testbed.optimal_action())
            self.update_action_estimate(action, reward)
            if random_walk_stddev > 0:
                testbed.random_walk(0, random_walk_stddev)
        return rewards, is_optimal

    def action_policy(self, epsilon: float) -> int:
        coin = random.uniform(0, 1)
        if coin > epsilon:
            max_value = np.max(self.action_estimates)
            max_actions = np.where(max_value == self.action_estimates)[0]
            action = random.choice(max_actions)
        else:
            action = self.testbed.random_action()
        self.action_counts[action] += 1
        return action

    def update_action_estimate(self, action: int, reward: float) -> None:
        action_estimate = self.action_estimates[action]
        if self.sample_average:
            action_count = self.action_counts[action]
            action_estimate += (reward-action_estimate) / (action_count + 1)
        else:
            action_estimate += self.step_size*(reward-action_estimate)
        self.action_estimates[action] = action_estimate
