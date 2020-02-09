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
        self.ucb = False
        self.c = 0.0
        self.gradient = False
        self.average_reward = 0.0
    
    def run(self, num_steps: int, epsilon: float = 0, 
            sample_average: bool = False,
            step_size: float = 0.1,
            random_walk_stddev: float = 0.0,
            initial_estimate: float = 0.0,
            ucb: bool = False,
            c: float = 0.0,
            gradient: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a tuple containing:
            1) an array with rewards per step
            2) a boolean array inidicating if the action is optimal per step"""
        self.action_counts = np.zeros(self.testbed.get_num_actions())
        self.sample_average = sample_average
        self.step_size = step_size
        self.ucb = ucb
        self.c = c
        self.gradient = gradient
        self.average_reward = 0.0

        testbed = deepcopy(self.testbed)
        num_actions = testbed.get_num_actions()
        rewards = np.zeros(num_steps, dtype=float)
        is_optimal = np.zeros(num_steps, dtype=bool)

        self.action_estimates = np.full(num_actions, dtype=float, 
                                        fill_value=initial_estimate)

        for step in range(num_steps):
            
            action = self.action_policy(epsilon, step)
            reward = testbed.act(action)

            self.average_reward += (reward - self.average_reward) / (step + 1.0)

            rewards[step] = reward
            is_optimal[step] = (action == testbed.optimal_action())
            self.update_action_estimate(action, reward)
            if random_walk_stddev > 0:
                testbed.random_walk(0, random_walk_stddev)
        return rewards, is_optimal

    def action_policy(self, epsilon: float, step: int) -> int:
        
        if self.ucb:
            step += 1
            action_counts = self.action_counts + 10**-20  # prevent overflow
            ucb_exploration = np.sqrt(np.log(step)/action_counts)
            bound_estimation = self.action_estimates + self.c * ucb_exploration
            action = np.argmax(bound_estimation)
        elif self.gradient:
            action_probabilities = self.softmax()
            actions = list(range(self.testbed.get_num_actions()))
            action = np.random.choice(actions, p=action_probabilities)
        else:
            coin = random.uniform(0, 1)
            if coin > epsilon:
                max_value = np.max(self.action_estimates)
                max_actions = np.where(max_value == self.action_estimates)[0]
                action = random.choice(max_actions)
            else:
                action = self.testbed.random_action()
        self.action_counts[action] += 1
        return action

    def softmax(self) -> np.array:
        exp_action_estimates = np.exp(self.action_estimates)
        return exp_action_estimates / np.sum(exp_action_estimates)
        
    def update_action_estimate(self, action: int, reward: float) -> None:
        action_estimate = self.action_estimates[action]
        if self.sample_average:
            action_count = self.action_counts[action]
            action_estimate += (reward-action_estimate) / (action_count + 1)
        elif self.gradient:
            action_probabilities = self.softmax()
            action_probability = action_probabilities[action]
            reward_diff = reward - self.average_reward
            action_estimate += self.step_size * reward_diff * (1-action_probability)
            
            self.action_estimates -= self.step_size * reward_diff*action_probabilities

        else:
            action_estimate += self.step_size*(reward-action_estimate)
        self.action_estimates[action] = action_estimate
