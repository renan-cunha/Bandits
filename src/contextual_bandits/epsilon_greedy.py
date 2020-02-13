import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from src.contextual_bandits.contextual_environment import ContextualEnvironment
from sklearn.exceptions import NotFittedError


class EpsilonGreedy:

    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        self.num_arms = 2
        self.classifiers = [LogisticRegression(solver="lbfgs", n_jobs=-1) for x in range(self.num_arms)]
        self.context_data = np.empty(0)
        self.rewards_data = np.empty(0)

    def simulate(self, c_env: ContextualEnvironment,
                 num_steps: int, training_freq: int,
                 progress_bar: bool = False) -> np.ndarray:
        """Returns rewards per step"""
        self.num_arms = c_env.get_num_arms()
        context_dim = c_env.get_context_dim()
        self.classifiers = [LogisticRegression() for x in range(self.num_arms)]
        self.context_data = np.zeros((num_steps, context_dim))
        self.rewards_data = np.full((num_steps, self.num_arms), -1, 
                                    dtype=float)

        rewards_history = np.zeros(num_steps)
        if progress_bar:
            pbar = tqdm(total=num_steps)
        for step in range(num_steps):
            context = c_env.get_context()
            action = self.action_policy(context)
            reward = c_env.act(action)
            rewards_history[step] = reward
            self.save_step(context, action, reward, step)
            if step % training_freq == 0:
                self.fit_classifier(step)
            if progress_bar:
                pbar.update(1)
        
        if progress_bar:
            pbar.close()
        return rewards_history

    def fit_classifier(self, step: int) -> None:
        step += 1
        contexts_so_far = self.context_data[:step]
        rewards_so_far = self.rewards_data[:step]
        for classifier_index, _ in enumerate(self.classifiers):
            action_rewards = rewards_so_far[:, classifier_index]
            index = np.argwhere(action_rewards != -1).flatten()
            action_rewards = action_rewards[index]
            if len(np.unique(action_rewards)) == 2:
                action_contexts = contexts_so_far[index]
                self.classifiers[classifier_index].fit(action_contexts,
                                                       action_rewards)

    def action_policy(self, context: np.ndarray) -> int:
        coin = random.uniform(0, 1)
        if coin > self.epsilon:
            rewards = np.zeros(len(self.classifiers))
            for classifier_index, classifier in enumerate(self.classifiers):
                action_rewards = self.rewards_data[:, classifier_index]
                if len(np.unique(action_rewards.flatten())) == 3:
                    try:
                        action_score = classifier.predict(context.reshape(1, -1))
                    except NotFittedError as e:
                        action_score = np.random.beta(3.0/self.num_arms, 4)
                else:
                    action_score = np.random.beta(3.0/self.num_arms, 4)
                rewards[classifier_index] = action_score

            max_rewards = max(rewards)
            best_actions = np.argwhere(rewards == max_rewards).flatten()
            action = np.random.choice(best_actions)
        else:
            action = random.randint(0, self.num_arms-1)
        return action

    def save_step(self, context: np.ndarray, action: int,
                  reward: float, step: int) -> None:
        self.context_data[step] = context
        self.rewards_data[step, action] = reward
