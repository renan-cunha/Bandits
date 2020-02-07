from typing import List
from src.arm import Arm
import random


class Bandit:

    def __init__(self, arms: List[Arm]):
        self.arms = arms
    
    def act(self, action: int):
        if action < 0 or action >= self.get_num_actions():
            raise ValueError(f"Action {action} should be on the interval "
                             f"[0, {self.get_num_actions()-1}]")
        return self.arms[action].pull()

    def get_num_actions(self) -> int:
        return len(self.arms)

    def random_walk(self, mean: float, stddev: float) -> None:
        for arm in self.arms:
            arm.random_walk()

    def random_action(self) -> int:
        return random.randint(0, self.get_num_actions()-1)
