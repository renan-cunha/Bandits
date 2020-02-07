from typing import List
from src.arm import Arm


class Bandit:

    def __init__(self, arms: List[Arm]):
        self.arms = arms
    
    def act(self, action: int):
        if action < 0 or action >= self.get_num_actions():
            raise ValueError(f"Action {action} should be on the interval "
                             f"[0, {self.get_num_actions()}]")
        return self.arms[action].pull()

    def get_num_actions(self) -> int:
        return len(self.arms)


