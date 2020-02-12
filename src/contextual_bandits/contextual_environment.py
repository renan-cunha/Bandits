import numpy as np
import random


class ContextualEnvironment:

    def __init__(self, X: np.ndarray, y: np.ndarray, random: bool = False):
        self.X = X
        self.y = y
        self.random = random
        self.index = 0

    def __get_index(self) -> int:
        return self.index % self.X.shape[0]
    
    def get_context(self) -> np.ndarray:
        if self.random:
            self.index = random.randint(0, self.X.shape[0]-1)
        index = self.__get_index()
        return self.X[index]

    def act(self, action: int) -> float:
        index = self.__get_index()
        result = self.y[index, action]
        if not self.random:
            self.index += 1
        return result

    def get_num_arms(self) -> int:
        return self.y.shape[1]

    def get_context_dim(self) -> int:
        return self.X.shape[1]
