import random


class Arm:

    def __init__(self, mean: float, stddev: float):
        self.mean = mean
        self.stddev = stddev

    def pull(self) -> float:
        return random.gauss(self.mean, self.stddev)
