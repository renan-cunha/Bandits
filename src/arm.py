import random


class Arm:

    def __init__(self, mean: float, stddev: float):
        self.mean = mean
        self.stddev = stddev

    def pull(self) -> float:
        return random.gauss(self.mean, self.stddev)

    def random_walk(self, mean: float, stddev: float) -> None:
        self.mean += random.gauss(mean, stddev)
