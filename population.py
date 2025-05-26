import numpy as np

class Population:
    def __init__(self, size, rule_length):
        self.size = size
        self.rule_length = rule_length

    def initialize_population(self):
        return [np.random.randint(0, 2, self.rule_length + 1) for _ in range(self.size)]