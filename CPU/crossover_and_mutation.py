import random
import numpy as np

class CrossoverAndMutation:
    def __init__(self, mutation_rate, rule_length):
        self.mutation_rate = mutation_rate
        self.rule_length = rule_length

    def crossover(self, p1, p2):
        pt = random.randint(1, self.rule_length - 1)
        c1 = np.concatenate([p1[:pt], p2[pt:]])
        c2 = np.concatenate([p2[:pt], p1[pt:]])
        return c1, c2

    def mutate(self, rule):
        mutation_mask = np.random.random(len(rule)) < self.mutation_rate
        rule[mutation_mask] = 1 - rule[mutation_mask]
        return rule
