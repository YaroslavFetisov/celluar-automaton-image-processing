from CUDA.image_processor_cuda import ImageProcessor
import cupy as cp
import numpy as np
import random
from tqdm import tqdm

from CUDA.range_optimizer_cuda import cupy_f1_score


class CellularAutomatonEdgeDetector:
    def __init__(self, image_path, population_size=30, max_generations=50, neighborhood_size=3,
                 mutation_rate=0.1, early_stopping=10):
        self.image_processor = ImageProcessor(image_path, neighborhood_size)
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.early_stopping = early_stopping
        self.rule_length = neighborhood_size ** 2 + 1  # +1 для результату

    def initialize_population(self):
        return cp.random.randint(0, 2, (self.population_size, self.rule_length), dtype=cp.uint8)

    def evaluate_rules_parallel(self, population):
        fitness = cp.zeros(self.population_size, dtype=cp.float32)
        for i, rule in enumerate(population):
            result = self.image_processor.apply_rule(self.image_processor.binary_image, rule)
            fitness[i] = cupy_f1_score(cp.asarray(self.image_processor.canny_flat), result.flatten())
        return fitness

    def selection(self, population, fitness):
        selected = []
        for _ in range(self.population_size):
            idx1, idx2 = random.sample(range(self.population_size), 2)
            winner = idx1 if fitness[idx1] > fitness[idx2] else idx2
            selected.append(population[winner].copy())
        return selected

    def crossover(self, p1, p2):
        crossover_point = random.randint(1, self.rule_length - 1)
        c1 = cp.concatenate((p1[:crossover_point], p2[crossover_point:]))
        c2 = cp.concatenate((p2[:crossover_point], p1[crossover_point:]))
        return c1, c2

    def mutate(self, individual):
        mask = cp.random.random(self.rule_length) < self.mutation_rate
        individual = (individual + mask) % 2
        return individual

    def evolve(self):
        population = self.initialize_population()
        best_rule = None
        best_fitness = 0
        no_improvement = 0
        fitness_history = []

        for generation in tqdm(range(self.max_generations)):
            fitness = self.evaluate_rules_parallel(population)
            current_best = cp.max(fitness)
            best_idx = cp.argmax(fitness)
            current_best_rule = population[best_idx]

            if current_best > best_fitness:
                best_fitness = current_best
                best_rule = current_best_rule.copy()
                no_improvement = 0
            else:
                no_improvement += 1

            fitness_history.append(float(best_fitness))

            if no_improvement >= self.early_stopping:
                print(f"\nРання зупинка на поколінні {generation}")
                break

            selected = self.selection(population, fitness)
            new_population = []
            for i in range(0, self.population_size, 2):
                if i + 1 >= len(selected):
                    break
                c1, c2 = self.crossover(selected[i], selected[i + 1])
                new_population.extend([self.mutate(c1), self.mutate(c2)])

            population = cp.array(new_population[:self.population_size])

        return best_rule, fitness_history

    def detect_edges(self, rule=None):
        if rule is None:
            rule, _ = self.evolve()
        return self.image_processor.apply_rule(self.image_processor.binary_image, rule)