from CPU.image_processor import ImageProcessor
from CPU.population import Population
from CPU.local_search import LocalSearch
from CPU.crossover_and_mutation import CrossoverAndMutation
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import random

class CellularAutomatonEdgeDetector:
    def __init__(self, image_path, population_size=30, max_generations=50, neighborhood_size=3,
                 mutation_rate=0.1, early_stopping=10, n_jobs=-1):
        self.image_processor = ImageProcessor(image_path, neighborhood_size)
        self.population = Population(population_size, neighborhood_size ** 2)
        self.local_search = LocalSearch()
        self.crossover_and_mutation = CrossoverAndMutation(mutation_rate, neighborhood_size ** 2)

        self.max_generations = max_generations
        self.early_stopping = early_stopping
        self.n_jobs = n_jobs

    def evaluate_rules_parallel(self, population):
        return Parallel(n_jobs=self.n_jobs)(delayed(self.image_processor.evaluate_rule)(rule) for rule in population)

    def evolve(self):
        population = self.population.initialize_population()
        best_rule = None
        best_fitness = 0
        no_improvement = 0
        fitness_history = []

        for generation in tqdm(range(self.max_generations)):
            fitness = self.evaluate_rules_parallel(population)
            current_best = max(fitness)
            best_idx = np.argmax(fitness)
            current_best_rule = population[best_idx]

            if generation % 3 == 0:
                top_rules = [population[i] for i in np.argsort(fitness)[-3:]]
                neighbors = self.local_search.perform_local_search(top_rules, self.evaluate_rules_parallel)

                if neighbors:
                    neighbor_fitness = self.evaluate_rules_parallel(neighbors)
                    best_neighbor_idx = np.argmax(neighbor_fitness)

                    if neighbor_fitness[best_neighbor_idx] > current_best:
                        worst_idx = np.argmin(fitness)
                        population[worst_idx] = neighbors[best_neighbor_idx]

            if current_best > best_fitness:
                best_fitness = current_best
                best_rule = current_best_rule.copy()
                no_improvement = 0
            else:
                no_improvement += 1

            fitness_history.append(best_fitness)

            if no_improvement >= self.early_stopping:
                print(f"\nРання зупинка на поколінні {generation}")
                break

            selected = self.selection(population, fitness)
            population = self.recombine(selected)

        return best_rule, fitness_history

    def selection(self, population, fitness):
        selected = []
        for _ in range(self.population.size):
            idx1, idx2 = random.sample(range(self.population.size), 2)
            winner = idx1 if fitness[idx1] > fitness[idx2] else idx2
            selected.append(population[winner].copy())
        return selected

    def recombine(self, selected):
        new_population = []
        for i in range(0, self.population.size, 2):
            if i + 1 >= len(selected):
                break

            p1, p2 = selected[i], selected[i + 1]
            c1, c2 = self.crossover_and_mutation.crossover(p1, p2)
            new_population.extend([self.crossover_and_mutation.mutate(c1), self.crossover_and_mutation.mutate(c2)])

        new_population.extend(self.population.initialize_population()[:2])
        return new_population[:self.population.size]

    def detect_edges(self, rule=None):
        if rule is None:
            rule, _ = self.evolve()
        return self.image_processor.apply_rule(self.image_processor.binary_image, rule)
