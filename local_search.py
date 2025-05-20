class LocalSearch:
    def __init__(self, num_neighbors=10):
        self.num_neighbors = num_neighbors

    def generate_neighbors(self, rule):
        neighbors = []
        for i in range(len(rule)):
            neighbor = rule.copy()
            neighbor[i] = 1 - neighbor[i]  # Інвертуємо один біт
            neighbors.append(neighbor)
            if len(neighbors) >= self.num_neighbors:
                break
        return neighbors

    def perform_local_search(self, best_rules, evaluate_rules_parallel):
        neighbors = []
        for rule in best_rules:
            neighbors.extend(self.generate_neighbors(rule))
        return neighbors
