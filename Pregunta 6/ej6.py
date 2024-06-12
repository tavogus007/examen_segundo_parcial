import random

class GeneticAlgorithm:
    def __init__(self, graph, start, population_size=100, generations=1000, mutation_rate=0.01):
        self.graph = graph
        self.start = start
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        nodes = list(self.graph.keys())
        nodes.remove(self.start)
        for _ in range(self.population_size):
            individual = [self.start] + random.sample(nodes, len(nodes))
            population.append(individual)
        return population

    def fitness(self, individual):
        total_distance = 0
        for i in range(len(individual) - 1):
            if individual[i + 1] in self.graph[individual[i]]:
                total_distance += self.graph[individual[i]][individual[i + 1]]
            else:
                # If no direct path, add a large penalty to discourage this path
                total_distance += float('inf')
        return total_distance

    def selection(self):
        population_fitness = [(individual, self.fitness(individual)) for individual in self.population]
        population_fitness.sort(key=lambda x: x[1])
        return [individual for individual, _ in population_fitness[:self.population_size//2]]

    def crossover(self, parent1, parent2):
        child = [None] * len(parent1)
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child[start:end] = parent1[start:end]
        pointer = 0
        for node in parent2:
            if node not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = node
        return child

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]

    def run(self):
        for generation in range(self.generations):
            new_population = self.selection()
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(new_population, 2)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            self.population = new_population
        best_individual = min(self.population, key=self.fitness)
        return best_individual, self.fitness(best_individual)


# Define the graph
graph = {
    'A': {'B': 7, 'C': 9, 'D': 10},
    'B': {'A': 7, 'D': 8, 'E': 4},
    'C': {'A': 9, 'D': 20, 'E': 5},
    'D': {'A': 10, 'B': 8, 'C': 20, 'E': 11},
    'E': {'B': 4, 'C': 5, 'D': 11}
}

# Run the genetic algorithm
ga = GeneticAlgorithm(graph, start='A')
best_path, best_distance = ga.run()

print(f"Mejor camino: {best_path}")
print(f"Mejor distancia: {best_distance}")
