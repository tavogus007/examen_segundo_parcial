import random
import math

def f(x):
    values = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
    freq = {}
    for elem in x:
        if elem in freq:
            freq[elem] += 1
        else:
            freq[elem] = 1
    return sum(values[elem] * freq[elem] for elem in freq)

def generate_population(size, elements):
    population = []
    for _ in range(size):
        combination = random.sample(elements, 3)
        population.append(combination)
    return population

def generate_neighbor(solution, elements):
    neighbor = solution[:]
    i = random.randint(0, 2)
    new_elem = random.choice([elem for elem in elements if elem not in solution])
    neighbor[i] = new_elem
    return neighbor

def evaluate_solution(solution):
    return f(solution)

def neighborhood_search(population, elements, max_iterations):
    best_solution = max(population, key=evaluate_solution)
    for _ in range(max_iterations):
        neighbor = generate_neighbor(best_solution, elements)
        if evaluate_solution(neighbor) > evaluate_solution(best_solution):
            best_solution = neighbor
    return best_solution

elements = ['A', 'B', 'C', 'D', 'E']
population_size = 100
max_iterations = 1000

population = generate_population(population_size, elements)
best_solution = neighborhood_search(population, elements, max_iterations)

print("Mejor solución:", best_solution)
print("Valor óptimo:", evaluate_solution(best_solution))