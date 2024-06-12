import random

def f(x):
    # función objetivo
    values = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
    freq = {}
    for elem in x:
        if elem in freq:
            freq[elem] += 1
        else:
            freq[elem] = 1
    return sum(values[elem] * freq[elem] for elem in freq)

def generate_neighbor(solution, elements):
    # función para generar vecinos
    neighbor = solution[:]
    i = random.randint(0, 2)
    new_elem = random.choice([elem for elem in elements if elem not in solution])
    neighbor[i] = new_elem
    return neighbor

elements = ['A', 'B', 'C', 'D', 'E']

solution = ['A', 'B', 'C']
fitness = f(solution)

for _ in range(100):  # número de iteraciones
    neighbors = []
    for _ in range(10):  # número de vecinos
        neighbor = generate_neighbor(solution, elements)
        neighbors.append(neighbor)
    neighbor_fitness = [f(neighbor) for neighbor in neighbors]
    best_neighbor = max(neighbors, key=lambda x: f(x))
    solution = best_neighbor
    fitness = f(solution)
    print("Solución:", solution, "Fitness:", fitness)

print("Mejor solución:", solution, "Valor óptimo:", fitness)

