import numpy as np

# Definir la función de activación de escalón
def step_function(x):
    return np.where(x>=0, 1, 0)

# Inicialización de la red neuronal
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = {
        'hidden_layer': {'weights': np.random.rand(n_inputs, n_hidden)},
        'output_layer': {'weights': np.random.rand(n_hidden, n_outputs)}
    }
    return network

# Propagación hacia adelante
def forward_propagate(network, inputs):
    hidden_layer_input = np.dot(inputs, network['hidden_layer']['weights'])
    hidden_layer_output = step_function(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, network['output_layer']['weights'])
    output_layer_output = step_function(output_layer_input)
    
    return output_layer_output

# Entrenamiento de la red
def train_network(network, inputs, targets, l_rate, n_epoch):
    for epoch in range(n_epoch):
        for (input, target) in zip(inputs, targets):
            # Propagación hacia adelante
            output = forward_propagate(network, input)
            
            # Aquí es donde se realizaría la retropropagación para actualizar los pesos,
            # pero con una función de activación de escalón, esto no se puede hacer de manera efectiva.
            
            # Supongamos, en cambio, que simplemente ajustamos los pesos aleatoriamente.
            # Esto NO es un entrenamiento correcto, solo es para ilustrar:
            if output != target:
                network['hidden_layer']['weights'] += l_rate * (target - output)
                network['output_layer']['weights'] += l_rate * (target - output)  # Corregido aquí

# Ejemplo de uso
n_inputs = 2  # Por ejemplo, 2 entradas
n_hidden = 2  # Por ejemplo, 2 neuronas en la capa oculta
n_outputs = 1  # Por ejemplo, 1 salida

# Inicializar la red
network = initialize_network(n_inputs, n_hidden, n_outputs)

# Datos de entrenamiento dummy
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 0])  # Por ejemplo, una función XOR

# Entrenamiento de la red
train_network(network, inputs, targets, l_rate=0.2, n_epoch=100)

# Probar la red
for input in inputs:
    output = forward_propagate(network, input)  
    print(f"Entrada: {input}, Salida: {output}")