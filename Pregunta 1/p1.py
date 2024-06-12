import pandas as pd
import numpy as np

# Carga del dataset Iris
data = pd.read_csv("Iris.csv")

# Prepara los datos
X = data.iloc[:, :-1].values  # Características
y = data.iloc[:, -1].values  # Clases
y = np.eye(3)[np.array([0 if i == 'Iris-setosa' else 1 if i == 'Iris-versicolor' else 2 for i in y])]  # Codifica las clases como one-hot

# Inicializa los pesos y sesgos
n_inputs = X.shape[1]  # Número de características
n_hidden = 4  # Número de neuronas en la capa oculta
n_outputs = 3  # Número de clases
W1 = np.random.randn(n_inputs, n_hidden)  # Pesos de la capa de entrada a la oculta
b1 = np.zeros((1, n_hidden))  # Sesgos de la capa oculta
W2 = np.random.randn(n_hidden, n_outputs)  # Pesos de la capa oculta a la salida
b2 = np.zeros((1, n_outputs))  # Sesgos de la capa de salida

# Define las funciones de activación
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

# Define la función de pérdida
def cross_entropy_loss(y_pred, y_true):
  return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# Define el algoritmo de entrenamiento
def train(X, y, epochs, learning_rate, b1, W1, W2, b2):  # Pasa todos los argumentos necesarios
  for epoch in range(epochs):
    # Paso 1: Calcula la salida de la red
    b1_expanded = np.broadcast_to(b1, (X.shape[0], b1.shape[1]))  # Expande b1
    z1 = X @ W1 + b1_expanded  # Calcula la salida de la capa oculta
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2  # Calcula la salida de la capa de salida
    a2 = softmax(z2)

    # Paso 2: Calcula la pérdida
    loss = cross_entropy_loss(a2, y)

    # Paso 3: Calcula el gradiente
    dz2 = a2 - y
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)
    dz1 = dz2 @ W2.T * a1 * (1 - a1)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Paso 4: Actualiza los pesos y sesgos
    W1 -= learning_rate * dW1  # Actualiza W1
    b1 -= learning_rate * db1  # Actualiza b1
    W2 -= learning_rate * dW2  # Actualiza W2
    b2 -= learning_rate * db2  # Actualiza b2

    # Imprime la pérdida cada 100 épocas
    if epoch % 100 == 0:
      print(f"Epoch {epoch}: Loss = {loss}")

  return epochs

# Entrena la red
epochs = train(X, y, epochs=1000, learning_rate=0.4, b1=b1, W1=W1, W2=W2, b2=b2)  # Pasa todos los argumentos necesarios

print(f"La red neuronal requirió {epochs} épocas para entrenar.")