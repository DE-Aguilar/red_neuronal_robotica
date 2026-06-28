# -----------------------
# AI de 3 capas para funcion lineal con una incognita.
# fecha: jun 28 2026
# Autor: Daniel Emiliano Lopez Aguilar
# archivo: main.py
# Objetivo: Por medio de entrenamiento la IA deberá hacercarse lo más posible al valor correcto de x.
# Comentarios y ajustes:
#       El resultado varia dependiendo del tamaño de datos de entrenamiento, su variabilidad y rango.
#       usa 3 capas. 
# -----------------------

import numpy as np
from network import NeuralNetwork
from layer import Layer
from train import Trainer
from activations import ActivationFunctions
from dataset import generate_linear_dataset


# -----------------------
# DATA
# -----------------------
# tamano de datos
dataset = generate_linear_dataset(size=50000)

X = np.array([inputs for inputs, target in dataset], dtype=np.float32)
Y = np.array([[target] for inputs, target in dataset], dtype=np.float32)


# -----------------------
# OPTIONAL: podria normalizarse, pero asi por el momento
# -----------------------
X = X / np.array([2500.0, 50.0, 50.0], dtype=np.float32)
Y = Y / 50.0


# -----------------------
# CAPAS RED NEURONAL
# -----------------------
network = NeuralNetwork()

network.add(
    # layer(Neuronas de entrada, neuronas de salida, funcion de entrada, funcion de salida)
    Layer(3, 32, ActivationFunctions.relu, ActivationFunctions.relu_derivative)
)

network.add(
    Layer(32, 32, ActivationFunctions.relu, ActivationFunctions.relu_derivative)
)

network.add(
    Layer(32, 1, ActivationFunctions.linear, ActivationFunctions.linear_derivative)
)

# -----------------------
# ENTRENAMIENTO
# -----------------------
print("Training started...")

history, data_epoch = Trainer.train(
    network,
    X,
    Y,
    epochs=20000,
    lr=0.005
)

print("Training finished.")


# -----------------------
# IMPRIMIR RESULTADOS
# COMENTARIO: TAMBIEN IMPRIME LA PERDIDA EN train.py EN TIEMPO REAL POR EPOCH
# -----------------------
pred = network.forward(X[:10])[-1]

print("\nPred vs True (first 10):")
for i in range(10):
    print(f"pred={pred[i][0]:.4f} | true={Y[i][0]:.4f}")


# -----------------------
# REVISION DE PESOS
# -----------------------
print("\nLayer shapes:")
for i, layer in enumerate(network.layers):
    print(i, layer.weights.shape)

# -----------------------
# REVISION DE PESOS
# -----------------------
def predict(network, c, a, b):
    # NORMALIZADO DE LA MISMA MANERA QUE LOS DATOS EN train.py
    # Comentario: se podria crear una var global para tener consistencia y evitar hacerlo manual
    x_input = np.array([[c / 2500.0, a / 50.0, b / 50.0]], dtype=np.float32)
    pred = network.forward(x_input)[-1]
    # output sin normalizar
    return pred[0][0] * 50.0

# EJEMPLO
c, a, b = 10.0, 2.0, 4.0  # (10-4)/2 = 3.0
result = predict(network, c, a, b)

# -----------------------
# IMPRIMEME GABO Y BETO :D
# -----------------------
print(f"\nFor c={c}, a={a}, b={b} → predicted x = {result:.4f} | true x = {(c-b)/a:.4f}")