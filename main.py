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
import itertools
from network import NeuralNetwork
from layer import Layer
from train import Trainer
from activations import ActivationFunctions
from dataset import generate_linear_dataset
from visualizacion import show_predictions, show_loss_gradient


# -----------------------
# DATOS INICIALES
# -----------------------
#input
# ATENCION ATENCION ANTENCION: 
# Numeros con mejores resultados por ahora
a, b, c = 15.0, 25.0, 30.0 # x = 22*24+50
data_size = 8000 #recomendado 7000 a 10000
epochs = 30000 #recomendado 30000 a 35000
minimo = 5.0 # valor minimo
maximo = 50.0 # valor maximo
lr = 0.02 

def compute_x_range(low, high):
    """Calcula el rango real de x = a*b + c evaluando las esquinas."""
    corners = [low, high]
    ab_values = [a * b for a, b in itertools.product(corners, repeat=2)]
    ab_min, ab_max = min(ab_values), max(ab_values)
    x_min = ab_min + low
    x_max = ab_max + high
    return x_min, x_max

x_min, x_max = compute_x_range(minimo, maximo)
# tamano de datos
dataset = generate_linear_dataset(size=data_size)

X = np.array([inputs for inputs, target in dataset], dtype=np.float32)
Y = np.array([[target] for inputs, target in dataset], dtype=np.float32)


# -----------------------
# Normalizado a valores x,x y -x,x . Faltan pruebas para -x,-x
# -----------------------
X_norm = np.zeros_like(X)
X_norm[:, 0] = (X[:, 0] - minimo) / (maximo - minimo)  # a
X_norm[:, 1] = (X[:, 1] - minimo) / (maximo - minimo)  # b
X_norm[:, 2] = (X[:, 2] - minimo) / (maximo - minimo)  # c

X = X_norm
Y = (Y - x_min) / (x_max - x_min)


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
    Layer(32, 32, ActivationFunctions.relu, ActivationFunctions.relu_derivative)
)

network.add(
    Layer(32, 1, ActivationFunctions.linear, ActivationFunctions.linear_derivative)
)

# -----------------------
# ENTRENAMIENTO
# COMENTARIO: TAMBIEN IMPRIME LA PERDIDA EN train.py EN TIEMPO REAL POR EPOCH
# -----------------------
print("Training started...")

history, data_epoch = Trainer.train(
    network,
    X,
    Y,
    epochs=epochs,
    lr=lr
)

print("Training finished.")

# Grafica funcion de perdida
# show_loss_gradient(data_epoch)

# -----------------------
# IMPRIMIR RESULTADOS
# -----------------------
#Grafica comparacion de prediccion y valor real.
pred = network.forward(X)[-1]  # shape (N, 1), all samples
show_predictions(pred,Y)

# -----------------------
# REVISION DE CAPAS
# -----------------------
print("\nLayer shapes:")
for i, layer in enumerate(network.layers):
    print(i, layer.weights.shape)

# -----------------------
# REVISION DE PESOS
# -----------------------
def predict(network, a, b, c):
    a_norm = (a - minimo) / (maximo - minimo)
    b_norm = (b - minimo) / (maximo - minimo)
    c_norm = (c - minimo) / (maximo - minimo)
    x_input = np.array([[a_norm, b_norm, c_norm]], dtype=np.float32)
    pred = network.forward(x_input)[-1]
    return pred[0][0] * (x_max - x_min) + x_min


result = predict(network, a, b, c)

# -----------------------
# IMPRIMEME GABO Y BETO :D
# -----------------------
print(f"\nFor a={a}, b={b}, c={c} → predicted x = {result:.4f} | true x = {a*b+c:.4f}")