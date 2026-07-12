# -----------------------
# Red neuronal para aproximar f(a,b,c)=ab+c
# fecha: jul 05 2026
# Autor: Daniel Emiliano Lopez Aguilar
# archivo: lcd.py
# Objetivo: This file is for internal purposes only for our organization,
#  but feel free to use it. It does the same thing but withouth fancy texts and tables.
# -----------------------
from dataset import generate_linear_dataset, test_cases
import numpy as np
import itertools
from topologies import Topologies
from train import Trainer


def lcd():
    # -------------------
    # MODIFICAME
    # -------------------
    data_size = 100  # recomendado 7000 a 10000
    epochs = 30  # recomendado 30000 a 35000
    minimo = 5.0  # valor minimo
    maximo = 50.0  # valor maximo
    Topologies.small()  # Opciones: .wide .medium .small .bottle_neck
    # -------------------
    tests = test_cases(1, minimo, maximo)
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
    # Normalizado a valores x,x y -x,x . Faltan pruebas para -x,-y
    # -----------------------
    X_norm = np.zeros_like(X)
    X_norm[:, 0] = (X[:, 0] - minimo) / (maximo - minimo)  # a
    X_norm[:, 1] = (X[:, 1] - minimo) / (maximo - minimo)  # b
    X_norm[:, 2] = (X[:, 2] - minimo) / (maximo - minimo)  # c

    X = X_norm
    Y = (Y - x_min) / (x_max - x_min)

    network = Topologies.small()
    history, data_epoch = Trainer.train(network, X, Y, epochs=epochs, lr=lr)

    def predict(network, a, b, c):
        a_norm = (a - minimo) / (maximo - minimo)
        b_norm = (b - minimo) / (maximo - minimo)
        c_norm = (c - minimo) / (maximo - minimo)
        x_input = np.array([[a_norm, b_norm, c_norm]], dtype=np.float32)
        pred = network.forward(x_input)[-1]
        return pred[0][0] * (x_max - x_min) + x_min

    for a, b, c, x in tests:
        continue
    result = predict(network, a, b, c)

    # -----------------------
    # CAPAS RED NEURONAL
    # -----------------------

    return f"\nFor a={a}, b={b}, c={c} → predicted x = {result:.4f} | true x = {a * b + c:.4f}"


print(lcd())
