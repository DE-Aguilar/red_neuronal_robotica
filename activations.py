# -----------------------
# AI de 3 capas para funcion lineal con una incognita.
# fecha: jun 28 2026
# Autor: Daniel Emiliano Lopez Aguilar
# archivo: activations.py
# Objetivo: Por medio de entrenamiento la IA deberá hacercarse lo más posible al valor correcto de x.
# Comentarios y ajustes:
#       El resultado varia dependiendo del tamaño de datos de entrenamiento, su variabilidad y rango.
#       usa 3 capas. 
# -----------------------
import numpy as np


class ActivationFunctions:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(sigmoid_output):
        # assumes input is already sigmoid(x)
        return sigmoid_output * (1 - sigmoid_output)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(np.float32)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x, dtype=np.float32)