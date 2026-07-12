# -----------------------
# Red neuronal para aproximar f(a,b,c)=ab+c
# fecha: jun 28 2026
# Autor: Daniel Emiliano Lopez Aguilar
# Archivo: layer.py
# Objetivo: Por medio de entrenamiento la IA deberá hacercarse lo más posible al valor correcto de x.
# -----------------------

import numpy as np


class Layer:
    def __init__(self, input_size, output_size, activation, derivative):
        self.activation = activation
        self.derivative = derivative

        # Instead of truncnorm
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(
            2.0 / input_size
        )
        self.bias = np.zeros((1, output_size))

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights) + self.bias
        self.output = self.activation(self.z)
        return self.output
