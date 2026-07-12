# -----------------------
# Red neuronal para aproximar f(a,b,c)=ab+c
# fecha: jun 28 2026
# Autor: Daniel Emiliano Lopez Aguilar
# archivo: network.py
# Objetivo: Por medio de entrenamiento la IA deberá hacercarse lo más posible al valor correcto de x.
# Comentarios y ajustes:
#       El resultado varia dependiendo del tamaño de datos de entrenamiento, su variabilidad y rango.
#       usa 3 capas.
# -----------------------
import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        outputs = [X]

        for layer in self.layers:
            outputs.append(layer.forward(outputs[-1]))

        return outputs

    def backward(self, outputs, gradient, lr):

        delta = gradient  # start from loss gradient at output layer

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            prev_output = outputs[i]  # input to this layer

            # activation derivative
            delta = delta * layer.derivative(layer.z)

            # gradients
            dW = prev_output.T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            # propagate error to previous layer
            if i > 0:
                delta = delta @ layer.weights.T

            # update
            layer.weights -= lr * dW
            layer.bias -= lr * db
