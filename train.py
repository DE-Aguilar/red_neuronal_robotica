# -----------------------
# Red neuronal para aproximar f(a,b,c)=ab+c
# fecha: jun 28 2026
# Autor: Daniel Emiliano Lopez Aguilar
# train.py
# Objetivo: Por medio de entrenamiento la IA deberá hacercarse lo más posible al valor correcto de x.
# -----------------------

from losses import Losses
import visualizacion as vs
from rich.console import Console
console = Console()
class Trainer:

    @staticmethod
    def train(network, X, Y, epochs, lr):
        history = []
        data_epoch = []

        for epoch in range(epochs):

            outputs = network.forward(X)

            loss, gradient = Losses.mse(outputs[-1], Y)

            network.backward(outputs, gradient, lr)

            history.append(loss)

            if (epoch + 1) % (epochs // 10) == 0:
                data_epoch.append((epoch + 1, loss))
                console.print(vs.richMessage(f"Epoch {epoch+1} | Loss: {loss:.6f}", "white", True))
                

        return history, data_epoch

        
        