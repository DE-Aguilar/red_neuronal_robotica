# -----------------------
# Red neuronal para aproximar f(a,b,c)=ab+c
# fecha: jun 28 2026
# Autor: Daniel Emiliano Lopez Aguilar
# archivo: losses.py
# Objetivo: Por medio de entrenamiento la IA deberá hacercarse lo más posible al valor correcto de x.
# Comentarios y ajustes:
#       El resultado varia dependiendo del tamaño de datos de entrenamiento, su variabilidad y rango.
#       usa 3 capas. 
# -----------------------
import numpy as np

class Losses:

    @staticmethod
    def mse(y_pred, y_true):
        diff = y_pred - y_true
        loss = np.mean(diff ** 2)
        gradient = (2 / y_true.shape[0]) * diff
        return loss, gradient