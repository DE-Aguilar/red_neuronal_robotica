# -----------------------
# AI de 3 capas para funcion lineal con una incognita.
# fecha: jun 28 2026
# Autor: Daniel Emiliano Lopez Aguilar
# archivo: dataset.py
# Objetivo: Por medio de entrenamiento la IA deberá hacercarse lo más posible al valor correcto de x.
# Comentarios y ajustes:
#       El resultado varia dependiendo del tamaño de datos de entrenamiento, su variabilidad y rango.
#       usa 3 capas. 
# -----------------------
import random
import numpy as np
def generate_linear_dataset(size=10000, low=-50, high=50):

    dataset = []

    for _ in range(size):

        a = 0
        while abs(a) < 0.1:
            a = random.uniform(low, high)

        b = random.uniform(low, high)
        x = random.uniform(low, high)

        c = a * x + b

        dataset.append(((c, a, b), x))

    return dataset

