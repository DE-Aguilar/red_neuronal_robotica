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
        
        # a = 0
        # while abs(a) < 0.1:
        #     a = float(random.randint(low, high))
        a = float(random.randint(low, high))
        b = float(random.randint(low, high))
        c = float(random.randint(low,high))
        x = a*b+c
        if a and b == 0:
            continue
        dataset.append(((a,b,c),x))


    return dataset

