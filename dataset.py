# -----------------------
# Red neuronal para aproximar f(a,b,c)=ab+c
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
        

        a = float(random.randint(low, high))
        b = float(random.randint(low, high))
        c = float(random.randint(low,high))
        x = a*b+c
        if a and b == 0:
            continue
        dataset.append(((a,b,c),x))


    return dataset
def test_cases(size, low, high):
    test_cases = []
    data = generate_linear_dataset(size, int(low), int(high))
    
    for i in data:
        
        test_cases.append(i[0]+(i[1],))
        
    return test_cases


