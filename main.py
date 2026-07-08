# -----------------------
# Red neuronal para aproximar f(a,b,c)=ab+c
# fecha: jun 28 2026
# Autor: Daniel Emiliano Lopez Aguilar
# archivo: main.py
# Objetivo: Por medio de entrenamiento la IA deberá hacercarse lo más posible al valor correcto de x.
# Comentarios y ajustes:
#       El resultado varia dependiendo del tamaño de datos de entrenamiento, su variabilidad y rango.
#       usa 3 capas. 
# -----------------------
from topologies import Topologies
import numpy as np
import itertools
from network import NeuralNetwork
from layer import Layer
from train import Trainer
from activations import ActivationFunctions
from dataset import generate_linear_dataset, test_cases
import visualizacion as vs


# -----------------------
# DATOS INICIALES
# -----------------------
#input
# ATENCION ATENCION ANTENCION: 
# Numeros con mejores resultados por ahora
ecuacion = "x=ab+c"
data_size = 10000 #
epochs = 35000 #
minimo = -50.0 # valor minimoj
maximo = 50.0 # valor maqximo
lr = 0.025
tests = test_cases(1000, minimo, maximo)
test_constant = tests[0] # (25.0, 20.0, 33.0, 533.0)
red_neuronal = Topologies.wide() # Opciones: .wide .medium .small .bottle_neck
print(f"data_size = {data_size}\nepochs = {epochs}\nminimo = {minimo}\nmaximo = {maximo}\n lr = {lr}")

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


# -----------------------
# CAPAS RED NEURONAL
# -----------------------
network = red_neuronal

# -----------------------
# ENTRENAMIENTO
# COMENTARIO: TAMBIEN IMPRIME LA PERDIDA EN train.py EN TIEMPO REAL POR EPOCH
# -----------------------
print("Entrenamiento iniciado...")

history, data_epoch = Trainer.train(
    network,
    X,
    Y,
    epochs = epochs,
    lr=lr
)
print("Entrenamiento terminado...")

vs.horizontalRule()
# -----------------------
# TABLA DE CAPAS
# -----------------------
vs.layerStructure(network.layers)
# -----------------------
# TABLA EPOCAS POR FUNCION
# -----------------------
data_epochs_table =[[str(item[0]), f"{item[1]:.5f}"] for item in data_epoch]

# vs.table("Funcion de perdida por epocas", ("Epocas (Epochs)","Func. Perdida (Loss)",), data_epochs_table)



# -----------------------
# IMPRIMIR RESULTADOS
# -----------------------
def predict(network, a, b, c):
    a_norm = (a - minimo) / (maximo - minimo)
    b_norm = (b - minimo) / (maximo - minimo)
    c_norm = (c - minimo) / (maximo - minimo)
    x_input = np.array([[a_norm, b_norm, c_norm]], dtype=np.float32)
    pred = network.forward(x_input)[-1]
    return pred[0][0] * (x_max - x_min) + x_min



rows = []
MAE = []
for a, b, c, x in tests:
    predicted = round(predict(network, a, b, c))

    error = abs(predicted - x)
    
    data =[str(x) for x in [a,b,c,x,predicted,error]]
    rows.append(data)
    MAE.append(error)

vs.table(f"Resultados de {ecuacion}", ("a", "b", "c", "Verdadero", "IA", "Diferencia"),rows[:20])

mae_value = sum(MAE)/len(MAE)
print(f"MAE: {mae_value}")

result = predict(network, a, b, c)

# Grafica funcion de perdida
vs.show_loss_gradient(data_epoch, epochs_num = epochs, title = "Funcion de perdida", data_size=data_size, min = minimo, max = maximo)