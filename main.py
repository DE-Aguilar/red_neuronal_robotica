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
from train import Trainer
from dataset import generate_linear_dataset, test_cases
import visualizacion as vs
from rich.console import Console
from rich.columns import Columns
from rich.table import Table
console = Console()
# -----------------------
# DATOS INICIALES
# -----------------------
#input
# ATENCION ATENCION ANTENCION: 
# Numeros con mejores resultados por ahora
ecuacion = "x=ab+c"
data_size = 30 #Recomendado 30000
epochs = 100 #100000
minimo = 20.0 # valor minimo
maximo = 50.0 # valor maximo
lr = 0.025 #Recomendado
red_neuronal = Topologies.medium() # Opciones: .wide .medium .small .bottle_neck
tests = test_cases(1000, minimo, maximo)
test_constant = tests[0]

init_values_message = vs.richMessage(
    f"""
        CONFIGURACION DE ENTRENAMIENTO
        x = f(a,b,c) = ab+c
        ---------------------------
        Cantidad Datos de Entrenamiento: {data_size:,}
        Epocas: {epochs:,}
        Valor Minimo: {minimo}
        Valor Maximo: {maximo}
        Tasa de aprendizaje: {lr}
""",
    "blue",
    True
)
# Print rocket with table aside
console.print(vs.title)
console.print(Columns([vs.rocket, init_values_message],  equal = False, expand= False))

vs.horizontalRule()

# print(f"data_size = {data_size}\nepochs = {epochs}\nminimo = {minimo}\nmaximo = {maximo}\nlr = {lr}")

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
vs.richMessage("Entrenamiento Iniciado ... ", "bold cyan")

history, data_epoch = Trainer.train(
    network,
    X,
    Y,
    epochs = epochs,
    lr=lr
)
vs.richMessage("Entrenamiento Terminado con Exito ", "bold green")

# -----------------------
# IMPRIMIR RESULTADOS
# FIXME Get all this prints and tables to another class or refactor for clarity
# -----------------------
def predict(network, a, b, c):
    a_norm = (a - minimo) / (maximo - minimo)
    b_norm = (b - minimo) / (maximo - minimo)
    c_norm = (c - minimo) / (maximo - minimo)
    x_input = np.array([[a_norm, b_norm, c_norm]], dtype=np.float32)
    pred = network.forward(x_input)[-1]
    return pred[0][0] * (x_max - x_min) + x_min


# Columnas para tabla
rows = []
# Mean absolute error
MAE = []
for a, b, c, x in tests:
    predicted = round(predict(network, a, b, c))

    error = abs(int(predicted - x))
    data_one = f"{int(a)} x {int(b)} + {int(c)} = {int(x)}"
    data =[str(x) for x in [data_one,predicted,error]]
    rows.append(data)
    MAE.append(error)

# times the neural net got the value right
correct_ai_prediction_quantity = sum(1 for i in MAE if i == 0)
exact = sum(1 for error in MAE if error == 0)
between_0_5 = sum(1 for error in MAE if 0 < error <= 5)
between_5_10 = sum(1 for error in MAE if 5 < error <= 10)
between_10_20 = sum(1 for error in MAE if 10 < error <= 20)
between_20_30 = sum(1 for error in MAE if 20 < error <= 30)
greater_30 = sum(1 for error in MAE if error > 30)
# Mean absolute error
mae_value = sum(MAE)/len(MAE)

vs.horizontalRule()
# -----------------------
# TABLA DE CAPAS
# -----------------------
network_layers = vs.layer_structure(network.layers)

topo_diagram = vs.show_topology_t_diagram(network_layer_info = network_layers, title = "Topologia", is_component=True)

topo_data = vs.show_network_layer_info(network_layer_info=network_layers,is_component=True)



# 2. Wrap each list of items into its own separate Columns renderable
col1 = Columns(topo_diagram, title="Topologia de red")
col2 = Columns([topo_data], title="Información")

# 3. Create a master table to act as a side-by-side grid container
grid = Table.grid(expand=True)  
grid.add_column(justify="center", ratio=1)
grid.add_column(justify="center", ratio=1)

# 4. Add both column layouts as a single row
grid.add_row(col1, col2)

# 5. Print the final result
console.print(grid)


# layout_results = Columns([topo_diagram, topo_diagram_two], expand=True, width=10000)

# console.print(layout_results)
# -----------------------
# TABLA EPOCAS POR FUNCION
# -----------------------
data_epochs_table =[[str(item[0]), f"{item[1]:.5f}"] for item in data_epoch]

vs.table(title = "Funcion de perdida por epocas",columns= ("Epocas (Epochs)","Func. Perdida (Loss)",),rows= data_epochs_table)


vs.table(f"Resultados de {ecuacion}", ("ab+c=x", "IA", "Diferencia"),rows[:20])


# TODO might need more statistics such as a residual plot, RMSE data and S squared data.
# print(f"MAE: {mae_value}")
# # 
# print(f"En efecto es ia {correct_ai_prediction_quantity} veces de {len(tests)}.")
# print (f"porcentaje de acertividad: {round(correct_ai_prediction_quantity/len(tests)*100,5)}%")
vs.richResults(
    mae=mae_value,
    correct=correct_ai_prediction_quantity,
    total=len(tests),
    between_0_5=between_0_5,
    between_5_10=between_5_10,
    between_10_20=between_10_20,
    between_20_30=between_20_30,
    greater_30=greater_30,
)

result = predict(network, a, b, c)

# Grafica funcion de perdida
vs.show_loss_gradient(data_epoch, epochs_num = epochs, title = "Funcion de perdida", data_size=data_size, min = minimo, max = maximo)