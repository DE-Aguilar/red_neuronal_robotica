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

import numpy as np
import itertools
from train import Trainer
from dataset import generate_linear_dataset, test_cases
from rich.console import Console
from rich.columns import Columns
from rich.table import Table
import visualization.texts as txt
from visualization.ascii import rocket, title, horizontal_line
from visualization.diagrams import layer_structure, show_topology_t_diagram
from visualization.tables import show_network_layer_info
from config import neural_network_config
from topologies import Topologies


console = Console()


def compute_x_range(low, high):
    """Calcula el rango real de x = a*b + c evaluando las esquinas."""
    corners = [low, high]
    ab_values = [a * b for (a, b) in itertools.product(corners, repeat=2)]
    # ab_values = [a * b for a, b in itertools.product(corners, repeat=2)]
    ab_min, ab_max = min(ab_values), max(ab_values)
    x_min = ab_min + low
    x_max = ab_max + high
    return x_min, x_max


def normalize(data: neural_network_config):
    # tamano de datos
    x_min, x_max = compute_x_range(data.minimo, data.maximo)
    dataset = generate_linear_dataset(size=data.data_size)

    X = np.array([inputs for inputs, target in dataset], dtype=np.float32)
    Y = np.array([[target] for inputs, target in dataset], dtype=np.float32)

    # -----------------------
    # Normalizado a valores x,x y -x,x . Faltan pruebas para -x,-y
    # -----------------------
    X_norm = np.zeros_like(X)
    X_norm[:, 0] = (X[:, 0] - data.minimo) / (data.maximo - data.minimo)  # a
    X_norm[:, 1] = (X[:, 1] - data.minimo) / (data.maximo - data.minimo)  # b
    X_norm[:, 2] = (X[:, 2] - data.minimo) / (data.maximo - data.minimo)  # c

    X = X_norm
    Y = (Y - x_min) / (x_max - x_min)
    return X, Y


def predict(a, b, c, data: neural_network_config):
    # FIX: Error on list type data.min. (It shouldnt be a list)
    min, max = data.minimo, data.maximo
    network = neural_network_config.red_neuronal
    a_norm = (a - min) / (max - min)
    b_norm = (b - min) / (max - min)
    c_norm = (c - min) / (max - min)
    x_input = np.array([[a_norm, b_norm, c_norm]], dtype=np.float32)
    pred = network.forward(x_input)[-1]
    return pred[0][0] * (x_max - x_min) + x_min


def calculate_results(tests, data: neural_network_config):
    # Columnas para tabla Muestra de resultados
    results_table_rows = []
    # Mean absolute error
    MAE = []
    data_in_predicted = data
    for a, b, c, x in tests:
        predicted = round(predict(a, b, c, data_in_predicted))

        error = abs(int(predicted - x))
        data_one = f"{int(a)} x {int(b)} + {int(c)} = {int(x)}"
        data = [str(x) for x in [data_one, predicted, error]]
        results_table_rows.append(data)
        MAE.append(error)
    mae_value = sum(MAE) / len(MAE)
    # 0 ,1, 5, 10, 20, 40, 60
    # Times the neural net got the value right
    # Keep only numbers greater than 10 using a lambda function
    # filt/ered_iterator = filter(lambda x: x < mae_value*0., error)
    # filtered_numbers = list(filtered_iterator)
    # correct_ai_prediction_quantity = sum(1 for i in MAE if i == 0)
    exact = sum(1 for error in MAE if error == 0)
    # between_0_5 = sum(1 for error in MAE if 0 < error <= 5)
    # between_5_10 = sum(1 for error in MAE if 5 < error <= 10)
    # between_10_20 = sum(1 for error in MAE if 10 < error <= 20)
    # between_20_30 = sum(1 for error in MAE if 20 < error <= 30)
    greater_30 = sum(1 for error in MAE if error > 30)

    # Mean absolute error

    return exact, mae_value, greater_30


def demo_show_data(data: neural_network_config, is_demo=True):
    visuals = {
        "title_ascci": title,
        "rocket_ascci": rocket,
        "init_values": txt.init_values_message(data),
    }
    visuals["display_init_values"] = Columns(
        [rocket, visuals["init_values"]], equal=False, expand=False
    )
    # Print rocket with table aside
    if is_demo:
        console.print(visuals["title_ascci"])
        console.print(visuals["display_init_values"])
    else:
        console.print(visuals["init_values"])
    console.print(horizontal_line)


def demo_show_neural_net(network: Topologies):
    network_layers = layer_structure(network.layers)

    topo_diagram = show_topology_t_diagram(
        network_layer_info=network_layers, is_component=True
    )

    topo_data = show_network_layer_info(
        network_layer_info=network_layers, is_component=True
    )

    # Wrap each list of items into its own separate Columns renderable
    col1 = Columns(topo_diagram, title="Topologia de red")
    col2 = Columns([topo_data])

    # Create a master table to act as a side-by-side grid container
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=2)
    grid.add_column(justify="left", ratio=6)

    grid.add_row(col1, col2)

    # Grid
    console.print(grid)

def demo_show_results():
    pass


# Pan y circo
net_data = neural_network_config
demo_show_data(neural_network_config, False)
# Network instance
network = neural_network_config.red_neuronal
demo_show_neural_net(
    network=network
)  # network calls network_config.red_neuronal which is Topologies.medium(), Topologies.medium() has network = NeuralNetwork(), NeuralNetwork has a var layers=[]

# Normalization of data
x_min, x_max = compute_x_range(
    neural_network_config.minimo, neural_network_config.maximo
)
X, Y = normalize(neural_network_config)

# Training
history, data_epoch = Trainer.train(
    network, X, Y, epochs=neural_network_config.epochs, lr=neural_network_config.lr
)

# Output data
data_epochs_table = [[str(item[0]), f"{item[1]:.5f}"] for item in data_epoch]
# vs.table(title = "Funcion de perdida por epocas",columns= ("Epocas (Epochs)","Func. Perdida (Loss)",),rows= data_epochs_table)
demo_show_results()
tests = test_cases(neural_network_config.test_size, net_data.minimo, net_data.maximo)
exact, mae_value, greater_30 = calculate_results(tests, net_data)


print(exact)
# vs.table(
#     f"Muestra de resultados de {data.ecuacion}", (data.ecuacion, "IA", "Diferencia"), results_table_rows[:20]
# )
# vs.horizontal_rule()
# vs.richResults(
# mae=mae_value,
# correct=correct_ai_prediction_quantity,
# total=len(tests),
# between_0_5=between_0_5,
# between_5_10=between_5_10,
# between_10_20=between_10_20,
# between_20_30=between_20_30,
# greater_30=greater_30,
# )

# result = predict(network, a, b, c)
# return result

# # results shown at the end.
# # Grafica funcion de perdida
# vs.show_loss_gradient(
#     data_epoch,
#     epochs_num=epochs,
#     title="Funcion de perdida",
#     data_size=data_size,
#     min=minimo,
#     max=maximo,
# )
