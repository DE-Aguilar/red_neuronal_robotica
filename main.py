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
from visualization.tables import show_network_layer_info, table as vstable, richResults
from config import Neural_network_data, Metrics_data
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


def normalize(data: Neural_network_data):
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


def predict(a, b, c, data: Neural_network_data):
    # FIX: Error on list type data.min. (It shouldnt be a list)
    min, max = data.minimo, data.maximo
    x_min, x_max = compute_x_range(min, max)
    network = Neural_network_data.red_neuronal
    a_norm = (a - min) / (max - min)
    b_norm = (b - min) / (max - min)
    c_norm = (c - min) / (max - min)
    x_input = np.array([[a_norm, b_norm, c_norm]], dtype=np.float32)
    pred = network.forward(x_input)[-1]
    return pred[0][0] * (x_max - x_min) + x_min


def calculate_results(tests, data: Neural_network_data, demo = True):
    neural_net_config = data
    # Mean absolute error
    errors = []
    results = []

    # calculate results uses accuracy
    for a, b, c, correct_ans in tests:
        predicted = round(predict(a, b, c, neural_net_config))
        error = abs(int(predicted - correct_ans))

        results.append([a,b,c, correct_ans, predicted, error])
        
        errors.append(error)
    mae = sum(errors) / len(errors)
    buckets, small_error, big_error  = accuracy(results)
    metrics = Metrics_data(
        mae=  mae,
        results=results,
        buckets= buckets,
        small_error=small_error,
        big_error=big_error
    )
    if demo:
       demo_show_results(data, metrics) 
    
    return metrics

def accuracy(results):
    buckets = {
        "exacts": 0,
        "<5%": 0,
        "<10%": 0,
        "<20%": 0,
        ">20%": 0,
        ">80%": 0,
    }
    big_error = []
    small_error = []
    for a,b,c, correct_ans, predicted, error in results:
        error_percent = abs(error) / abs(correct_ans) * 100
        sample = [correct_ans, predicted, error]
        predicted_data = [a,b,c,correct_ans, predicted, error]
        if error == 0:
            buckets["exacts"] += 1
        elif error_percent <= 5:
            buckets["<5%"] += 1
            small_error.append(predicted_data)
        elif error_percent <= 10:
            buckets["<10%"] += 1
        elif error_percent <= 20:
            buckets["<20%"] += 1
        elif error_percent >20:
            buckets[">20%"] += 1
        elif error_percent > 80:
            buckets[">80%"] += 1
            big_error.append(predicted_data)

    return buckets, small_error, big_error


def demo_show_data(data: Neural_network_data, is_demo=True):
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


def demo_show_results(data: Neural_network_data, metrics: Metrics_data
                      ):

    

    pass


# Pan y circo
net_data = Neural_network_data
demo_show_data(Neural_network_data, False)
# Network instance
network = Neural_network_data.red_neuronal
demo_show_neural_net(
    network=network
)  # network calls network_config.red_neuronal which is Topologies.medium(), Topologies.medium() has network = NeuralNetwork(), NeuralNetwork has a var layers=[]

X, Y = normalize(Neural_network_data)

# Training
history, data_epoch = Trainer.train(
    network, X, Y, epochs=Neural_network_data.epochs, lr=Neural_network_data.lr
)

# Output data
data_epochs_table = [[str(item[0]), f"{item[1]:.5f}"] for item in data_epoch]
# vs.table(title = "Funcion de perdida por epocas",columns= ("Epocas (Epochs)","Func. Perdida (Loss)",),rows= data_epochs_table)
# demo_show_results()
tests = test_cases(Neural_network_data.test_size, net_data.minimo, net_data.maximo)
metrics = calculate_results(tests, net_data)
alpha=metrics.mae
beta=metrics.big_error
charly=metrics.small_error
delta=metrics.buckets
print("mae")

console.print(alpha)
print("big")
console.print(beta)
print("small")

console.print(charly)
print("buckets")
console.print(delta)

