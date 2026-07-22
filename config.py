from dataclasses import dataclass
from topologies import Topologies


@dataclass
class Neural_network_data:
    ecuacion = "x=ab+c"
    data_size = 10000  # Recomendado 30000
    epochs = 100000  # 100000
    minimo = 20.0  # valor minimo
    maximo = 50.0  # valor maximo
    lr = 0.025  # Recomendado
    red_neuronal = Topologies.medium()  # Opciones: .wide .medium .small .bottle_neck
    test_size = 300
        


@dataclass
class Metrics_data:
    mae: float
    results: list
    buckets: dict
    small_error: list
    big_error: list
