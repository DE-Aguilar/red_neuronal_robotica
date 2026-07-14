from dataclasses import dataclass
from visualization.ascii import rocket, title
from topologies import Topologies
from rich.columns import Columns
import visualization.texts as txt
@dataclass
class neural_network_config:
    ecuacion = "x=ab+c"
    data_size = 30  # Recomendado 30000
    epochs = 100  # 100000
    minimo = 20.0  # valor minimo
    maximo = 50.0  # valor maximo
    lr = 0.025  # Recomendado
    red_neuronal = Topologies.medium()  # Opciones: .wide .medium .small .bottle_neck
    test_size = 10
    

