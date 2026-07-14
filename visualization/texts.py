# -----------------------
# Red neuronal para aproximar f(a,b,c)=ab+c
# fecha: jul 05 2026
# Autor: Daniel Emiliano Lopez Aguilar
# archivo: visualizacion.py
# Objetivo: Por medio de entrenamiento la IA deberá hacercarse lo más posible al valor correcto de x.
# Comentarios y ajustes:
#       El resultado varia dependiendo del tamaño de datos de entrenamiento, su variabilidad y rango.
#       usa 3 capas.
# -----------------------

from rich.console import Console

from rich.panel import Panel

from rich.text import Text
from rich import box
from config import neural_network_config

console = Console()


def richMessage(text, color, is_component=False):
    panel = Panel(
        f"[bold {color}]{text}[/bold {color}]",
        box=box.ROUNDED,
        border_style="cyan",
        expand=False,  # Keeps the panel tight around the text instead of full-width
    )

    message = Text(text, style=color)
    return message if is_component else panel

def init_values_message(data: neural_network_config):
    
    return richMessage(
    f"""
CONFIGURACION DE ENTRENAMIENTO
{data.ecuacion}
---------------------------
Cantidad Datos de Entrenamiento: {data.data_size}
Epocas: {data.epochs}
Valor Minimo: {data.minimo}
Valor Maximo: {data.maximo}
Tasa de aprendizaje: {data.lr}
""",
    "blue",
    True,
    )
