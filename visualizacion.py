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
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
console = Console()
def show_loss_gradient(data_epoch, epochs_num, title, data_size, min, max):
    epochs = [epoch for epoch, _ in data_epoch]
    losses = [loss for _, loss in data_epoch]

    plt.figure(figsize=(8, 4))
    plt.title(str(title))
    plt.plot(epochs, losses, marker='o')
    plt.title(f"""Epochs: {epochs_num} | Cantidad datos: {data_size} | Valor max: {max} | Valor min: {min}""")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True)
    plt.show()

def horizontalRule():
    print("\n")
    print("-"*64)
    print("\n")


def table(title, columns, rows):
    table = Table(title=title)
    for column in columns:
        table.add_column(str(column))
    
    for row in rows:

        table.add_row(*row, style='bright_green')


    console = Console()
    console.print(table)
    horizontalRule()

def layerStructure(network_layers):
    for i, layer in enumerate(network_layers):
        layer_info = [(str(i), *map(str, layer.weights.shape)) for i, layer in enumerate(network_layers)]
    table("Capas", ("Capa", "N. Entrada", "N. Salida"), layer_info)
    return layer_info


def richMessage(text, color):
    return Text(text, style=color)

def richResults(
    mae,
    correct,
    total,
    between_0_5,
    between_5_10,
    between_10_20,
    between_20_30,
    greater_30,
    ):
    accuracy = round((correct / total) * 100,5)

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="cyan")
    table.add_column(style="bold white")

    table.add_row("MAE (Error medio absoluto)", f"{mae:.6f}")
    table.add_row("🎯 Predicciones correctas exactas", f"{correct:,} / {total:,}")

    table.add_row(
        "🟢 Error 0–5",
        f"{between_0_5:,} ({between_0_5 / total * 100:.2f}%)"
    )

    table.add_row(
        "🟡 Error 5–10",
        f"{between_5_10:,} ({between_5_10 / total * 100:.2f}%)"
    )

    table.add_row(
        "🟠 Error 10–20",
        f"{between_10_20:,} ({between_10_20 / total * 100:.2f}%)"
    )

    table.add_row(
        "🔴 Error 20–30",
        f"{between_20_30:,} ({between_20_30 / total * 100:.2f}%)"
    )

    table.add_row(
        "⚫ Error >30",
        f"{greater_30:,} ({greater_30 / total * 100:.2f}%)"
    )

    table.add_row(
        "📊 Accuracy",
        f"[bold green]{accuracy:.5f}%[/bold green]"
    )

    console.print(
        Panel.fit(
            table,
            title="[bold green]Resultados de entrenamiento[/bold green]",
            border_style="green",
        )
    )

def printer(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return Syntax(content, lexer="text", background_color="default")

rocket = printer('assets/rocket.txt')
title = printer('assets/title.txt')