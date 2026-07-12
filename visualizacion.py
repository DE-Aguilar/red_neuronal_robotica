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
from rich.columns import Columns
from rich.align import Align
from rich import box

console = Console()


def show_loss_gradient(data_epoch, epochs_num, title, data_size, min, max):
    epochs = [epoch for epoch, _ in data_epoch]
    losses = [loss for _, loss in data_epoch]

    plt.figure(figsize=(8, 4))
    plt.title(str(title))
    plt.plot(epochs, losses, marker="o")
    plt.title(
        f"""Epochs: {epochs_num} | Cantidad datos: {data_size} | Valor max: {max} | Valor min: {min}"""
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True)
    plt.show()


def horizontalRule():
    print("\n")
    print("-" * 64)
    print("\n")


def table(title, columns, rows):
    table = Table(title=title)
    for column in columns:
        table.add_column(str(column))

    for row in rows:
        table.add_row(*row, style="bright_green")

    console = Console()
    console.print(table)
    horizontalRule()


def layer_structure(network_layers):
    layers = []
    network_layer_info = []
    for i, layer in enumerate(network_layers):
        # layers info mapping in the form of eg. Topology.medium() :  [('0', '3', '16'), ('1', '16', '8'), ('2', '8', '1')]
        # as layer number, input neurons, output neurons.
        layers = [
            (str(i), *map(str, layer.weights.shape))
            for i, layer in enumerate(network_layers)
        ]
    for layer in layers:
        # Get layer's neuron's quantity
        network_layer_info.append(int(layer[1]))
        # Get exit layer neuron's quantity
    network_layer_info.append(int(layers[-1][-1]))
    return network_layer_info


def show_topology_t_diagram(network_layer_info, title, is_component=False):
    items = []
    for i, n in enumerate(network_layer_info):
        # Neurons represented with dots vertical dots in column.
        panel = Panel(
            Align.center("\n".join("●" for _ in range(n)), vertical="middle"),
            title=str(n),
            height=max(network_layer_info),
        )
        items.append(panel)
        # Little arrow thought would look cool.
        if i < len(network_layer_info) - 1:
            items.append(Align.center(Text("→", style="bold cyan"), vertical="middle"))
    result = Columns(items, title=title)
    return items if is_component else console.print(result)


def show_network_layer_info(network_layer_info, is_component=False):
    net = network_layer_info
    total_parameters = 0
    # Paramethers size calculted as: Parameters=(inputs×outputs)+outputs
    network_layers = len(net) - 1
    for layer in range(network_layers):
        input_layer_size = net[layer]
        output_layer_size = net[layer + 1]
        total_parameters += input_layer_size * output_layer_size + output_layer_size

    table = Table(show_header=False, box=None, pad_edge=False)

    table.add_row(
        richMessage("Neuronas de entrada (Input neurons):", "blue", True),
        richMessage(f"{network_layer_info[0]}", "white", True),
    )
    table.add_row(
        richMessage("Capas ocultas (Hidden layers):", "blue", True),
        richMessage(f"{len(network_layer_info) - 2}", "white", True),
    )

    table.add_row(
        richMessage("Neuronas de salida (Output neurons):", "blue", True),
        richMessage(f"{network_layer_info[-1]}", "white", True),
    )

    table.add_row(
        richMessage("Total de capas (Total layers):", "blue", True),
        richMessage(f"{len(network_layer_info)}", "white", True),
    )

    table.add_row(
        richMessage("Parámetros (parameters):", "blue", True),
        richMessage(f"{total_parameters:,}", "white", True),
    )

    table_info = Panel.fit(
        table,
        title="[bold green]Resultados de entrenamiento[/bold green]",
        border_style="green",
        width=200,
    )
    return table_info if is_component else console.print(table_info)


def richMessage(text, color, is_component=False):
    panel = Panel(
        f"[bold {color}]{text}[/bold {color}]",
        box=box.ROUNDED,
        border_style="cyan",
        expand=False,  # Keeps the panel tight around the text instead of full-width
    )

    message = Text(text, style=color)
    return message if is_component else console.print(panel)


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
    accuracy = round((correct / total) * 100, 5)

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="cyan")
    table.add_column(style="bold white")

    table.add_row("MAE (Error medio absoluto)", f"{mae:.6f}")
    table.add_row("🎯 Predicciones correctas exactas", f"{correct:,} / {total:,}")

    table.add_row("🟢 Error 0–5", f"{between_0_5:,} ({between_0_5 / total * 100:.2f}%)")

    table.add_row(
        "🟡 Error 5–10", f"{between_5_10:,} ({between_5_10 / total * 100:.2f}%)"
    )

    table.add_row(
        "🟠 Error 10–20", f"{between_10_20:,} ({between_10_20 / total * 100:.2f}%)"
    )

    table.add_row(
        "🔴 Error 20–30", f"{between_20_30:,} ({between_20_30 / total * 100:.2f}%)"
    )

    table.add_row("⚫ Error >30", f"{greater_30:,} ({greater_30 / total * 100:.2f}%)")

    table.add_row("📊 Accuracy", f"[bold green]{accuracy:.5f}%[/bold green]")

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


rocket = printer("assets/rocket.txt")
title = printer("assets/title.txt")
