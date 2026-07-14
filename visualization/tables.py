from rich.table import Table
from rich.console import Console
from rich.panel import Panel

console = Console()


def table(title, columns, rows):
    table = Table(title=title)
    for column in columns:
        table.add_column(str(column))

    for row in rows:
        table.add_row(*row, style="bright_green")

    console = Console()
    console.print(table)


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

        f"[bold green]Neuronas de entrada (Input neurons):[bold green]",
        
        f"[bold white]{network_layer_info[0]}[bold white]",
        
    )
    table.add_row(
        "[bold green]Capas ocultas (Hidden layers):[/bold green]",
        f"[bold white]{len(network_layer_info) - 2}[/bold white]",
    )

    table.add_row(
        "[bold green]Neuronas de salida (Output neurons):[/bold green]",
        f"[bold white]{network_layer_info[-1]}[/bold white]",
    )

    table.add_row(
        "[bold green]Total de capas (Total layers):[/bold green]",
        f"[bold white]{len(network_layer_info)}[/bold white]",
    )

    table.add_row(
        "[bold green]Parámetros (parameters):[/bold green]",
        f"[bold white]{total_parameters:,}[/bold white]",
    )

    table_info = Panel.fit(
        table,
        title="[bold white]Datos de red neuronal[/bold white]",
        border_style="white",
        width=200,
    )
    return table_info if is_component else console.print(table_info)


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
