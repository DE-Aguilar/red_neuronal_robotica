from rich.panel import Panel
from rich.columns import Columns
from rich.align import Align
from rich.console import Console
from rich.text import Text

console = Console()


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


def show_topology_t_diagram(network_layer_info, is_component=False):
    items = []
    for i, n in enumerate(network_layer_info):
        # Neurons represented with dots vertical dots in column.
        panel = Panel(
            Align.center("\n".join("●" for _ in range(n)), vertical="middle"),
            height=max(network_layer_info),
        )
        items.append(panel)
        # Little arrow thought would look cool.
        if i < len(network_layer_info) - 1:
            items.append(Align.center(Text("→", style="bold cyan"), vertical="middle"))
    result = Columns(items)
    return items if is_component else console.print(result)
