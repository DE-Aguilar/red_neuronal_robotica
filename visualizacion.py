import matplotlib.pyplot as plt
import numpy as np
import rich as r
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt

def show_loss_gradient(data_epoch):
    epochs = [epoch for epoch, _ in data_epoch]
    losses = [loss for _, loss in data_epoch]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True)
    plt.show(block=False)

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

def layerStructure(network_layers):
    print("\nLayer shapes:")
    for i, layer in enumerate(network_layers):
        
        print(i, layer.weights.shape)

    horizontalRule()
