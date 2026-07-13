import matplotlib.pyplot as plt


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
