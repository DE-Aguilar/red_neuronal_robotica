import matplotlib.pyplot as plt
import numpy as np

def show_predictions( pred, Y, percentage=1):
    """
    Shows:
      1. Training loss
      2. Predictions vs Ground Truth
      3. Prints first and last <percentage>% of predictions
    """

    n = len(Y)
    count = max(1, int(n * percentage / 100))

    # --------------------------
    # Print predictions
    # --------------------------
    print(f"\nPred vs True (first {percentage}% | {count} samples):")
    for i in range(count):
        print(f"pred={pred[i][0]:.4f} | true={Y[i][0]:.4f}")

    print(f"\nPred vs True (last {percentage}% | {count} samples):")
    for i in range(n - count, n):
        print(f"pred={pred[i][0]:.4f} | true={Y[i][0]:.4f}")

    # --------------------------
    # Graph 1: Loss
    # --------------------------
   # -------- First percentage --------
   # -------- First percentage --------
    first_pred = pred[:count].flatten()
    first_true = Y[:count].flatten()
    first_diff = np.abs(first_pred - first_true)

    # -------- Last percentage --------
    last_pred = pred[-count:].flatten()
    last_true = Y[-count:].flatten()
    last_diff = np.abs(last_pred - last_true)

    plt.figure(figsize=(12,5))

    plt.plot(first_diff, marker='o', label=f'First {percentage}%')
    plt.plot(last_diff, marker='x', label=f'Last {percentage}%')

    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Prediction Error Comparison')
    plt.xlabel('Sample')
    plt.ylabel('|Prediction - True|')
    plt.grid(True)
    plt.legend()



   
    plt.show()


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