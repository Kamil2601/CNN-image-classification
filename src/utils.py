import matplotlib.pyplot as plt

def plot_history(history: dict):
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(1, 2, 1)
    plt.plot(history['train_loss'], '-o')
    plt.plot(history['val_loss'], '--<')
    plt.legend(['Train loss', 'Validation loss'], fontsize=10)
    ax.set_xlabel('Epochs', size=15)
    ax = fig.add_subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], '-o')
    plt.plot(history['val_accuracy'], '-<')
    plt.legend(['Train acc.', 'Validation acc.'], fontsize=10)
    ax.set_xlabel('Epochs', size=10)