import numpy as np
import matplotlib.pyplot as plt

# Convert square units to decibels
def db(x):
    return 10 * np.log10(x)

# Show and update console progress bar
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Show final model metrics and plot trainig curves
def show_model_metrics(model, modelname):
    # Show model metrics
    print(modelname + ' Metrics:')
    print('Train Loss: {0:.5f}'.format(model.loss_train[-1]))
    print('Valid Loss: {0:.5f}'.format(model.loss_valid[-1]))
    print('Train Accuracy: {0:.2f} %'.format(model.accuracy_train[-1]))
    print('Valid Accuracy: {0:.2f} %'.format(model.accuracy_valid[-1]))

    # Initialize multiple plot figure
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))

    # Plot step reduction epochs
    ax[0].axvline(x = 15, color = 'r', linestyle='dashed', label='Step reduction 1')
    ax[0].axvline(x = 35, color = 'g', linestyle='dashed', label='Step reduction 2')
    ax[1].axvline(x = 15, color = 'r', linestyle='dashed')
    ax[1].axvline(x = 35, color = 'g', linestyle='dashed')

    # Plot loss curves
    epochs = np.arange(1, len(model.loss_train) + 1)
    ax[0].plot(epochs, db(model.loss_train), label='Train set')
    ax[0].plot(epochs, db(model.loss_valid), label='Valid set')

    # Plot accuracy curves
    ax[1].plot(epochs, model.accuracy_train)
    ax[1].plot(epochs, model.accuracy_valid)

    # Set axis limits
    ax[0].set_xlim([0, epochs[-1]])
    ax[1].set_xlim([0, epochs[-1]])

    # Set plot titles and labels
    ax[0].set_xlabel("Epoch Number")
    ax[0].set_ylabel("Loss (dB)")
    ax[1].set_xlabel("Epoch Number")
    ax[1].set_ylabel("Accuracy (%)")
    ax[0].set_title("Loss Curves")
    ax[1].set_title("Accuracy Curves")

    # Setup parent figure
    fig.suptitle(modelname + ' Training Results', y=1.04)
    fig.subplots_adjust(hspace=1.5, wspace=0.4)
    fig.legend(bbox_to_anchor=(0.9, -0.2), loc='lower right', ncol=4)
    plt.show()