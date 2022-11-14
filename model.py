import h5py
import numpy as np
from utils import printProgressBar

# ReLU fuction
def relu(x):
    return (x > 0) * x

# ReLU derivative
def drelu(x):
    return (x > 0) * 1

# tan function
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# tanh derivative
def dtanh(x):
    return 1 - np.tanh(x)**2

# Softmax function
def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

class NeuralNet:

    def __init__(self, size, bsize, step, h, dh, init=None):
        # Initialize network dimensions and parameters
        self.L = len(size) - 1  # No. of layers (excl. input layer)
        self.size = size
        self.bsize = bsize
        self.step = step

        # Initialize activation function and its derivative
        self.h = h
        self.dh = dh

        # Initialize metrics
        self.loss_train = []  # Training set loss across training
        self.loss_valid = []  # Validation set loss across training
        self.accuracy_train = [] # Training set accuracy accross training
        self.accuracy_valid = [] # Validation set accuracy accross training

        # Initialize weights and biases
        self.W = []
        self.b = []
        for i in range(1, len(size)):
            # Kaiming He initialization (ReLU)
            if init == 'he':
                var = 2 / size[i-1]
                Wi = np.random.normal(0, var, size=(size[i], size[i-1]))
                bi = np.random.normal(0, var, size=size[i])
            # Xavier initialization (tanh)
            elif init == 'xavier':
                bound = np.sqrt(6 / (size[i] + size[i-1]))
                Wi = np.random.uniform(-bound, bound, size=(size[i], size[i-1]))
                bi = np.random.uniform(-bound, bound, size=size[i])
            # Standard normal initialization
            else:
                Wi = np.random.standard_normal(size=(size[i], size[i-1]))
                bi = np.random.standard_normal(size=size[i])
            self.W.append(Wi)
            self.b.append(bi)

    def set_weights(self, W, b):
        self.W = [el.copy() for el in W]
        self.b = [el.copy() for el in b]

    # Read model from file
    def load_model(self, filename):
        self.W = []
        self.b = []
        with h5py.File(filename, 'r') as data:
            self.size = data['size'][:]
            self.step = data['step']
            self.L = len(self.size) - 1
            for i in range(0, self.L):
                self.W.append(data['W' + str(i)][:])
                self.b.append(data['b' + str(i)][:])
            self.loss_train = list(data['loss_train'][:])
            self.loss_valid = list(data['loss_valid'][:])
            self.accuracy_train = list(data['accuracy_train'][:])
            self.accuracy_valid = list(data['accuracy_valid'][:])

    # Save model to file
    def save_model(self, filename):
        with h5py.File(filename, 'w') as file:
            file.create_dataset('size', data = self.size)
            file.create_dataset('step', data = self.step)
            for i in range(0, self.L):
                file.create_dataset('W' + str(i), data = self.W[i])
                file.create_dataset('b' + str(i), data = self.b[i])
            file.create_dataset('loss_train', data = self.loss_train)
            file.create_dataset('loss_valid', data = self.loss_valid)
            file.create_dataset('accuracy_train', data = self.accuracy_train)
            file.create_dataset('accuracy_valid', data = self.accuracy_valid)
    
    # Perform forward pass through network
    def forward(self, x):
        a = [x]  # Activations
        da = []  # Activation derivatives
        
        # Compute activations for each layer
        for l in range(0, self.L):
            s = self.W[l] @ a[l] + self.b[l]
            if l < self.L - 1:
                a.append(self.h(s))
                da.append(self.dh(s))
            else:
                a.append(softmax(s))

        return a, da

    # Compute gradient deltas for each layer
    def compute_deltas(self, y, a, da):
        d = [None] * self.L
        d[self.L - 1] = a[self.L] - y
        for l in range(self.L - 2, -1, -1):
            d[l] = da[l] * (self.W[l + 1].T @ d[l + 1])
        return d

    # Train model and compute training metrics
    def train(self, x_train, y_train, x_valid=None, y_valid=None, epochs=10):
        # Make deep copies of training data
        x_train = x_train.copy()
        y_train = y_train.copy()

        # Initialize trainig parameters
        N = len(x_train)
        batches = N // self.bsize
        bstep = self.step / self.bsize  # Step normalized by batch size

        # Train model
        for epoch in range(0, epochs):
            # Initialize progress output for epoch
            printProgressBar(0, batches, prefix = 'Epoch ' + str(epoch) + ':', suffix = 'Complete', length = 50)

            # Initialize training set metrics
            loss_train_n = 0
            accuracy_train_n = 0

            # Learning rate reduction
            total_epochs = self.loss_train
            if total_epochs in [15, 35]:
                self.step = self.step / 2
                bstep = self.step / self.bsize

            for b in range(0, batches):
                # Extract batch
                b_start, b_end = b * self.bsize, (b + 1) * self.bsize
                x_batch = x_train[b_start: b_end]
                y_batch = y_train[b_start: b_end]

                # Initialize sums for weight and bias deltas
                dW = []
                db = []
                for i in range(1, len(self.size)):
                    dW.append(np.zeros((self.size[i], self.size[i-1])))
                    db.append(np.zeros(self.size[i]))

                # Train model on batch
                for i in range(0, self.bsize):
                    # Perform forward pass and compute deltas
                    a, da = self.forward(x_batch[i])
                    d = self.compute_deltas(y_batch[i], a, da)

                    # Update sums for weight and bias deltas
                    for l in range(0, self.L):
                        dW[l] = dW[l] + np.outer(d[l], a[l])
                        db[l] = db[l] + d[l]

                    # Compute loss on training sample
                    loss_train_n = loss_train_n + self.compute_loss(a[self.L], y_batch[i])
                    accuracy_train_n = accuracy_train_n + self.evaluate(a[self.L], y_batch[i])

                # Update weights
                for l in range(0, self.L):
                    self.W[l] = self.W[l] - bstep * dW[l]
                    self.b[l] = self.b[l] - bstep * db[l]

                # Display progress
                printProgressBar(b + 1, batches, prefix = 'Epoch ' + str(epoch) + ':', suffix = 'Complete', length = 50)

            # Shuffle training data
            if epoch < epochs - 1:
                p = np.random.permutation(N)
                x_train = x_train[p]
                y_train = y_train[p]

            # Compute loss and accuracy on test set
            loss_train_n = loss_train_n / N
            accuracy_train_n = accuracy_train_n / N * 100
            self.loss_train.append(loss_train_n)
            self.accuracy_train.append(accuracy_train_n)
        
            # Compute loss and accuracy on validation set
            if x_valid != None:
                loss_valid_n, accuracy_valid_n = self.compute_metrics(x_valid, y_valid)
                self.loss_valid.append(loss_valid_n)
                self.accuracy_valid.append(accuracy_valid_n)

            # Show metrics after epoch
            print('Train Set : (Loss = {0:.5f} | Accuracy = {1:.2f} %)'
                .format(self.loss_train[-1], self.accuracy_train[-1]))
            if x_valid != None:
                print('Valid Set : (Loss = {0:.5f} | Accuracy = {1:.2f} %)'
                    .format(self.loss_valid[-1], self.accuracy_valid[-1]))

    # Compute model loss and accuracy for sample set
    def compute_metrics(self, x, y):
        N = len(x)
        loss = 0
        correct = 0
        for i in range(0, N):
            a, da = self.forward(x[i])
            loss = loss + self.compute_loss(a[self.L], y[i])
            correct = correct + self.evaluate(a[self.L], y[i])
        loss = loss / N
        accuracy = correct / N * 100
        return loss, accuracy

    # Compute the cross entropy loss for a single sample
    def compute_loss(self, a, y):
        ai = a[np.argmax(y)]
        if ai == 0:
            ai = 1e-15
        return -1 * np.log(ai)

    # Evaluate if prediction is correct
    def evaluate(self, a, y):
        return 1 * (np.argmax(a) == np.argmax(y))