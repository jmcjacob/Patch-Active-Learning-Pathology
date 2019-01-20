import gc
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as functional


class Strategy:
    """
    The strategy class is how the active learning will be run and is used as the base class for the query strategies.
    It implements the basic needs of active learning strategies such as training and making predictions from the
    inputted model. The query method is an abstract method to be implemented by the query strategy classes.
    """

    def __init__(self, x, y, labeled_indices, model, data_handler, arguments):
        """
        The initiliser for the strategy class sets all the class members to the inputted values.
        It also sets the device that should be used to train the model based on if cuda if supported
        device is available.
        :param x: Array of training data.
        :param y: Array of training labels.
        :param labeled_indices: Binary Array of if the composing index should be treated as labeled.
        :param model: The Pytorch model that should be used within the active learning loop.
        :param data_handler: The data handler used to feed the model data.
        :param arguments: The ArgumentParser object containing the arguments for the class
        """

        self.x = x
        self.y = y
        self.labeled_indices = labeled_indices
        self.model = model
        self.data_handler = data_handler
        self.arguments = arguments
        self.pool_size = len(y)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def log(self, message):
        """
        Logging function for use within the strategy class and its derivative classes.
        :param message: The message to be logged and printed.
        """

        if self.arguments.verbose:
            print(message)
        if self.arguments.log_file != '':
            if not os.path.isfile(self.arguments.log_file):
                os.makedirs(os.path.dirname(self.arguments.log_file))
            print(message, file=open(self.arguments.log_file, 'a'))

    def query(self, n):
        """
        Abstract method to be overridden in the derivative classes.
        This method should return an array of indexes the query strategy has selected to be labeled.
        :param n: The number of items of data to selected for labeling.
        """

        pass

    def update(self, labeled_indices):
        """
        Method for updating the array that indicated which data should be treated as labeled.
        :param labeled_indices: Binary Array of if the composing index should be treated as labeled.
        """

        self.labeled_indices = labeled_indices

    def reset_model(self):
        """
        Method for resetting the weights of the model. This can be used between active learning iterations so see
        how adding new data improves the training of the model from anew rather than continuing to improve a single
        model.
        """

        def weights_init(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)

        self.model.apply(weights_init)

    def train(self):
        """
        This method will train the model using the training specified as labeled. This method uses a random validation
        set sampled from the labeled training data based on a specified percentage. Early stopping is also used and uses
        the training loss and the validation loss to determine when to stop training.
        """

        # Resets the model before training if specified.
        if self.arguments.reset_model:
            self.reset_model()

        min_val = 1000

        # Defines the classifier and optimiser for the model.
        self.classifier = self.model.to(self.device)
        optimiser = torch.optim.Adadelta(self.classifier.parameters())

        # Creates the data handler for the labeled training data.
        labeled_train = np.arange(self.pool_size)[self.labeled_indices]
        data_handler = self.data_handler(self.x[labeled_train], self.y[labeled_train])

        self.log("Training Data: {}".format(int(len(data_handler.x) * (1 - self.arguments.val_per))))
        self.log("Validation Data: {}".format(int(len(data_handler.x) * self.arguments.val_per)))

        lengths = [int(len(data_handler.y)*(1-self.arguments.val_per)), int(len(data_handler.y)*self.arguments.val_per)]
        if sum(lengths) > len(data_handler.y):
            lengths[1] -= 1
        elif sum(lengths) < len(data_handler.y):
            lengths[0] += 1

        # Splits the labeled training data into training and validation sets.
        train_handler, val_handler = torch.utils.data.random_split(data_handler, lengths)

        # Creates data loaders from the data handlers.
        train_loader = DataLoader(train_handler, shuffle=True, batch_size=self.arguments.batch_size)
        val_loader = DataLoader(val_handler, shuffle=True, batch_size=100)

        # Sets the list for the training and validation losses.
        train_losses, val_losses = [], []

        # The main training loop for the model.
        for epoch in range(self.arguments.max_epochs):
            # Sets the model to training mode.
            self.classifier.train()

            # Resets the training loss for the epoch and the batch count.
            train_loss, train_batch = 0, 0

            # Cycles though the batches of the training by enumerating through the training data loader.
            for batch_index, (x, y, index) in enumerate(train_loader):
                # Copies the training data and labels to the device.
                x, y = x.to(self.device), y.to(self.device)

                # Resets the gradients in the optimiser.
                optimiser.zero_grad()

                # Performs a forward pass with the data through the model.
                out = self.classifier(x)

                _, indices = y.max(1)

                # Calculates the loss comparing the output and the labels.
                loss = functional.cross_entropy(out, indices, torch.tensor(data_handler.weights, device=self.device))

                # Adds the batch loss to the epoch loss.
                train_loss += loss.detach().item()

                # Performs a backward pass though the model using the loss
                loss.backward()

                # Updates the weights of the model using the gradients.
                optimiser.step()

                # Updates the batch count.
                train_batch += 1

            # Sets the model to evaluation mode.
            self.classifier.eval()

            # Resets the epochs validation loss and validation batch count.
            val_loss, val_batch = 0, 0

            # Sets the model to not calculate gradients when making predictions.
            with torch.no_grad():
                # Cycles though the batches of the validation by enumerating through the validation data loader.
                for batch_index, (x, y, index) in enumerate(val_loader):
                    # Moves the validation data and labels to the device.
                    x, y = x.to(self.device), y.to(self.device)

                    # Performs a forward pass with the data through the model.
                    out = self.classifier(x)

                    _, indices = y.max(1)

                    # Calculates the loss comparing the output and the labels.
                    loss = functional.cross_entropy(out, indices, torch.tensor(data_handler.weights, device=self.device))

                    # Adds the batch loss to the epoch loss.
                    val_loss += loss.detach().item()

                    # Updates the batch count.
                    val_batch += 1

            # Adds the training epoch loss and validation loss to the list of losses.
            train_losses.append(train_loss / train_batch)
            val_losses.append(val_loss / val_batch)

            if (val_loss / val_batch) < min_val:
                min_val = val_loss / val_batch
                torch.save(self.classifier.state_dict(), 'Min_Val.pt')

            # Logs the epoch and losses at the end of the epoch.
            self.log("Epoch: {0}\tLoss: {1:.5f}\tVal Loss: {2:.5f}".format(epoch + 1, train_losses[-1], val_losses[-1]))

            # Early stopping to decide if training should be stopped earlier.
            # Checks if the minimum number of epochs has been achieved.
            if epoch >= self.arguments.min_epochs:
                # Calculates the general loss using the validation losses.
                g_loss = 100 * ((val_losses[-1] / min(val_losses[:-1])) - 1)

                # Calculates the training progress using a window from the training losses.
                t_progress = 1000 * ((sum(train_losses[-(self.arguments.batch + 1):-1]) /
                                      (self.arguments.batch * min(train_losses[-(self.arguments.batch + 1):-1]))) - 1)

                # Checks if the early stopping has reached its target.
                if not (t_progress <= 0 and g_loss <= 0):
                    if g_loss / t_progress >= self.arguments.target:
                        # Ends training by exiting the main training loop.
                        break

            # Forces the garbage collector to collect the unused values.
            gc.collect()

        # Logs when the training finished.
        self.log("Training Finished at Epoch {0}".format(epoch + 1))

    def predict(self, x, y):
        """
        This method is used for making predictions on a set of data with the trained model.
        :param x: The data that the model will make predictions on.
        :param y: The labels for the data the model is making predictions on.
        :return: A list of softmax predictions and the predicted labels for each piece of data.
        """

        # Creates the data handler fot the inputted data.
        data_handler = self.data_handler(x, y, False)
        test_loader = DataLoader(data_handler, shuffle=False, batch_size=1000)

        self.classifier.load_state_dict(torch.load('Min_Val.pt'))

        # Sets the classifier to evaluation mode or training mode if dropout is selected to be used.
        if self.model.dropout:
            self.classifier.train()
        else:
            self.classifier.eval()

        # Creates an empty array or the predictions and predicted label.
        predictions = torch.zeros([len(data_handler.y), 4])
        predicted_label = torch.zeros(len(data_handler.y), dtype=torch.long)

        # Ensures that gradients are not calculated.
        with torch.no_grad():
            # Enumerates though the dataset using the data loader.
            for batch_index, (x, _, index) in enumerate(test_loader):
                # Moves the data to the specified device.
                x = x.to(self.device)

                # Performs a forward pass from the model using the data.
                out = self.classifier(x)

                # Uses a softmax function on the output to return the predictions and adds them to a list.
                prediction = functional.softmax(out, dim=1)
                predictions[index] = prediction.cpu()

                # Uses the maz function on the output to return the predicted label and adds them to a list.
                prediction = out.max(1)[1]
                predicted_label[index] = prediction.cpu()

        # Returns the lists of predictions and predicted labels.
        return predictions, predicted_label

    def get_embeddings(self, x, y):
        """
        Method for extracting the embedded representation features from the model using the inputted data.
        :param x: Array of data to have features extracted.
        :param y: Array of labels for the data.
        :return: Array of feature representations for the data.
        """

        # Creates the data handler for the inputted data.
        data_handler = self.data_handler(x, y)
        test_loader = DataLoader(data_handler, shuffle=False, batch_size=1000)

        # Sets the classifier to evaluation mode or training mode if dropout is selected to be used.
        if self.model.dropout:
            self.classifier.train()
        else:
            self.classifier.eval()

        # Creates an empty array for the embeddings to be extracted.
        embedding = torch.zeros([len(y), self.classifier.get_embedding_dim()])

        # Ensures that gradients are not calculated.
        with torch.no_grad():
            # Enumerates though the dataset using the data loader.
            for batch_index, (x, _, index) in enumerate(test_loader):
                # Moves the data to the specified device.
                x = x.to(self.device)

                # Performs a forward pass though the model using the data and returns the embeddings.
                embeddings = self.classifier(x)

                # Adds the embeddings to the list.
                embedding[index] = embeddings.cpu()

        # Returns the list of embeddings.
        return embedding
