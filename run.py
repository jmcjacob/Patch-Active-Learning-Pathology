import model
import config
import dataset
import strategy
import query_strategies

import os
import torch
import numpy as np


def log(arguments, message):
    """
    Function to handle printing and logging od messages.
    :param arguments: An ArgumentParser object.
    :param message: String with the message to be printed or logged.
    """

    if arguments.verbose:
        print(message)
    if arguments.log_file != '':
        if not os.path.isfile(arguments.log_file):
            if not os.path.isdir(os.path.dirname(arguments.log_file)):
                os.makedirs(os.path.dirname(arguments.log_file))
        print(message, file=open(arguments.log_file, 'a'))


if __name__ == '__main__':
    # Loads the arguments from the config file and command line and sets the description for the application.
    arguments = config.load_config(description="Active Learning Experiment Framework")

    log(arguments, "Arguments Loaded")

    # Sets the seeds for numpy and pytorch to the defined seeds.
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)

    # Sets CUDNN to be used by torch,
    torch.backends.cudnn.enabled = True

    # Extracts the training and testing data from the defined dataset.
    x_train, y_train, x_test, y_test = dataset.get_dataset(arguments.dataset_dir)
    data_handler = dataset.DataHandler

    """
    To conduct Active Learning a binary array is used that states if a peice of data should used within the labeled
    training data or should be treated as unlabeled data. An initial random sample of data of a specified size if then
    marked as labelled. 
    """
    labeled_indices = np.zeros(len(y_train), dtype=bool)
    labeled_temp = np.arange(len(y_train))
    np.random.shuffle(labeled_temp)
    labeled_indices[labeled_temp[:arguments.init_labels]] = True

    """
    This sets the model that will be used within the active learning experiments this can be replaced by any other
    pytorch model based of the nn.module class. The forward pass method also needs to include two outputs, the output
    of the model and an intermediate output from the centre (if not needed it can return None as the second output).
    Another method needed to be implemented is the get_embedding_dim that should specify the size of the
    intermediate output from the model.
    """
    model = model.Model()

    query_strategy = None

    """
    This is a query strategy is set. Each query strategy should be its own class that inherits from the Strategy base
    class with a custom query method. If no query method is selected the model will be trained using fully supervised
    learning.
    """

    if arguments.query_strategy.lower() == "random":
        query_strategy = query_strategies.RandomSampling(x_train, y_train, labeled_indices, model, data_handler,
                                                         arguments)
    if arguments.query_strategy.lower() == "least_confident":
        query_strategy = query_strategies.LeastConfident(x_train, y_train, labeled_indices, model, data_handler,
                                                         arguments)
    if arguments.query_strategy.lower() == "least_confident_dropout":
        model.dropout = True
        query_strategy = query_strategies.LeastConfident(x_train, y_train, labeled_indices, model, data_handler,
                                                         arguments, arguments.dropout_iterations)
    if arguments.query_strategy.lower() == "margin":
        query_strategy = query_strategies.MarginSampling(x_train, y_train, labeled_indices, model, data_handler,
                                                         arguments)
    if arguments.query_strategy.lower() == "margin_dropout":
        model.dropout = True
        query_strategy = query_strategies.MarginSampling(x_train, y_train, labeled_indices, model, data_handler,
                                                         arguments, arguments.dropout_iterations)
    if arguments.query_strategy.lower() == "entropy":
        query_strategy = query_strategies.EntropySampling(x_train, y_train, labeled_indices, model, data_handler,
                                                          arguments)
    if arguments.query_strategy.lower() == "entropy_dropout":
        model.dropout = True
        query_strategy = query_strategies.EntropySampling(x_train, y_train, labeled_indices, model, data_handler,
                                                          arguments, arguments.dropout_iterations)
    if arguments.query_strategy.lower() == "bald":
        model.dropout = True
        query_strategy = query_strategies.BALDSampling(x_train, y_train, labeled_indices, model, data_handler,
                                                       arguments, arguments.dropout_iterations)
    if arguments.query_strategy.lower() == "kmeans":
        query_strategy = query_strategies.KMeansSampling(x_train, y_train, labeled_indices, model, data_handler,
                                                         arguments)
    if arguments.query_strategy.lower() == "kcentre":
        query_strategy = query_strategies.KCentreGreedySampling(x_train, y_train, labeled_indices, model,
                                                                data_handler, arguments)
    if arguments.query_strategy.lower() == "core_set":
        query_strategy = query_strategies.CoreSetSampling(x_train, y_train, labeled_indices, model, data_handler,
                                                          arguments)
    if arguments.query_strategy.lower() == "deep_fool":
        query_strategy = query_strategies.DeepFoolSampling(x_train, y_train, labeled_indices, model, data_handler,
                                                           arguments)

    # If no query strategy the model will be trained in a supervised manner.
    if query_strategy is None:
        query_strategy = strategy.Strategy(x_train, y_train, labeled_indices, model, data_handler, arguments)
        labeled_indices[:] = True
        arguments.num_iterations = 0

    # Logs information about the current active learning iteration.
    log(arguments, "\n---------- Iteration 0")
    log(arguments, "Number of initial labeled regions: {}".format(list(labeled_indices).count(True)))
    log(arguments, "Number of initial unlabeled regions: {}".format(len(y_train) - list(labeled_indices).count(True)))
    log(arguments, "Number of testing regions: {}\n".format(len(y_test)))

    query_strategy.train()
    _, predictions = query_strategy.predict(x_test, y_test)
    accuracy = np.zeros(arguments.num_iterations + 1)

    y = []
    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            y.append(y_test[i][j])
    y = torch.tensor(y).max(1)[1]

    accuracy[0] = 1.0 * (y == predictions).sum().item() / len(y)
    log(arguments, "\nTesting Accuracy {}\n\n\n".format(accuracy[0]))

    for iteration in range(1, arguments.num_iterations+1):
        # Runs the specified query method and return the selected indices to be annotated.
        query_indices = query_strategy.query(arguments.query_labels)

        # Update the selected indices as labeled.
        labeled_indices[query_indices] = True
        query_strategy.update(labeled_indices)

        # Logs information about the current active learning iteration.
        log(arguments, "\n---------- Iteration " + str(iteration))
        log(arguments, "Number of initial labeled data: {}".format(list(labeled_indices).count(True)))
        log(arguments, "Number of initial unlabeled data: {}".format(len(y_train) - list(labeled_indices).count(True)))
        log(arguments, "Number of testing data: {}".format(len(y_test)))

        # Train the model with the new training set.
        query_strategy.train()

        # Get the predictions from the testing set.
        _, predictions = query_strategy.predict(x_test, y_test)

        y = []
        for i in range(len(y_test)):
            for j in range(len(y_test[i])):
                y.append(y_test[i][j])
        y = torch.tensor(y).max(1)[1]

        # Calculates the accuracy of the model based on the model's predictions.
        accuracy[iteration] = 1.0 * (y == predictions).sum().item() / len(y)

        # Logs the testing accuracy.
        log(arguments, "Testing Accuracy {}\n\n\n".format(accuracy[iteration]))

    # Logs the accuracies from all iterations.
    log(arguments, accuracy)
    log(arguments, "\n\n\n\n\n")
