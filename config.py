import sys
from argparse import ArgumentParser
from configparser import ConfigParser


def load_config(description):
    """
    Loads the arguments from the config file and the command line.
    Arguments from the command line override the config file.
    The config file will be loaded from the default location ./config.ini unless specified in the command line.
    :return: A ArgumentParser object containing the loaded configurations.
    """

    # Sets the description of the application and creates an ArgumentParser to read command line arguments.
    parser = ArgumentParser(description=description)

    # Creates a ConfigParser to read config file arguments.
    config = ConfigParser()

    # Loads either given config file from the command line or default config file.
    if len(sys.argv) > 1:
        if sys.argv[1] == "--config_file":
            config.read(sys.argv[1].split('=')[1])
        else:
            config.read("config.ini")
    else:
        config.read("config.ini")

    # Standard Parameters
    parser.add_argument("--config_file", type=str, default="config.ini",
                        help="File path to configurations file.")
    parser.add_argument("--verbose", action="store_true",
                        default=config["standard"]["verbose"].lower() == "true",
                        help="Boolean if the program should display outputs while running.")
    parser.add_argument("--log_file", type=str, default=config["standard"]["log_file"],
                        help="Filepath where the program's output should be logged.")
    parser.add_argument("--seed", type=int, default=int(config["standard"]["seed"]),
                        help="Integer for the random seed.")

    # Active Learning Parameters
    parser.add_argument("--query_strategy", type=str, default=config["active_learning"]["query_strategy"],
                        help="Define the type of query strategy.")
    parser.add_argument("--init_labels", type=int, default=int(config["active_learning"]["init_labels"]),
                        help="The number of data that should be labelled initially.")
    parser.add_argument("--query_labels", type=int, default=int(config["active_learning"]["query_labels"]),
                        help="The number of data that should be labelled each iteration.")
    parser.add_argument("--num_iterations", type=int, default=int(config["active_learning"]["num_iterations"]),
                        help="The number of iterations the active learner should run for.")
    parser.add_argument("--reset_model", action="store_true",
                        default=config["active_learning"]["reset_model"].lower() == "true",
                        help="Reset the weights of the model betwen active learning iterations.")
    parser.add_argument("--dropout_iterations", type=int, default=int(config["active_learning"]["dropout_iterations"]),
                        help="The number of prediction iterations to be performed that will be averaged.")

    # Model Parameters
    parser.add_argument("--val_per", type=float, default=float(config["model"]["val_per"]),
                        help="The percentage of training data should be used for validation.")
    parser.add_argument("--learning_rate", type=float, default=float(config["model"]["learning_rate"]),
                        help="The learning rate to train the model.")
    parser.add_argument("--batch_size", type=int, default=int(config["model"]["batch_size"]),
                        help="The batch size for training the model.")
    parser.add_argument("--momentum", type=float, default=float(config["model"]["momentum"]),
                        help="The momentum for training the model.")

    # Early Stopping Parameters
    parser.add_argument("--min_epochs", type=int, default=int(config["early_stopping"]["min_epochs"]),
                        help="The minimum number of epochs to train a model.")
    parser.add_argument("--max_epochs", type=int, default=int(config["early_stopping"]["max_epochs"]),
                        help="The maximum number of epochs to train a model.")
    parser.add_argument("--batch", type=int, default=int(config["early_stopping"]["batch"]),
                        help="The size of the batch the early stopping method should use.")
    parser.add_argument("--target", type=float, default=float(config["early_stopping"]["target"]),
                        help="The target for the early stopping method.")

    # Dataset Parameters
    parser.add_argument("--dataset_dir", type=str, default=str(config["dataset"]["dataset_dir"]),
                        help="The directory of the dataset.")

    """
    To add new arguments to the ArgumentParser the asrgument needs to be added here and an entry to the config file
    needs to be added. The default value for the argument should be set to the argument from the config file.
    """

    return parser.parse_args()
