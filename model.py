import torch.nn as nn
import torch.nn.functional as functional


class Model(nn.Module):
    """
    The class for the model that will be used within the active learning experiments.
    This class can be replaced with a custom model depending on the dataset being used
    or task for the model such as segmentation.
    """

    def __init__(self, dropout=False):
        """
        The initialiser for the Model class that sets the functions of the models.
        """

        # Call the initiliser for the parent class.
        super(Model, self).__init__()

        # Defines the functions for each layer in the neural network.
        self.conv1 = nn.Conv2d(3, 36, kernel_size=4)
        self.conv2 = nn.Conv2d(36, 48, kernel_size=3)

        self.fc1 = nn.Linear(1200, 512)
        self.fc2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, 4)

        self.dropout = dropout
        self.embedding_size = 0

    def forward(self, x):
        """
        This method for the model is used to perform a forward pass.
        This method returns the output from the model and an intermediate output from the model
        used in some active learning query strategy.
        :param x: The data that will be passed through the model.
        :return: The output of the model.
        """

        # Uses the defined functions to perform a forward pass.
        conv1 = functional.max_pool2d(functional.relu(self.conv1(x)), 2)
        conv2 = functional.max_pool2d(functional.relu(self.conv2(conv1)), 2)
        flat = conv2.view(-1, 1200)

        fc1 = functional.relu(self.fc1(flat))
        fc1 = functional.dropout(fc1, training=self.training)

        fc2 = functional.relu(self.fc2(fc1))
        fc2 = functional.dropout(fc2, training=self.training)

        out = self.out(fc2)

        self.embedding_size = out.shape[1]

        # Method outputs the final output from the model.
        return out

    def get_embedding_dim(self):
        """
        Method that is used to get the size of the immediate output from the model.
        :return: An integer representing the size of the intermediate output.
        """

        return self.embedding_size
