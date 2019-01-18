import torch
import numpy as np
from strategy import Strategy


class EntropySampling(Strategy):
    """
    This class is for the entropy based query method that selects data with the highest entropy across the softmax
    predictions. It should be noted that with a binary classification problem entropy sampling is equal to least
    confident sampling. This method is a simple method for active learning that could be useful for use as a baseline in
    comparision with other methods.
    """

    def __init__(self, x, y, labeled_indices, model, data_handler, arguments, iterations=1):
        super(EntropySampling, self).__init__(x, y, labeled_indices, model, data_handler, arguments)
        self.number_iterations = iterations

    def query(self, n):
        """
        Method for querying the data to be labeled. This method selects data with the highest entropy value.
        :param n: Amount of data to query.
        :return: Array of indices to data selected to be labeled.
        """

        unlabeled_indices = np.arange(self.pool_size)[~self.labeled_indices]
        probabilities = []

        for i in range(self.number_iterations):
            prob = self.predict(self.x[unlabeled_indices], self.y[unlabeled_indices])[0].numpy()
            probabilities.append(prob)

        if self.number_iterations > 1:
            probabilities = np.var(np.asarray(probabilities), axis=0)

        regions = []
        for i in self.y[unlabeled_indices]:
            regions.append(len(i))

        probabilities = torch.as_tensor(probabilities)

        log_probabilities = torch.log(probabilities)
        uncertainties = (probabilities * log_probabilities).sum(1)

        regions_uncertainities = []

        count = 0
        for region in regions:
            region_uncertainities = uncertainties[count:count + region]
            count = region
            # regions_uncertainities.append(max(region_uncertainities))
            regions_uncertainities.append(np.average(region_uncertainities))

        regions_uncertainities = torch.from_numpy(np.array(regions_uncertainities))

        return unlabeled_indices[regions_uncertainities.sort()[1][:n]]
