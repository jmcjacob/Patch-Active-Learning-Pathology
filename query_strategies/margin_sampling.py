import torch
import numpy as np
from strategy import Strategy


class MarginSampling(Strategy):
    """
    This class is for the margin based query method that selects data where the two highest predictions have the least
    difference. This method is a simple method for active learning that could be useful for use as a baseline in
    comparision with other methods.
    """

    def __init__(self, x, y, labeled_indices, model, data_handler, arguments, iterations=1):
        super(MarginSampling, self).__init__(x, y, labeled_indices, model, data_handler, arguments)
        self.number_iterations = iterations

    def query(self, n):
        """
        Method for querying the data to be labeled. This method selects data with the smallest diffence in the top two
        predictions.
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

        probabilities_sorted, indices = probabilities.sort(descending=True)
        uncertainties = probabilities_sorted[:, 0] - probabilities_sorted[:, 1]

        regions_uncertainities = []

        count = 0
        for region in regions:
            region_uncertainities = uncertainties[count:count + region]
            count = region
            # regions_uncertainities.append(max(region_uncertainities))
            regions_uncertainities.append(np.average(region_uncertainities))

        regions_uncertainities = torch.from_numpy(np.array(regions_uncertainities))

        return unlabeled_indices[regions_uncertainities.sort()[1][:n]]
