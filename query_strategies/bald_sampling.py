import torch
import numpy as np
from strategy import Strategy

# TODO: Add Comments to the class expplaining its uses and how it works.

class BALDSampling(Strategy):
    def __init__(self, x, y, labeled_indices, model, data_handler, arguments, iterations=1):
        super(BALDSampling, self).__init__(x, y, labeled_indices, model, data_handler, arguments)
        self.number_iterations = iterations

    def query(self, n):
        unlabeled_indices = np.arange(self.pool_size)[~self.labeled_indices]
        probabilities = []

        for i in range(self.number_iterations):
            prob = self.predict(self.x[unlabeled_indices], self.y[unlabeled_indices])[0].numpy()
            probabilities.append(prob)

        probabilities = torch.as_tensor(probabilities)

        mean_probabilities = probabilities.mean(0)
        entropy1 = (-mean_probabilities * torch.log(mean_probabilities)).sum(1)
        entropy2 = (-probabilities * torch.log(probabilities)).sum(2).mean(0)
        uncertainties = entropy2 - entropy1

        regions = []
        for i in self.y[unlabeled_indices]:
            regions.append(len(i))

        regions_uncertainities = []

        count = 0
        for region in regions:
            region_uncertainities = uncertainties[count:count + region]
            count = region
            # regions_uncertainities.append(max(region_uncertainities))
            regions_uncertainities.append(np.average(region_uncertainities))

        regions_uncertainities = torch.from_numpy(np.array(regions_uncertainities))

        return unlabeled_indices[regions_uncertainities.sort()[1][:n]]
