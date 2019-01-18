import torch
import numpy as np
from strategy import Strategy
from sklearn.cluster import KMeans


class KMeansSampling(Strategy):
    """
    This class is for K-Means sampling using the embedded features sampling for active learning. This method uses the
    trained model to extract features for the data. These features are then clustered using the K-Means clustering
    method and the centers of the clusters are then selected to be annotated.
    """

    def query(self, n):
        """
        Method for querying the data to be labeled. This method selects data that are centers of clusters specified by
        the K-Means clustering method.
        :param n: Amount of data to query.
        :return: Array of indices to data selected to be labeled.
        """

        unlabeled_indices = np.arange(self.pool_size)[~self.labeled_indices]

        predictions = self.predict(self.x, self.y)[0].numpy()

        regions = []
        for i in self.y:
            regions.append(len(i))

        region_predictions = []

        count = 0
        for r in regions:
            region = predictions[count:count + r]
            count = r
            region_predictions.append(np.average(region, 0))

        region_predictions = np.array(region_predictions)

        cluster_learner = KMeans(n_clusters=n, n_jobs=-1)
        cluster_learner.fit(region_predictions)

        cluster_indices = cluster_learner.predict(region_predictions)
        centers = cluster_learner.cluster_centers_[cluster_indices]
        distances = (region_predictions - centers) ** 2
        distances = distances.sum(axis=1)
        query_indices = np.array([np.arange(region_predictions.shape[0])[cluster_indices == i]
                                  [distances[cluster_indices == i].argmin()] for i in range(n)])

        return unlabeled_indices[query_indices]
