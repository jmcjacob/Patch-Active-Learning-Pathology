import numpy as np
from strategy import Strategy
from scipy.spatial import distance_matrix


class KCentreGreedySampling(Strategy):
    """
    This class is for K-Centre sampling using the embedded features for active learning. This method uses the model
    trained model to extract features from the data. These features are then covered using the K-Centre cover method
    calculated using the greedy method. The centres of the covers are selected to be annotated.
    """

    # def query(self, n):
    #     """
    #     Method for querying the data to be labeled. This method selects data that are centers of clusters specified by
    #     the K-Centre cover method.
    #     :param n: Amount of data to query.
    #     :return: Array of indices to data selected to be labeled.
    #     """
    #
    #     labeled_flag = self.labeled_indices.copy()
    #     embedding = self.get_embeddings(self.x, self.y)
    #     embedding = embedding.numpy()
    #
    #     distance_matrix = np.dot(embedding.astype(np.float16), embedding.transpose().astype(np.float16))
    #     square = np.array(distance_matrix.diagonal()).reshape(len(self.x), 1)
    #     distance_matrix *= -2
    #     distance_matrix += square
    #     distance_matrix += square.transpose()
    #     distance_matrix = np.sqrt(distance_matrix)
    #
    #     matrix = distance_matrix[~labeled_flag, :][:, labeled_flag]
    #
    #     for i in range(n):
    #         matrix_minimum = matrix.min(axis=1)
    #         q_index_ = matrix_minimum.argmax()
    #         q_index = np.arange(self.pool_size)[~labeled_flag][q_index_]
    #         labeled_flag[q_index] = True
    #         matrix = np.delete(matrix, q_index_, 0)
    #         matrix = np.append(matrix, distance_matrix[~labeled_flag, q_index][:, None], axis=1)
    #
    #     return np.arange(self.pool_size)[(self.labeled_indices ^ labeled_flag)]

    def query(self, n):
        unlabeled_indices = np.arange(self.pool_size)[~self.labeled_indices]
        labeled_indices = np.arange(self.pool_size)[self.labeled_indices]

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

        labeled = region_predictions[labeled_indices]
        unlabeled = region_predictions[unlabeled_indices]

        greddy_indices = []

        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j + 100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        farthest = np.argmax(min_dist)
        greddy_indices.append(farthest)
        for i in range(n-1):
            dist = distance_matrix(unlabeled[greddy_indices[-1], :].reshape((1, unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greddy_indices.append(farthest)

        return np.array(greddy_indices, dtype=int)

