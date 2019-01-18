import numpy as np

from strategy import Strategy


class RandomSampling(Strategy):
    """
    The class is for random query method that randomly selects data to be labeled rather than use any
    intelligent methods. It is useful for a baseline test to beat.
    """

    def query(self, n):
        """
        Method for querying the data to be labeled. This method selects data to be labeled at random.
        :param n: Amount of data to query.
        :return: Array of indices to data selected to be labeled.
        """

        temp = np.where(self.labeled_indices == False)
        return np.random.choice(temp[0], n, replace=False)
