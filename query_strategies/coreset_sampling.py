from strategy import Strategy

import os
import gc
import torch
import pickle
import shutil
import numpy as np
import gurobipy as gurobi
from scipy.spatial import distance_matrix

# TODO: Comment and better format everything.


class CoreSetSampling(Strategy):
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

        print("Calculating Greedy K-Center Solution...")
        new_indices, max_delta = self.greedy_k_center(region_predictions[labeled_indices],
                                                      region_predictions[unlabeled_indices], n)
        new_indices = unlabeled_indices[new_indices]
        outlier_count = int(len(self.x) / 10000)
        submipnodes = 20000

        eps = 0.01
        upper_bound = max_delta
        lower_bound = max_delta / 2.0
        print("Building MIP Model...")
        model, graph = self.mip_model(region_predictions, labeled_indices, len(labeled_indices) + n, upper_bound,
                                      outlier_count, greddy_indices=new_indices)

        model.Params.SubMIPNodes = submipnodes
        points = model.__data[0]
        model.optimize()

        indices = [i for i in graph if points[i].x == 1]

        current_delta = upper_bound
        bound = upper_bound - lower_bound
        while bound < eps:
            print("upper bound is {ub}, lower bound is {lb}".format(ub=upper_bound, lb=lower_bound))
            if model.getAttr(gurobi.GRB.Attr.Status) in [gurobi.GRB.INFEASIBLE, gurobi.GRB.TIME_LIMIT]:
                print("Optimization Failed - Infeasible!")

                lower_bound = max(current_delta, self.get_graph_min(region_predictions, current_delta))
                current_delta = (upper_bound + lower_bound) / 2.0

                del model
                gc.collect()
                model, graph = self.mip_model(region_predictions, labeled_indices, len(labeled_indices) + n, current_delta,
                                              outlier_count, greedy_indices=indices)
                points, outliers = model.__data
                model.Params.SubMIPNodes = submipnodes

            else:
                print("Optimization Succeeded!")
                upper_bound = min(current_delta, self.get_graph_max(region_predictions, current_delta))
                current_delta = (upper_bound + lower_bound) / 2.0
                indices = [i for i in graph if points[i].X == 1]

                del model
                gc.collect()
                model, graph = self.mip_model(region_predictions, labeled_indices, len(labeled_indices) + n, current_delta,
                                              outlier_count, greedy_indices=indices)
                points, outliers = model.__data
                model.Params.SubMIPNodes = submipnodes

            if upper_bound - lower_bound > eps:
                model.optimize()

        return np.array(indices)

    def greedy_k_center(self, labeled, unlabeled, n):
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

        return np.array(greddy_indices, dtype=int), np.max(min_dist)

    def mip_model(self, embeddings, labeled_indices, n, delta, outlier_count, greddy_indices):
        model = gurobi.Model("Core Set Selection")

        points = {}
        outliers = {}
        for i in range(embeddings.shape[0]):
            if i in labeled_indices:
                points[i] = model.addVar(ub=1., lb=1., vtype="B", name="points_{}".format(i))
            else:
                points[i] = model.addVar(vtype="B", name="points_{}".format(i))
        for i in range(embeddings.shape[0]):
            outliers[i] = model.addVar(vtype="B", name="outliers_{}".format(i))
            outliers[i].start = 0

        if greddy_indices is not None:
            for i in greddy_indices:
                points[i].start = 1.0

        model.addConstr(sum(outliers[i] for i in outliers) <= outlier_count, "budget")

        model.addConstr(sum(points[i] for i in range(embeddings.shape[0])) == n, "budget")
        graph = {}
        print("Updating Neighborhoods In MIP Model...")
        for i in range(0, embeddings.shape[0], 1000):
            print("At Point " + str(i))

            if i+1000 > embeddings.shape[0]:
                distances = self.get_distance_matrix(embeddings[i:], embeddings)
                amount = embeddings.shape[0] - i
            else:
                distances = self.get_distance_matrix(embeddings[i:i+1000], embeddings)
                amount = 1000

            distances = np.reshape(distances, (amount, -1))
            for j in range(i, i+amount):
                graph[j] = [(idx, distances[j-i, idx]) for idx in np.reshape(np.where(distances[j-i, :] <= delta),(-1))]
                neighbors = [points[idx] for idx in np.reshape(np.where(distances[j-i, :] <= delta),(-1))]
                neighbors.append(outliers[j])
                model.addConstr(sum(neighbors) >= 1, "coverage+outliers")

        model.__data = points, outliers
        model.Params.MIPFocus = 1
        model.params.TIME_LIMIT = 180

        return model, graph

    def get_distance_matrix(self, x, y):
        x_input = torch.tensor(x)
        y_input = torch.tensor(y)
        dot = torch.tensordot(x_input, torch.t(y_input), 1)
        x_norm = torch.reshape(torch.sum(torch.pow(x_input, 2), dim=1), (-1, 1))
        y_norm = torch.reshape(torch.sum(torch.pow(y_input, 2), dim=1), (1, -1))
        dist_mat = x_norm + y_norm - 2.0 * dot
        sqrt_dist_mat = torch.sqrt(torch.clamp(dist_mat, 0, 10000)).numpy()
        return sqrt_dist_mat

    def get_graph_min(self, embedding, delta):
        print("Getting Graph Minimum...")
        minimum = 10000
        for i in range(0, embedding.shape[0], 1000):
            print("At Point " + str(i))

            if i + 1000 > embedding.shape[0]:
                distances = self.get_distance_matrix(embedding[i:], embedding)
            else:
                distances = self.get_distance_matrix(embedding[i:i + 1000], embedding)

            distances = np.reshape(distances, (-1))
            distances[distances < delta] = 10000
            minimum = min(minimum, np.min(distances))
            gc.collect()
        return minimum

    def get_graph_max(self, embedding, delta):
        print("Getting Graph Maximum...")
        maximum = 0
        for i in range(0, embedding.shape[0], 1000):
            print("At Point " + str(i))

            if i + 1000 > embedding.shape[0]:
                distances = self.get_distance_matrix(embedding[i:], embedding)
            else:
                distances = self.get_distance_matrix(embedding[i:i + 1000], embedding)

            distances = np.reshape(distances, (-1))
            distances[distances > delta] = 0
            maximum = max(maximum, np.max(distances))
            gc.collect()
        return maximum
