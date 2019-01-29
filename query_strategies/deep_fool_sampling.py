import torch
import numpy as np
from strategy import Strategy

# TODO Refactor and comment


class DeepFoolSampling(Strategy):
    def query(self, n):
        unlabeled_indices = np.arange(self.pool_size)[~self.labeled_indices]
        self.classifier.cpu()
        self.classifier.eval()
        distances = []

        data_pool = self.data_handler(self.x[unlabeled_indices], self.y[unlabeled_indices], False)
        for i in range(len(data_pool)):
            if i % 1000 == 0:
                print("adv {}/{}".format(i, len(data_pool)))
            x, y, index = data_pool[i]
            distances.append(self.calculate_distanced(x))
        print("adv {0}/{0}".format(len(data_pool)))

        regions = []
        for i in self.y[unlabeled_indices]:
            regions.append(len(i))

        probabilities = torch.as_tensor(distances)

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

    def old_query(self, n):
        unlabeled_indices = np.arange(self.pool_size)[~self.labeled_indices]
        self.classifier.cpu()
        self.classifier.eval()
        distance = np.zeros(unlabeled_indices.shape)

        data_pool = self.data_handler(self.x[unlabeled_indices], self.y[unlabeled_indices])
        for i in range(len(unlabeled_indices)):
            if i % 100 == 0:
                print("adv {}/{}".format(i, len(unlabeled_indices)))
            x, y, index = data_pool[i]
            distance[i] = self.calculate_distanced(x)

        regions = []
        for i in self.y[unlabeled_indices]:
            regions.append(len(i))

        regions_uncertainities = []

        count = 0
        for region in regions:
            region_uncertainities = distance[count:count + region]
            count = region
            # regions_uncertainities.append(max(region_uncertainities))
            regions_uncertainities.append(np.average(region_uncertainities))

        regions_uncertainities = torch.from_numpy(np.array(regions_uncertainities))

        return unlabeled_indices[regions_uncertainities.argsort()[:n]]

    def calculate_distanced(self, x):
        unsqueezed = torch.unsqueeze(x, 0)
        unsqueezed.requires_grad_()
        eta = torch.zeros(unsqueezed.shape)

        out = self.classifier(unsqueezed + eta)
        num_classes = out.shape[1]
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()

        i_iter = 0

        while py == ny and i_iter < 50:
            out[0, py].backward(retain_graph=True)
            grad_np = unsqueezed.grad.data.clone()
            value_1 = np.inf
            ri = None

            for i in range(num_classes):
                if i == py:
                    continue

                unsqueezed.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = unsqueezed.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())

                if value_i < value_1:
                    ri = value_i / np.linalg.norm(wi.numpy().flatten()) * wi

            eta += ri.clone()
            unsqueezed.grad.data.zero_()
            out = self.classifier(unsqueezed + eta)
            py = out.max(1)[1].item()
            i_iter += 1

        return (eta * eta).sum()
