import os
import cv2
import numpy as np
from PIL import Image
import scipy.io as io
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset
from imgaug import augmenters as iaa


def get_dataset(dataset_dir):
    """
    Function for loading the dataset and returning four arrays containing:
    training data, training labels, testing data and testing labels.
    This function loads the CRCHistoPhenotypes Labeled Cell Nuclei Dataset.
    The final 20 images are used as a testing dataset.
    :return: Four Arrays containing training data, training labels, testing data and testing labels.
    """

    x_train, x_test = [], []
    y_train, y_test = [], []

    for i in range(100):
        if os.path.isdir(os.path.join(dataset_dir, "img{}".format(i + 1))):
            epithelial = io.loadmat(os.path.join(dataset_dir, "img{0}/img{0}_epithelial.mat".format(i + 1)))["detection"]
            fibroblast = io.loadmat(os.path.join(dataset_dir, "img{0}/img{0}_fibroblast.mat".format(i + 1)))["detection"]
            inflammatory = io.loadmat(os.path.join(dataset_dir, "img{0}/img{0}_inflammatory.mat".format(i + 1)))["detection"]
            others = io.loadmat(os.path.join(dataset_dir, "img{0}/img{0}_others.mat".format(i + 1)))["detection"]
            image = cv2.imread(os.path.join(dataset_dir, "img{0}/img{0}.bmp".format(i + 1)))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.copyMakeBorder(image, 15, 15, 15, 15, cv2.BORDER_WRAP)
            image = np.asarray(image)

            for pi in range(0, 500, 100):
                for pj in range(0, 500, 100):
                    temp_x, temp_y = [], []
                    for epi in epithelial:
                        if pi <= epi[0] < pi + 100 and pj <= epi[1] < pj + 100:
                            x, y = int(epi[0]) + 15, int(epi[1]) + 15
                            temp_x.append(image[y - 15:y + 15, x - 15:x + 15])
                            temp_y.append([1., 0., 0., 0.])

                    for fib in fibroblast:
                        if pi <= fib[0] < pi + 100 and pj <= fib[1] < pj + 100:
                            x, y = int(fib[0]) + 15, int(fib[1]) + 15
                            temp_x.append(image[y - 15:y + 15, x - 15:x + 15])
                            temp_y.append([0., 1., 0., 0.])

                    for inf in inflammatory:
                        if pi <= inf[0] < pi + 100 and pj <= inf[1] < pj + 100:
                            x, y = int(inf[0]) + 15, int(inf[1]) + 15
                            temp_x.append(image[y - 15:y + 15, x - 15:x + 15])
                            temp_y.append([0., 0., 1., 0.])

                    for oth in others:
                        if pi <= oth[0] < pi + 100 and pj <= oth[1] < pj + 100:
                            x, y = int(oth[0]) + 15, int(oth[1]) + 15
                            temp_x.append(image[y - 15:y + 15, x - 15:x + 15])
                            temp_y.append([0., 0., 0., 1.])

                    # for thing in temp_x:
                    #     cv2.namedWindow("image")
                    #     cv2.imshow("image", thing)
                    #     cv2.waitKey(0)

                    if temp_x != []:
                        if i < 80:
                            x_train.append(temp_x)
                            y_train.append(temp_y)
                        else:
                            x_test.append(temp_x)
                            y_test.append(temp_y)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def transform():
    """
    Function that will be used within the data handler to transform the input images.
    This can be modified to adjust the transforms the images to the specific dataset.
    If no transforms is required this function can return None.
    :return: A transform function.
    """
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])


class DataHandler(Dataset):
    """
    Class for handling the dataset when being used to train a neural network.
    """

    def __init__(self, x, y, augmentation=True):
        """
        Initilisation method for the data handler class that sets the initial parameters.
        :param x: An array of data.
        :param y: An array of labels.
        """

        self.x = []
        self.y = []
        for i in range(len(x)):
            for j in range(len(x[i])):
                self.x.append(x[i][j])
                self.y.append(y[i][j])
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.transform = transform()

        self.weights = []
        counter = Counter(np.argmax(self.y, 1))
        max_val = float(max(counter.values()))
        for i in range(4):
            self.weights.append(max_val / counter[i])

        if augmentation:
            a = iaa.Sequential([iaa.Fliplr(1.0)]).augment_images(self.x)
            b = iaa.Sequential([iaa.Flipud(1.0)]).augment_images(self.x)
            c = iaa.Sequential([iaa.GaussianBlur(1.0)]).augment_images(self.x)
            d = iaa.Sequential([iaa.ChannelShuffle(1.0)]).augment_images(self.x)
            e = iaa.Sequential([iaa.Sharpen(1.0)]).augment_images(self.x)

            self.x = np.concatenate([self.x, a, b, c, d, e])
            self.y = np.concatenate([self.y, self.y, self.y, self.y, self.y, self.y])

    def __getitem__(self, index):
        """
        Method to get a single item from the dataset from a specific index.
        :param index: The location in the array where the data will be taken from.
        :return: A single item of data, the label for the data and the index where it was extracted from.
        """

        x, y = self.x[index], self.y[index]
        if transforms is not None:
            x = Image.fromarray(x, mode='RGB')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        """
        Method for returning the size of dataset handled by the data handler.
        :return: An integer containing the length of the dataset.
        """

        return len(self.x)


