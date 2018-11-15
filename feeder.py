import numpy as np
import mnist


class feeder_mnist:
    def __init__(self):
        self.train_images = mnist.train_images()
        self.train_labels = mnist.train_labels()
        self.test_images = mnist.test_images()
        self.test_labels = mnist.test_labels()

    def train_data(self, n = None, onehot = True, index = None, seed = 31):
        np.random.seed(seed)
        if index is None:
            order = np.arange(len(self.train_images))
            np.random.shuffle(order)
        else:
            order = index
        if n is not None:
            order = order[:n]
            np.random.shuffle(order)

        X = self.train_images[order, :, :]
        Y = self.train_labels[order]
        if onehot:
            label = np.zeros((len(Y),10))
            for i in range(len(Y)):
                label[i, Y[i]] = 1
            Y = label
        return {'X':X,'Y':Y}

    def test_data(self, n=None, onehot=True):

        order = np.arange(len(self.test_images))
        #np.random.shuffle(order)
        if n is not None:
            order = order[:n]
        X = self.test_images[order, :, :]
        Y = self.test_labels[order]
        if onehot:
            label = np.zeros((len(Y), 10))
            for i in range(len(Y)):
                label[i, Y[i]] = 1
            Y = label
        return {'X': X, 'Y': Y}

    def class_balanced_index(self):
        res = []
        for i in range(10):
            res.append(np.array(np.where(self.train_labels == i)))
        shortest_len = min([x.shape[1] for x in res])
        res = np.concatenate([x[:, :shortest_len] for x in res], axis=0)
        res = np.reshape(res.T, -1)
        return res





