import numpy as np
import matplotlib.pyplot as plt

class K_NN:

    def __init__(self, k=3):
        self.k = k

    def distance(self, x1, x2):
        return np.dot(x1,x2)/(np.linalg.norm(x1) * np.linalg.norm(x2))
    
    def fit(self, x_dataset, y_dataset):
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.num_users, self.num_items = x_dataset.shape

    def predict(self, x_test):
        distances = [self.distance(x_test,self.x_dataset[:,i]) for i in range(len(self.y_dataset))]
        distances = np.array(distances)
        k_index = np.argsort(distances)[:self.k]
        k_nearest_label = [self.y_dataset[i] for i in k_index]
        classes = np.unique(k_nearest_label)
        class_common = ""
        n_common = 0
        for i in classes:
            n_class = np.count_nonzero(k_nearest_label == i)
            if n_class > n_common:
                class_common = i
                n_common = n_class
            elif n_class == n_common:
                n_dis_class = distances[k_index[k_nearest_label == i]]
                n_dis_common = distances[k_index[k_nearest_label == class_common]]
                sm_class = np.sum(n_dis_class)
                sm_common = np.sum(n_dis_common)
                if sm_class < sm_common:
                    class_common = i
                    n_common = n_class
        return class_common