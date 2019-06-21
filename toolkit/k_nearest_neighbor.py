import numpy as np


class KNN():
    def __init__(self, k, weighting = False, vdm = False):
        self.k = k
        self.w = weighting
        self.vdm = vdm

    def knn(self, features, labels, inputs):
        self.features = features
        self.labels = labels
        self.inputs = inputs
        self.nominal_indicies = np.where(features.attr_types == 'nominal')[0]

        if self.vdm:
            self.build_vdmd()

        for i in self.nominal_indicies:
            # Assign distances between discrete inputs
            data = features.data[:,i]



        # Find the k nearest neighbors
        closest_class = np.zeros(inputs.instance_count)

        for n, input in enumerate(inputs):
            print(n)
            indicies, neighbor_distances = self.closest_neighbor_indicies(features.data, input.data)

            # Get a list of classes within the closest neighbors
            neighbor_classes = labels.data[indicies]
            classes = np.unique(neighbor_classes)

            if len(classes) == 1:
                closest_class[n] = classes[0]
            else:

                if self.w:
                    weights = 1 / neighbor_distances
                    weights = weights / np.linalg.norm(weights)
                    # Normalize the weights

                    counts = [0]*(int(max(classes))+1)
                    for i in range(len(neighbor_classes)):
                        counts[int(neighbor_classes[i])] += weights[i]

                else:
                    counts = [np.sum( np.where(labels.data[indicies[:self.k]] == c )[0],axis=0) for c in classes]

                closest_class[n] = classes[np.argmax(counts)]

        return closest_class

    def knn_regression(self, features, labels, inputs):

        predictions = []

        for input in inputs.data:
            # Find k nearest neighbors
            indicies, neighbor_distances = self.closest_neighbor_indicies(features.data, input)
            # Get the mean label position of the closest neighbors
            neighbor_positions = np.ravel(labels.data[indicies])

            if self.w:
                weights = 1 / neighbor_distances
                # Normalize the weights
                weights = weights / np.linalg.norm(weights)
                mean_position = np.sum(neighbor_positions * np.array(weights)) / len(neighbor_positions)
            else:
                mean_position = np.sum(neighbor_positions,axis=0) / len(indicies)
            predictions.append(mean_position)


        return predictions

    def closest_neighbor_indicies(self,feature_data, input_data):

        distances = np.array([self.get_distance(f, input_data) for f in feature_data])
        indicies = np.argsort(distances, axis= 0)[:self.k]

        return indicies, distances[indicies]

    def get_distance(self, data, input):
        # Handles nominal data, unknown data
        d = 0
        for i in range(len(data)):
            if i in self.nominal_indicies:
                if self.vdm:
                    vdm_idx = self.nominal_indicies.index(i)
                    d += self.vdmd[vdm_idx][data[i],input[i]]
                else:
                    d += 1 if data[i] != input[i] else 0
            else:
                # If any data is unknown, the distance is always set to 1
                # This is extremely conservative, but it will keep uknown data from messing with accuracy
                if data[i] == np.nan or input[i] == np.nan:
                    d += 1
                else:
                    d += (data[i] - input[i])**2
        return d

    def build_vdmd(self):
        self.vdmd = []

        # Build c indicies
        self.c_indicies = []
        for c in range(len(self.labels.enum_to_str[0])):
            self.c_indicies.append(np.where(self.labels.data == c)[0])


        for nom in self.nominal_indicies:
            self.vdmd.append(self.build_vdma(nom))


    def build_vdma(self, col_idx):
        n = len(self.features.enum_to_str[col_idx])
        vdma = np.zeros((n,n))

        for i in range(n):
            for j in range(i+1,n):
                n_i = np.count_nonzero(self.features.data[col_idx] == i)
                n_j = np.count_nonzero(self.features.data[col_idx] == j)

                diff = 0
                for c_ind in self.c_indicies:
                    n_i_c = np.count_nonzero(self.features.data[col_idx][c_ind] == i)
                    n_j_c = np.count_nonzero(self.features.data[col_idx][c_ind] == j)

                    diff += abs((n_i_c / n_i) - (n_j_c / n_j))

                vdma[i,j] = diff

        # Symmetricize it
        vdma += vdma.T

        return vdma
