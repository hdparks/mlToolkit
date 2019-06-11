from toolkit.arff import Arff
import numpy as np
from scipy import stats

class KMeans():
    DEFAULT_MAX_ITER = 100
    DEFAULT_TOL = .1

    def __init__(self, k, MAX_ITER=DEFAULT_MAX_ITER, tol = DEFAULT_TOL):
        self.k = k
        self.MAX_ITER = MAX_ITER
        self.tol = tol

    def train(self, dataset, verbose = False):
        """ Trains until k clusters have stabilized """

        # INITIALIZATION
        # choose k cluster centers randomly
        n = len(dataset.data[0])
        self.nominal_indicies = np.where(np.array(dataset.attr_types) == 'nominal')[0]

        # TODO: FIGURE THIS !@#$ OUT
        self.centers = np.random.random((self.k,n)) * [np.max(dataset.data[:,i]) - np.min(dataset.data[:,i]) for i in range(len)] - [np.min(dataset.data[:,i]) for i in range(len)]

        # LEARNING
        for iter in range(self.MAX_ITER):
            clusters = []
            i_clusters = []
            for i in range(self.k):
                clusters.append(list())
                i_clusters.append(list())
            # Assign each datapoint to a cluster
            for i, point in enumerate(dataset.data):
                # Compute distance (squared) to each center
                distances = []
                for center in self.centers:
                    distances.append(self.get_distance(point, center))

                fav_point = np.argmin(distances)
                clusters[fav_point].append(point)
                i_clusters[fav_point].append(i)

            # Move cluster center to the mean average
            new_centers = np.array([self.calc_mean_av(cluster) for cluster in clusters])

            # Check for convergence
            delta = np.linalg.norm(new_centers - self.centers)
            if delta <= self.tol:
                self.centers = new_centers
                break

            self.centers = new_centers
        # DEBUG:
        print(f"Converged in {iter} iterations")

        if verbose:
            print('k',self.k)
            print('centroids:',self.centers)
            print('instances in centroid',i_clusters)
            print([self.calc_sse(cluster, centroid) for cluster, centroid in zip(clusters,self.centers)])
            print([self.calc_sse(dataset.data, self.calc_mean_av(dataset.data))])



    def get_distance(self, point_a, point_b):
        d = 0
        for i,(a,b) in enumerate(zip(point_a, point_b)):
            if a == np.nan or b == np.nan:
                # Unknown values immediately have distance 1
                d += 1
            elif i in self.nominal_indicies:
                # Nominal distance is 0 if same, 1 if different
                d += 0 if a == b else 1
            else:
                d += (a-b)**2

        return d

    def calc_mean_av(self, cluster):
        cluster = np.array(cluster)
        mean = []
        for i in range(len(cluster[0])):
            if i in self.nominal_indicies:
                # If nominal, just add the most common one
                mean.append(stats.mode(cluster[:,i])[0][0])
            else:
                mean.append(np.sum(cluster[:,i]) / len(cluster))

        return mean

    def calc_sse(self, cluster, centroid):
        sse = 0
        for point in cluster:
            sse += self.get_distance(point, centroid)
        return sse
