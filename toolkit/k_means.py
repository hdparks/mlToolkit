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

        # Tracking varibales
        self.centroids = []
        self.clusters = []
        self.sse_cluster = []
        self.sse_total = []

    def train(self, dataset, verbose = False, centers = None):
        """ Trains until k clusters have stabilized """
        self.data = dataset.data
        self.instance_count = dataset.instance_count

        # INITIALIZATION
        n = len(dataset.data[0])
        self.nominal_indicies = np.where(np.array(dataset.attr_types) == 'nominal')[0]


        # Caclulate distances
        self.distances = np.zeros((dataset.instance_count, dataset.instance_count))
        for i, point_a in enumerate(self.data):
            for j, point_b in enumerate(self.data[:i]):
                self.distances[i,j] = self.get_distance(point_a, point_b)
        # Make it symmetrical
        self.distances += self.distances.T
        self.distances = np.array(self.distances)
        # choose k cluster centers as random instances
        if type(centers) != type(None):
            self.centers = centers
        else:
            center_ind = np.random.choice(dataset.instance_count, self.k, replace=False)
            self.centers = dataset.data[center_ind]

        self.centroids.append(self.centers)


        # LEARNING
        for iter in range(self.MAX_ITER):
            if verbose:
                print("iteration",iter)

            # Keep track of clusters, as well as indicies of cluster elements
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

            self.clusters.append(i_clusters)

            # Move cluster center to the mean average
            new_centers = np.array([self.calc_mean_av(cluster) for cluster in clusters])

            # Tracking data
            self.centroids.append(new_centers)
            sse = [self.calc_sse(cluster, centroid) for cluster, centroid in zip(clusters,new_centers)]
            self.sse_cluster.append(sse)
            self.sse_total.append(sum(sse))

            # Check for convergence
            if np.all(new_centers == self.centers):
                self.centers = new_centers
                break

            self.centers = new_centers



        print(f"Converged in {iter} iterations")

        if verbose:
            print('k',self.k)
            print('initial centroids')
            print(self.centroids[0])

            for i in range(iter + 1):
                print('Iteration', i)
                print('centroids:')
                for c in self.centroids[i+1]:
                    print(c)

                print('instances in centroid:')
                for idx in self.clusters[i]:
                    print(len(idx),idx)

                print('cluster sse')
                for s in self.sse_cluster[i]:
                    print(s)

                print('total cluster sse:')
                print(self.sse_total[i])
        else:
            print('k',self.k)
            print('centroids')
            print(self.centroids[-1])
            print('cluster sse')
            for i in self.sse_cluster[-1]:
                print(i)
            print('total cluster')
            print(self.sse_total[-1])
        return self.sse_total[-1]


    def get_distance(self, point_a, point_center):
        d = 0
        for i,(a,b) in enumerate(zip(point_a, point_center)):
            if np.isnan(a) or np.isnan(b):
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
        mean = np.zeros(cluster.shape[1])
        for i in range(len(mean)):
            if i in self.nominal_indicies:
                # If nominal, just add the most common one
                nan_fliter = np.isnan(cluster[:,i])
                # If all nan, use nan
                if np.all(nan_fliter):
                    mean[i] = cluster[0,i]
                else:
                    mode = stats.mode(cluster[:,i][~nan_fliter])[0][0]
                    mean[i]= mode
            else:
                nan_fliter = np.isnan(cluster[:,i])
                # If all nan, use nan
                if np.all(nan_fliter):
                    mean[i] = cluster[0,i]

                else:
                    sum = np.sum(cluster[:,i][~nan_fliter]) / len(cluster[~nan_fliter])
                    mean[i] = sum

        return mean

    def calc_sse(self, cluster, centroid):
        sse = 0
        for point in cluster:
            sse += self.get_distance(point, centroid)
        return sse

    def get_attributes(self,index):
        return list(self.dataset.enum_to_str[index].keys())

    def calc_silhouette_score(self):
        print('calculating silhouette scores')
        # Get clusters
        silhouette_index = np.zeros(self.instance_count)
        for i, point in enumerate(self.data):
            cluster_scores = np.zeros(len(self.clusters[-1]))
            my_cluster = None
            for cl, cluster in enumerate(self.clusters[-1]):
                if i in cluster:
                    sd = self.distances[cluster,i]
                    cluster_scores[cl] = sum(sd) / (len(cluster) - 1)
                    my_cluster = cl
                else:
                    sd = sum(self.distances[cluster,i])
                    cluster_scores[cl] = sd / len(cluster)
            # Take min of cluster score that isnt current cluster
            a = cluster_scores[my_cluster]
            if a == 0:
                silhouette_index[i] = 0
            else:
                b = min(cluster_scores[np.arange(self.k) != my_cluster])
                silhouette_index[i] = (b - a)/max(a,b)

        return sum(silhouette_index)/len(silhouette_index)
