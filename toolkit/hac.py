import numpy as np
from scipy import stats
from toolkit.arff import Arff

class HAC():
    def __init__(self, simple=True):
        self.simple = simple

    def train(self, dataset, verbose = False, printk = [], silhouette = False):
        output = []
        # INITIALIZATION
        self.nominal_indicies = np.where(np.array(dataset.attr_types) == 'nominal')[0]
        self.data = dataset.data
        self.instance_count = dataset.instance_count
        self.n = self.data.shape[1]

        # Calculate the distance from every node to every other node (O(n^2))
        self.distances = np.zeros((self.instance_count,self.instance_count))
        for i in range(self.instance_count):
            for j in range(i+1,self.instance_count):
                self.distances[i,j] = self.get_distance(self.data[i], self.data[j])

        # Make symmetrical
        self.distances += self.distances.T

        # Turn every point into its own cluster
        self.clusters = []
        for i in range(len(dataset.data)):
            self.clusters.append([i])

        self.history = [self.clusters]

        iter = 1
        # TRAINING
        while(len(self.clusters) > 1):

            print('iteration',iter)
            iter += 1
            # Perform hierarchal agglomeration
            j,i,d = self.find_nearest_cluster()

            # Merge clusters
            cluster_j = self.clusters.pop(j)
            self.clusters[i] = self.clusters[i] + cluster_j

            if verbose:
                print("Merging clusters", i,'and',j, 'dist:',d)
                for i, cl in enumerate(self.clusters):
                    print(i, cl)


            if len(self.clusters) in printk:
                print('clusters:',len(self.clusters))
                centroids = [self.calculate_centroid(cluster) for cluster in self.clusters]
                print('centroids:')
                for i, c in enumerate(centroids):
                    print(len(self.clusters[i]),c)

                sses = [self.calculate_sse(cl,ce) for cl, ce in zip(self.clusters, centroids)]
                print('sse:')
                for s in sses:
                    print(s)

                print("total sse:", sum(sses))
                output.append(sum(sses))
                if silhouette:
                    print(self.calc_silhouette_score())

        return output


    def find_nearest_cluster(self):
        """ Returns indicies of nearest cluster according to distance rule"""
        cluster_distances = np.ones((len(self.clusters),len(self.clusters)))
        for i, cluster_a in enumerate(self.clusters):
            for j, cluster_b in enumerate(self.clusters[:i]):

                point_distances = np.ones((len(cluster_a),len(cluster_b)))
                for a, point_a in enumerate(cluster_a):
                    for b, point_b in enumerate(cluster_b):
                        # print(self.distances[point_a,point_b])
                        point_distances[a,b] = self.distances[point_a,point_b]
                        # print(point_distances[a,b])

                # Get the optimal distance according to distance rule
                if self.simple:
                    cluster_distances[i,j] = np.min(point_distances)
                else:
                    cluster_distances[i,j] = np.max(point_distances)

        # Find the least distance between two clusters
        cluster_distances += np.triu(np.ones((len(self.clusters),len(self.clusters))) * np.inf)
        x,y = np.unravel_index(cluster_distances.argmin(),cluster_distances.shape)
        return x,y, cluster_distances[x,y]



    def calculate_centroid(self, cluster):
        points = self.data[cluster]
        centroid = np.zeros(self.n)

        for i in range(self.n):
            # Nan_filter
            nan_filter = np.isnan(points[:,i])
            if np.all(nan_filter):
                centroid[i] = points[0,i]
                continue

            if i in self.nominal_indicies:
                centroid[i] = stats.mode(points[:,i][~nan_filter])[0][0]
            else:
                centroid[i] = sum(points[:,i][~nan_filter])/len(points[:,i][~nan_filter])

        return centroid

    def get_distance(self, point_a, point_b):
        d = 0
        for i in range(len(point_a)):
            if np.isnan(float(point_a[i])) or np.isnan(float(point_b[i])):
                d += 1
            elif i in self.nominal_indicies:
                d += 0 if point_a[i] == point_b[i] else 1
            else:
                d += (point_a[i] - point_b[i])**2
        return np.sqrt(d)

    def calculate_sse(self, cluster, centroid):
        points = self.data[cluster]

        sse = 0
        for point in points:
            sse += self.get_distance(point,centroid)

        return sse

    def calc_silhouette_score(self):
        print('calculating silhouette scores')
        # Get clusters
        silhouette_index = np.zeros(self.instance_count)
        for i, point in enumerate(self.data):
            cluster_scores = np.zeros(len(self.clusters))
            my_cluster = None
            for cl, cluster in enumerate(self.clusters):
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
                b = min(cluster_scores[np.arange(len(self.clusters)) != my_cluster])
                silhouette_index[i] = (b - a)/max(a,b)

        return sum(silhouette_index)/len(silhouette_index)
