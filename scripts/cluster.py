from toolkit.k_means import KMeans
from toolkit.hac import HAC
from toolkit.arff import Arff
from matplotlib import pyplot as plt
import numpy as np

def setup():
    arff = Arff('datasets/labor.arff',label_count = 1)
    # Trim the id column
    arff = arff.create_subset_arff(col_idx = slice(1,None))
    arff = arff.get_features()
    hac = HAC()
    hac.nominal_indicies = np.where(np.array(arff.attr_types) == 'nominal')[0]
    print('33,44',hac.get_distance(arff.data[33],arff.data[44]))
    print('25,34',hac.get_distance(arff.data[25],arff.data[34]))


def prob0():
    arff = Arff('datasets/labor.arff',label_count = 1)
    # Trim the id column
    arff = arff.create_subset_arff(col_idx = slice(1,None))
    arff = arff.get_features()
    km = KMeans(5)
    km.train(arff, verbose = True,centers = arff.data[:5])

def prob0hac():
    arff = Arff('datasets/labor.arff',label_count = 1)
    # Trim the id column
    arff = arff.create_subset_arff(col_idx = slice(1,None))
    arff = arff.get_features()
    hac = HAC()
    hac.train(arff, verbose = True, printk=[5])

def prob0haccomplete():
    arff = Arff('datasets/labor.arff',label_count = 1)
    # Trim the id column
    arff = arff.create_subset_arff(col_idx = slice(1,None))
    arff = arff.get_features()
    hac = HAC(simple = False)
    hac.train(arff, verbose = True, printk=[5])


def prob1():
    arff = Arff('datasets/sponge.arff')

    km = KMeans(4)
    print(arff.data[:4])
    km.train(arff,verbose=True,centers = arff.data[:4])

def prob1hac():
    arff = Arff('datasets/sponge.arff')

    hac = HAC()
    hac.train(arff,printk=[4])

def prob1haccomplete():
    arff = Arff('datasets/sponge.arff')

    hac = HAC(simple=False)
    hac.train(arff,printk=[4])

def prob2():
    iris = Arff('datasets/iris.arff')
    features = iris.get_features()
    # features.normalize()
    # Train k means for 2-7
    ks= [2,3,4,5,6,7]
    for k in ks:
        km = KMeans(k)
        km.train(features)


    hac2 = HAC(simple = False)
    hac2.train(features,printk=ks)

def prob2wclass():
    iris = Arff('datasets/iris.arff',label_count=0)
    # features.normalize()
    # Train k means for 2-7
    ks= [2,3,4,5,6,7]
    for k in ks:
        km = KMeans(k)
        km.train(iris)


    hac2 = HAC(simple = False)
    hac2.train(iris,printk=ks)

def prob2_v3():
    iris = Arff('datasets/iris.arff',label_count=0)
    for _ in range(5):
        km = KMeans(4)
        km.train(iris)

def plot_prob2():
    kmeans = [152, 78.95, 57.47,70.58,48.07,34.6]
    # hac = [128.14,123.40,122.00,114.20,112.29,111.07]
    hac2 = [171.60,100.95,87.60,82.497,71.47,67.74]
    domain = np.arange(2,8)

    plt.plot(domain, kmeans,label="K Means SSE")
    plt.plot(domain, hac2, label="HAC (Complete Link) SSE")
    plt.title("Sum Squared Error vs # of Clusters")
    plt.xlabel("# of Clusters")
    plt.ylabel("Sum Squared Errors")
    plt.legend()
    plt.show()

def plot_prob2wlabel():
    kmeans = [202.36, 87.33, 70.7,59.67,51.61,58.21]
    hac2 = [150.53,105.64,96.47,82.70,77.60,70.35]
    domain = np.arange(2,8)

    plt.plot(domain, kmeans,label="K Means SSE")
    plt.plot(domain, hac2, label="HAC (Complete Link) SSE")
    plt.title("Sum Squared Error vs # of Clusters")
    plt.xlabel("# of Clusters")
    plt.ylabel("Sum Squared Errors")
    plt.legend()
    plt.show()

def prob3():
    arff = Arff('datasets/abalone.arff',label_count = 0)
    domain = np.arange(2,8)

    ssekmm = []
    for k in domain:
        km = KMeans(k)
        ssek = km.train(arff)
        ssekmm.append(ssek)

    hac = HAC()
    hac2 = HAC(simple=False)
    ssehac =  hac.train(arff, printk=domain)
    ssehac2 = hac2.train(arff,printk=domain)

    plt.plot(domain, ssekmm, label="K-Means SSE")
    plt.plot(domain, ssehac[::-1], label="HAC (Single-Link) SSE")
    plt.plot(domain, ssehac2[::-1], label="HAC (Complete-Link) SSE")
    plt.title("Abalone SSE vs # of Clusters")
    plt.xlabel("# of Clusters")
    plt.ylabel('SSE')
    plt.legend()
    plt.show()

def prob3_normalized():
    arff = Arff('datasets/abalone.arff',label_count = 0)
    arff.normalize()
    domain = np.arange(2,8)

    ssekmm = []
    for k in domain:
        km = KMeans(k)
        ssek = km.train(arff)
        ssekmm.append(ssek)

    hac = HAC()
    hac2 = HAC(simple=False)
    ssehac =  hac.train(arff, printk=domain)
    ssehac2 = hac2.train(arff,printk=domain)

    plt.plot(domain, ssekmm, label="K-Means SSE")
    plt.plot(domain, ssehac[::-1], label="HAC (Single-Link) SSE")
    plt.plot(domain, ssehac2[::-1], label="HAC (Complete-Link) SSE")
    plt.title("Abalone SSE (Normalized) vs # of Clusters")
    plt.xlabel("# of Clusters")
    plt.ylabel('SSE')
    plt.legend()
    plt.show()

def prob4():
    arff = Arff('datasets/abalone.arff',label_count = 0)
    arff.normalize()
    domain = np.arange(2,8)

    ssekmm = []
    for k in domain:
        km = KMeans(k)
        ssek = km.train(arff)
        ssekmm.append(ssek)
        print(km.calc_silhouette_score())

def prob4h():
    arff = Arff('datasets/abalone.arff',label_count = 0)
    arff.normalize()
    domain = np.arange(2,8)
    print('single link --------------------')
    hoc = HAC()
    hoc.train(arff,printk=domain,silhouette=True)
    print('complete link -----------------------')
    hoc = HAC(simple = False)
    hoc.train(arff,printk=domain,silhouette=True)
