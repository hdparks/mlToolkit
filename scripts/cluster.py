from toolkit.k_means import KMeans
from toolkit.hac import HAC
from toolkit.arff import Arff
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
