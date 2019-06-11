from toolkit.k_means import KMeans
from toolkit.arff import Arff

def prob1():
    arff = Arff('datasets/sponge.arff')
    arff.normalize()

    km = KMeans(4)
    km.train(arff,verbose=True)
