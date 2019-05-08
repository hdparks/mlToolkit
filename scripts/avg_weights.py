from toolkit.perceptron_learner import PerceptronLearner
from toolkit.arff import Arff
import sys
import numpy as np

def rnd4(obj):
    if isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, (int, float, complex)):
        return "{:.4f}".format(obj)

arff = Arff(sys.argv[1])
features = arff.get_features()
labels = arff.get_labels()

pl = PerceptronLearner()

weights = []

for i in range(10):
    pl.train(features, labels)
    weights.append(pl.weights)

avg_weights = np.sum(weights, axis=0) / 10
names = arff.get_attr_names()
for i in range(len(avg_weights)):
    print(rnd4(avg_weights[i]), names[i])
