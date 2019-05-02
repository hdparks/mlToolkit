from toolkit.perceptron_learner import PerceptronLearner
from toolkit.arff import Arff

arff = Arff('datasets/perceptron.arff')
features = arff.get_features()
labels = arff.get_labels()
pl = PerceptronLearner()
pl.train(features, labels)
pl.graph_2D_features(features, labels)
