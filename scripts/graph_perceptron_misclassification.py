from toolkit.perceptron_learner import PerceptronLearner
from toolkit.arff import Arff
import sys
from matplotlib import pyplot as plt
import numpy as np

def main():
    arff = Arff(sys.argv[1])
    pl = PerceptronLearner()
    features = arff.get_features()
    labels = arff.get_labels()

    accuracy_matrix = np.zeros((5,20))

    for i in range(5):

        pl.train(features, labels)

        a = pl.accuracy_tracker[:20]
        # pad to make 20 wide
        a = np.pad(a, (0,20 - len(a)),'constant',constant_values = a[-1] )
        accuracy_matrix[i] = a

    # Average the accuracies of each step
    print(accuracy_matrix)
    avg_accuracy = np.sum(accuracy_matrix, axis=0) / 5
    print(avg_accuracy)

    plt.plot(1 - avg_accuracy)
    plt.xlabel("Epochs")
    plt.ylabel("Avg Misclassification Rate")
    plt.title("Avg Misclassification Rate Over Epochs")

    plt.show()

if __name__ == '__main__':
    main()
