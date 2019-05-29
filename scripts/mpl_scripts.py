from toolkit.mlp_learner import MultilayerPerceptronLearner
from toolkit.arff import Arff
from matplotlib import pyplot as plt
import numpy as np

def prob_2():
    # Get arff from iris database
    iris = Arff('datasets/iris.arff')

    # Get a random 75/25 split
    iris.shuffle()

    validation = iris.create_subset_arff(slice(iris.instance_count//4))
    training = iris.create_subset_arff(slice(iris.instance_count//4,None))

    # Make the learner
    # shape = [4 8 3]
    learner = MultilayerPerceptronLearner([4,8,3],momentum = 0)
    learner.set_validation(validation.get_features().data, validation.get_labels().data)
    learner.set_training(training.get_features(), training.get_labels())

    training_mse = []
    validation_mse = []
    classification_accuracy = []

    for epoch in range(learner.max_epoch):
        learner.train_one_epoch()
        training_mse.append(learner.get_mse())
        validation_mse.append(learner.get_mse(True))
        classification_accuracy.append(learner.get_accuracy(True))
        if learner.check_for_convergence():
            break

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(training_mse, label="Training Data MSE")
    ax1.plot(validation_mse, label ="Validation Data MSE")
    ax1.legend()
    ax2.plot(classification_accuracy, label="Classification Accuracy")
    plt.legend()
    plt.show()

def prob_3():
    # read in vowels dataset
    arff = Arff('datasets/vowels.arff')

    # Leave out the test/train and person features, which are unceccessary.
    arff = arff.create_subset_arff(col_idx= slice(2,None), label_count = 1)

    # Get a 75/25 split
    arff.shuffle()
    training = arff.create_subset_arff(slice(arff.instance_count//4))
    test = arff.create_subset_arff(slice(arff.instance_count//4,-1))

    # Get a 15% Validation set
    validation = training.create_subset_arff(slice(arff.instance_count//5))
    training = training.create_subset_arff(slice(arff.instance_count//5,None))

    
    learner = MultilayerPerceptronLearner([training.features_count,2 * training.features_count,11])
    learner.set_training(training.get_features(), training.get_labels())
    learner.set_validation(validation.get_features().data,validation.get_labels().data)

    learner.train()
    print(learner.get_accuracy(True))
