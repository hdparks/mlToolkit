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
    v_features = validation.get_features()
    v_labels = validation.get_labels()

    training = iris.create_subset_arff(slice(iris.instance_count//4,None))
    features = training.get_features()
    labels = training.get_labels()
    # Make the learner
    # shape = [4 8 3]
    learner = MultilayerPerceptronLearner([4,8,3],momentum = 0)

    training_mse = []
    validation_mse = []
    classification_accuracy = []

    for epoch in range(learner.max_epoch):
        learner.train_one_epoch(features, labels)
        training_mse.append(learner.get_mse(features, labels))
        validation_mse.append(learner.get_mse(v_features, v_labels))
        classification_accuracy.append(learner.get_accuracy(v_features, v_labels))
        if learner.check_for_convergence(v_features, v_labels):
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
    t_features = test.get_features()
    t_labels = test.get_labels()

    # Get a 15% Validation set
    validation = training.create_subset_arff(slice(arff.instance_count//5))
    training = training.create_subset_arff(slice(arff.instance_count//5,None))
    v_features = validation.get_features()
    v_labels = validation.get_labels()


    training_mse = []
    validation_mse = []
    test_mse = []
    epochs = []
    domain = np.logspace(-3,0)
    for lr in domain:
        mse = 0
        vmse = 0
        tmse = 0
        e = 0
        for _ in range(3):
            learner = MultilayerPerceptronLearner([training.features_count,2 * training.features_count,11],momentum = 0)
            learner.zero_weights()
            learner.lr = lr
            learner.max_epoch = 500
            learner.train(training.get_features(), training.get_labels(), validation.get_features(), validation.get_labels())
            e += learner.epochs
            tmse += learner.get_mse(test.get_features(),test.get_labels())
            mse += learner.get_mse(training.get_features(), training.get_labels())
            vmse += learner.get_mse(validation.get_features(), validation.get_labels())
        epochs.append(e/3)
        training_mse.append(mse/3)
        validation_mse.append(vmse/3)
        test_mse.append(tmse/3)


    plt.semilogx(domain, test_mse,label="Test Set MSE")
    plt.semilogx(domain, training_mse,label="Training Set MSE")
    plt.semilogx(domain, validation_mse, label="Validation Set MSE")
    plt.title("MSE vs Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.show()

    plt.semilogx(domain, epochs)
    plt.title("Number of Training Epochs vs Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Number of Training Epochs")
    plt.show()

def prob_4(lr):
    # read in vowels dataset
    arff = Arff('datasets/vowels.arff')

    # Leave out the test/train and person features, which are unceccessary.
    arff = arff.create_subset_arff(col_idx= slice(2,None), label_count = 1)

    # Get a 75/25 split
    arff.shuffle()
    training = arff.create_subset_arff(slice(arff.instance_count//4))
    test = arff.create_subset_arff(slice(arff.instance_count//4,-1))
    t_features = test.get_features()
    t_labels = test.get_labels()

    # Get a 15% Validation set
    validation = training.create_subset_arff(slice(arff.instance_count//5))
    training = training.create_subset_arff(slice(arff.instance_count//5,None))
    v_features = validation.get_features()
    v_labels = validation.get_labels()

    domain = 2 ** np.arange(0,8)
    training_mse = []
    validation_mse = []
    test_mse = []
    for nodes in domain:
        mse = 0
        vmse = 0
        tmse = 0
        for _ in range(100):
            learner = MultilayerPerceptronLearner([training.features_count,nodes,11], momentum=0)
            # learner.zero_weights()
            learner.lr = lr
            learner.max_epoch = 500
            learner.train(training.get_features(), training.get_labels(), validation.get_features(), validation.get_labels())

            tmse += learner.get_mse(test.get_features(),test.get_labels())
            mse += learner.get_mse(training.get_features(), training.get_labels())
            vmse += learner.get_mse(validation.get_features(), validation.get_labels())

        training_mse.append(mse/100)
        validation_mse.append(vmse/100)
        test_mse.append(tmse/100)


    plt.semilogx(domain, test_mse,basex=2,label="Test Set MSE")
    plt.semilogx(domain, training_mse,basex=2,label="Training Set MSE")
    plt.semilogx(domain, validation_mse,basex=2, label="Validation Set MSE")
    plt.title("MSE vs Number of Hidden Nodes")
    plt.xlabel("Number of Hidden Nodes")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.show()

def prob_5(lr, hidden):
    # read in vowels dataset
    arff = Arff('datasets/vowels.arff')

    # Leave out the test/train and person features, which are unceccessary.
    arff = arff.create_subset_arff(col_idx= slice(2,None), label_count = 1)

    # Get a 75/25 split
    arff.shuffle()
    training = arff.create_subset_arff(slice(arff.instance_count//4))
    test = arff.create_subset_arff(slice(arff.instance_count//4,-1))
    t_features = test.get_features()
    t_labels = test.get_labels()

    # Get a 15% Validation set
    validation = training.create_subset_arff(slice(arff.instance_count//5))
    training = training.create_subset_arff(slice(arff.instance_count//5,None))
    v_features = validation.get_features()
    v_labels = validation.get_labels()


    epochs = []
    accuracy = []
    domain = np.linspace(0,1,20)
    for momentum in domain:
        e = 0
        acc = 0
        for _ in range(10):
            learner = MultilayerPerceptronLearner([training.features_count,hidden,11], momentum=momentum)
            # learner.zero_weights()
            learner.lr = lr
            learner.max_epoch = 500
            learner.train(training.get_features(), training.get_labels(), validation.get_features(), validation.get_labels())
            acc += learner.get_accuracy(t_features, t_labels)
            e += learner.epochs
        epochs.append(e/10)
        accuracy.append(acc/10)

    print(accuracy)
    plt.plot(domain, epochs)
    plt.title("Number of Training Epochs vs Momentum")
    plt.xlabel("Momentum Constant")
    plt.ylabel("Number of Training Epochs")
    plt.show()


def prob_6():
    # read in vowels dataset
    arff = Arff('datasets/vowels.arff')

    # Leave out the test/train and person features, which are unceccessary.
    arff = arff.create_subset_arff(col_idx= slice(2,None), label_count = 1)

    # Get a 75/25 split
    arff.shuffle()
    training = arff.create_subset_arff(slice(arff.instance_count//4))
    test = arff.create_subset_arff(slice(arff.instance_count//4,-1))
    t_features = test.get_features()
    t_labels = test.get_labels()

    # Get a 15% Validation set
    validation = training.create_subset_arff(slice(arff.instance_count//5))
    training = training.create_subset_arff(slice(arff.instance_count//5,None))
    v_features = validation.get_features()
    v_labels = validation.get_labels()
    features = training.get_features()
    labels = training.get_labels()

    taccuracy = []
    accuracy = []
    domain = np.arange(1,10)
    for i in domain:
        tacc = 0
        acc = 0
        for _ in range(10):
            learner = MultilayerPerceptronLearner([training.features_count]+[32]*i+[11], momentum=.85)
            # learner.zero_weights()
            learner.lr = .1
            learner.max_epoch = 500
            learner.train(training.get_features(), training.get_labels(), validation.get_features(), validation.get_labels())
            tacc += learner.get_accuracy(t_features, t_labels)
            acc += learner.get_accuracy(features,labels)
        accuracy.append(acc/10)
        taccuracy.append(tacc/10)

    plt.plot(domain, accuracy,label="Training Set Accuracy")
    plt.plot(domain, taccuracy, label="Test Set Accuracy")
    plt.title("Number of Hidden Layers vs Accuracy")
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def prob_6_b():
    # read in vowels dataset
    arff = Arff('datasets/vowels.arff')

    # Leave out the test/train and person features, which are unceccessary.
    arff = arff.create_subset_arff(col_idx= slice(2,None), label_count = 1)

    # Get a 75/25 split
    arff.shuffle()
    training = arff.create_subset_arff(slice(arff.instance_count//4))
    test = arff.create_subset_arff(slice(arff.instance_count//4,-1))
    t_features = test.get_features()
    t_labels = test.get_labels()

    # Get a 15% Validation set
    validation = training.create_subset_arff(slice(arff.instance_count//5))
    training = training.create_subset_arff(slice(arff.instance_count//5,None))
    v_features = validation.get_features()
    v_labels = validation.get_labels()
    features = training.get_features()
    labels = training.get_labels()

    taccuracy = []
    accuracy = []
    domain = 2 ** np.arange(1,7)
    for i in domain:
        tacc = 0
        acc = 0
        for _ in range(3):
            learner = MultilayerPerceptronLearner([training.features_count] + [i]*(32//i) + [11], momentum=.85)
            # learner.zero_weights()
            learner.lr = .1
            learner.max_epoch = 500
            learner.train(training.get_features(), training.get_labels(), validation.get_features(), validation.get_labels())
            tacc += learner.get_accuracy(t_features, t_labels)
            acc += learner.get_accuracy(features,labels)
        accuracy.append(acc/3)
        taccuracy.append(tacc/3)

    plt.semilogx(domain, accuracy,basex= 2,label="Training Set Accuracy")
    plt.semilogx(domain, taccuracy, basex=2, label="Test Set Accuracy")
    plt.title("Node Distribution vs Accuracy")
    plt.xlabel("Number of Nodes per Hidden Layer")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
