import numpy as np
from matplotlib import pyplot as plt
from toolkit.k_nearest_neighbor import KNN
from toolkit.arff import Arff

def prob_2():
    # try first without normalizing
    train = Arff('datasets/magic_telescope_train.arff')
    test = Arff('datasets/magic_telescope_test.arff')

    k = KNN(3)
    predictions = k.knn(train.get_features(), train.get_labels(), test.get_features())

    acc = predictions == np.ravel(test.get_labels().data)

    print("Before normalization:", sum(acc)/len(acc))

    train.normalize()
    test.normalize()
    predictions = k.knn(train.get_features(), train.get_labels(), test.get_features())

    acc = predictions == np.ravel(test.get_labels().data)

    print("After normalization:", sum(acc)/len(acc))

    print("PART TWO:")
    krange = np.arange(1,16,2)
    accs = []
    for k in krange:
        knn = KNN(k)
        predictions = knn.knn(train.get_features(), train.get_labels(), test.get_features())
        acc = predictions == np.ravel(test.get_labels().data)
        print("k:",k, "accuracy:",sum(acc)/len(acc))
        accs.append(sum(acc)/len(acc))

    plt.plot(krange, accs)
    plt.title("K Size Versus Accuracy")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.show()

def prob_3():
    # Use regression knn on housing price prediction dataset
    train = Arff('datasets/housing_train.arff')
    test = Arff('datasets/housing_test.arff')
    train.normalize()
    test.normalize()

    krange = np.arange(1,16,2)
    mses = []
    for k in krange:
        knn = KNN(k)
        preds = knn.knn(train.get_features(), train.get_labels(), test.get_features())
        mse = sum( (preds -  np.ravel(test.get_labels().data))**2 )/ len(preds)
        mses.append(mse)

    plt.plot(krange, mses)
    plt.title("K Size Versus MSE on Housing Prices")
    plt.xlabel("K")
    plt.ylabel("Mean Squared Error")
    plt.show()

def prob_4_telescope():
    # Repeat experiments for magic telescope and housing using weights (w = 1/dist**2)
    train = Arff('datasets/magic_telescope_train.arff')
    test = Arff('datasets/magic_telescope_test.arff')
    train.normalize()
    test.normalize()

    krange = np.arange(1,16,2)
    accs = []
    for k in krange:
        knn = KNN(k,weighting=True)
        predictions = knn.knn(train.get_features(), train.get_labels(), test.get_features())
        acc = predictions == np.ravel(test.get_labels().data)
        print("k:",k, "accuracy:",sum(acc)/len(acc))
        accs.append(sum(acc)/len(acc))

    plt.plot(krange, accs)
    plt.title("K Size Versus Accuracy")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.show()

def prob_4_housing():

    # Repeat experiments for magic telescope and housing using weights (w = 1/dist**2)
    train = Arff('datasets/housing_train.arff')
    test = Arff('datasets/housing_test.arff')
    train.normalize()
    test.normalize()

    krange = np.arange(1,16,2)
    mses = []
    for k in krange:
        knn = KNN(k,weighting=True)
        preds = knn.knn_regression(train.get_features(), train.get_labels(), test.get_features())
        mse = np.sum( (preds -  np.ravel(test.get_labels().data))**2 ,axis=0)/ len(preds)
        mses.append(mse)

    plt.plot(krange, mses)
    plt.title("K Size Versus MSE on Housing (Weighted)")
    plt.xlabel("K")
    plt.ylabel("Mean Squared Error")
    plt.show()

def prob_5():
    # Repeat experiments for magic telescope and housing using weights (w = 1/dist**2)
    arff = Arff('datasets/credit.arff')
    arff.shuffle()
    test = arff.create_subset_arff(slice(arff.instance_count//4))
    train = arff.create_subset_arff(slice(arff.instance_count//4,None))


    train.normalize()
    test.normalize()

    krange = np.arange(1,16,2)
    accs = []
    for k in krange:
        knn = KNN(k,weighting=True)
        predictions = knn.knn(train.get_features(), train.get_labels(), test.get_features())
        acc = predictions == np.ravel(test.get_labels().data)
        print("k:",k, "accuracy:",sum(acc)/len(acc))
        accs.append(sum(acc)/len(acc))

    plt.plot(krange, accs)
    plt.title("K Size Versus Accuracy on Credit Approval (Weighted)")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.show()

def prob_6():
    # Repeat experiments for magic telescope and housing using weights (w = 1/dist**2)
    arff = Arff('datasets/credit.arff')
    arff.shuffle()
    test = arff.create_subset_arff(slice(arff.instance_count//4))
    train = arff.create_subset_arff(slice(arff.instance_count//4,None))


    train.normalize()
    test.normalize()

    krange = np.arange(1,16,2)
    accs = []
    for k in krange:
        knn = KNN(k,weighting=True,vdm = True)
        predictions = knn.knn(train.get_features(), train.get_labels(), test.get_features())
        acc = predictions == np.ravel(test.get_labels().data)
        print("k:",k, "accuracy:",sum(acc)/len(acc))
        accs.append(sum(acc)/len(acc))

    plt.plot(krange, accs)
    plt.title("K Size Versus Accuracy on Credit Approval (Weighted)")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.show()
