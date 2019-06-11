from toolkit.arff import Arff
from toolkit.decision_tree_learner import DecisionTreeLearner
import numpy as np

d = None

def prob_0():
    arff = Arff('datasets/lenses.arff')
    d = DecisionTreeLearner()
    f = arff.get_features()
    l = arff.get_labels()
    d.train(f,l)
    print(d.tree)
    # print(d.get_accuracy(f,l))

def prob_1():
    arff = Arff('datasets/lenses.arff')
    acc = []
    for _ in range(10):

        arff.shuffle()
        testing = arff.create_subset_arff(slice(arff.instance_count//5))
        training = arff.create_subset_arff(slice(arff.instance_count//5, None))

        d = DecisionTreeLearner()

        features = training.get_features()
        labels = training.get_labels()


        t_feat = testing.get_features()
        t_labels = testing.get_labels()

        d.train(features, labels)

        accuracy = d.get_accuracy(t_feat, t_labels)
        print(accuracy)
        acc.append(accuracy)

    print(sum(acc)/len(acc))

def prob_2():
    # Get accuracy on cars.arff
    arff = Arff('datasets/cars.arff')
    arff.shuffle()

    acc, tacc, = k_fold_cv(arff, 10)
    print('cars:')
    print('acc',acc)
    print('tacc',tacc)
    print('tot',sum(tacc)/len(tacc))
    print()

    # Get accuracy of voting.arff
    arff = Arff('datasets/voting.arff')
    arff.shuffle()
    acc,tacc, = k_fold_cv(arff, 10)

    print('voting;')
    print('acc',acc)
    print('tacc',tacc)
    print('tot',sum(tacc)/len(tacc))
    print()

def prob_3():
    print('cars')
    arff = Arff('datasets/cars.arff')
    arff.shuffle()
    d = DecisionTreeLearner()
    d.train(arff.get_features(), arff.get_labels())

    a = d.tree

    print()
    print('voting')
    arff = Arff('datasets/voting.arff')
    arff.shuffle()
    d = DecisionTreeLearner()
    d.train(arff.get_features(), arff.get_labels())

    b = d.tree

    return a, b

def prob_5():
        arff = Arff('datasets/cars.arff')
        arff.shuffle()

        test = arff.create_subset_arff(slice(arff.instance_count//10))
        training = arff.create_subset_arff(slice(arff.instance_count//10,None))

        tf = test.get_features()
        tl = test.get_labels()

        splits = k_fold_cv(arff)

        arff = arff.create_subset_arff(slice(arff.instance_count//4,None))
        d = DecisionTreeLearner()
        d.train(arff.get_features(), arff.get_labels())

        a = d.tree

        arff = Arff('datasets/voting.arff')
        arff.shuffle()
        arff = arff.create_subset_arff(slice(arff.instance_count//4,None))
        d = DecisionTreeLearner()
        d.train(arff.get_features(), arff.get_labels())

        b = d.tree

        return a, b


def k_fold_split(arff,k):
    n = arff.instance_count
    splits = np.linspace(0,n,k+1).astype(int)
    return [arff.create_subset_arff(slice(splits[i],splits[i+1])) for i in range(len(splits)-1)]

def k_fold_cv(arff, k):
    splits = k_fold_split(arff,k)
    t_f = splits[-1].get_features()
    t_l = splits[-1].get_labels()
    acc = []
    tacc = []
    d_list = []
    for i in range(k-1):
        f = splits[i].get_features()
        l = splits[i].get_labels()
        d = DecisionTreeLearner()
        d.train(f,l)
        acc.append(d.get_accuracy(f,l))
        tacc.append(d.get_accuracy(t_f,t_l))
        dlist.append(d)
    return acc, tacc, dlist
