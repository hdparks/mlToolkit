from toolkit.perceptron_learner import PerceptronLearner
from toolkit.arff import Arff
import numpy as np
from scipy.stats import mode

def main():
    # Train each perceptron on its own split data set
    fast_v_mid = Arff('datasets/restaurants/fast_v_mid.arff')
    fast_v_fine = Arff('datasets/restaurants/fast_v_fine.arff')
    mid_v_fine = Arff('datasets/restaurants/mid_v_fine.arff')

    pl_fast_mid = PerceptronLearner()
    pl_fast_fine = PerceptronLearner()
    pl_mid_v_fine = PerceptronLearner()

    # Train each perceptron
    train_perceptron(pl_fast_mid,fast_v_mid)
    train_perceptron(pl_fast_fine,fast_v_fine)
    train_perceptron(pl_mid_v_fine,mid_v_fine)

    # Run on new data
    # Burger King
    burger_king = [4,2,2]

    # Cheesecake Factory
    cheesecake_factory = [2,4,4]

    # Best fine-dining in the world
    best_fine_dining = [1,5,3]

    print_findings("Burger King", determine_category(burger_king,pl_fast_mid,pl_fast_fine,pl_mid_v_fine))
    print_findings("Cheesecake Factory", determine_category(cheesecake_factory,pl_fast_mid,pl_fast_fine,pl_mid_v_fine))
    print_findings("'Best Food In The World'", determine_category(best_fine_dining,pl_fast_mid,pl_fast_fine,pl_mid_v_fine))

def print_findings(name, category):
    c_names = ["fast-food","mid-range","fine dining"]
    print("{} is a {} restaurant".format(name, c_names[category]))

def determine_category(input,fm,ff,mf):
    # fast-food = 0
    # mid-range = 1
    # fine-dining = 2

    fm_net = input @ fm.weights[:-1]
    ff_net = input @ ff.weights[:-1]
    mf_net = input @ mf.weights[:-1]

    fm_vote = 1 if fm_net > -fm.weights[-1] else 0
    ff_vote = 2 if ff_net > -ff.weights[-1] else 0
    mf_vote = 2 if mf_net > -mf.weights[-1] else 1

    if fm_vote == ff_vote and fm_vote == mf_vote:
        return np.argmax(np.abs([fm_net - fm.weights[-1], ff_net - ff.weights[-1], mf_net - mf.weights[-1]]))
    else:
        return mode([fm_vote,ff_vote,mf_vote])[0][0]

def train_perceptron(perceptron, arff):
    # Split the arff
    features = arff.get_features()
    labels = arff.get_labels()
    perceptron.train(features, labels)

if __name__ == '__main__':
    main()
