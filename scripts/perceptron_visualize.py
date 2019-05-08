from toolkit.perceptron_learner import PerceptronLearner
from toolkit.arff import Arff
import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib import animation

def main():
    arff = Arff(sys.argv[1])
    features = arff.get_features()
    labels = arff.get_labels()

    pl =  PerceptronLearner()
    pl.train(features, labels)

    visualize_training(features, labels, pl)

def visualize_training(features, labels, pl):
    """
    Show an animation of the training process
    """
    print("Visualizing training")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Take out each feature type, one at a time
    label_map = get_label_map(labels)

    for key in label_map.keys():
        like_ind = label_map[key]
        like_data = np.array([features[i] for i in like_ind])

        plt.scatter(like_data[:,0],like_data[:,1],label=key)

    # get limits
    xmin = features.column_min(0) - .5
    xmax = features.column_max(0) + .5
    ymin = features.column_min(1) - .5
    ymax = features.column_max(1) + .5

    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)

    # Track the current dividing line, as well as the number of epochs passed
    divider, = plt.plot([],[])
    epoch_tracker = plt.text(-1,.9, '', fontsize=15)

    def update(i):
        """
        1.) Get the next set of weights from the tracker
        2.) Calculate and draw the new divider line
        3.) Update the epoch counter
        4.) If we are at the end of an epoch, plot a dashed divider line to track progress
        """
        epoch = i//features.instance_count
        w = pl.weights_tracker[i]
        a = pl.accuracy_tracker[epoch]
        divider.set_data([xmin,xmax],[(-xmin * w[0] - w[2]) / w[1], (-xmax * w[0] - w[2]) / w[1]])
        epoch_tracker.set_text("{} {}".format(epoch + 1, a))

        # Keep a shadow of the hyperplane at the end of each epoch
        if i % features.instance_count == 0:
            plot_hyperplane(w,xmin,xmax,iter = i, alpha = .3, color='black',linestyle='dashed')

        return divider

    ani = animation.FuncAnimation(fig, update, frames=range(len(pl.weights_tracker)), interval=250,repeat=False)
    plt.legend()

    # optional save file
    if len(sys.argv) >= 3 :
        ani.save(sys.argv[2], writer='imagemagick', fps=5)

    plt.show()


def plot_hyperplane(w, xmin, xmax, iter = None, alpha=None, color=None,linestyle=None):
    """
    Plot the separating hyperplane defined by the supplied weights vector
    """
    #   Find the points on the dividing hyperplane
    #   using p[0]w[0] + p[1]w[1] = -w[2], or p[1] = (-p[0]w[0] - w[2])/w[1]
    x_points = [xmin, xmax]
    y_points = [(-xmin * w[0] - w[2]) / w[1], (-xmax * w[0] - w[2]) / w[1]]

    plt.plot(x_points, y_points, label="Decision line" if iter is None else iter, alpha=alpha, color=color,linestyle=linestyle)

def get_label_map(labels):
    """
    Given a matrix of labels, generate a map from label type to indices
    """
    label_map = dict()
    for i,v in enumerate(np.ravel(labels.data)):
        if v in label_map.keys():
            label_map.get(v).append(i)
        else:
            label_map[v] = [i]
    return label_map

if __name__ == '__main__':
    main()
