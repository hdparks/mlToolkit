from .supervised_learner import SupervisedLearner
from .arff import Arff
import numpy as np
from matplotlib import pyplot as plt


# MAX_ITER is the maximum number of epochs to run before termination
MAX_ITER = 100

# LR is the learning rate, which measures how large a step is taken at each update
LR = .1

# TOL is the convergence threshold. If the error between epochs does not change
#   more than TOL, we consider the training as converging
TOL = .01

class PerceptronLearner(SupervisedLearner):
    """docstring for PerceptronLearner."""

    def train(self, features, labels, visualize = False, filename = None,lr=LR):
        """
        Parameters:
            :type features: Matrix, a matrix of input values
            :type labels: Matrix, a matrix of expected output values
        """
        ### INITIALIZATION
        #   Start with random weights in range [-1,1)
        weights = 2 * np.random.random((features.features_count + 1,labels.label_count)) - 1


        ### TRAINING
        #   Iterate until
        #   a.) We've done MAX_ITER iterations
        #   b.) All the outputs are correct (Our model predicts sufficiently well)
        weights_tracker = [weights]
        convergence_ticker = 5
        old_error = np.array([np.inf]*labels.instance_count)

        for iter in range(MAX_ITER):
            # train one epoch
            error = []
            for row in range(features.instance_count):
                input = np.concatenate((features[row],[1]))

                # Calculate the output using the recall function
                output = np.array(input @ weights) > 0
                output = output.astype(np.int)

                error.append(output - labels[row])
                # Update the weights
                weights = weights - lr * np.outer(input, output - labels[row])

                weights_tracker.append(weights)

            # Check to see if error value is converging
            delta_error = np.linalg.norm(error - old_error)
            if delta_error < TOL:
                convergence_ticker -= 1

                if convergence_ticker is 0:
                    print("--Training converged in {} epochs".format(iter + 1))
                    break
            else:
                convergence_ticker = 5

            old_error = np.array(error)


        self.weights = weights

        ### VISUALIZATION
        if visualize:
            self.visualize_training(features,labels,weights_tracker,filename)


    def visualize_training(self,features,labels,weights_tracker,filename):
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
            w = weights_tracker[i]
            divider.set_data([xmin,xmax],[(-xmin * w[0] - w[2]) / w[1], (-xmax * w[0] - w[2]) / w[1]])
            epoch_tracker.set_text(i//features.instance_count + 1)

            # Keep a shadow of the hyperplane at the end of each epoch
            if i % features.instance_count == 0:
                plot_hyperplane(w,xmin,xmax,iter = i, alpha = .3, color='black',linestyle='dashed')

            return divider

        ani = animation.FuncAnimation(fig, update, frames=range(len(weights_tracker)), interval=250,repeat=False)
        plt.legend()

        if filename:
            ani.save(filename, writer='imagemagick', fps=5)
        plt.show()

    def predict_all(self, features):
        """
        Use the weights calculated by training to predict the output of features
        self.train must be called before this function

        :type features: [float]
        """

        predictions = features.data @ self.weights[:-1] > -self.weights[-1]

        return predictions

    def graph_2D_features(self, features, labels):
        """
        Graphs the content of the arff, along with the calculated dividing line
        according to the calculated weights
        """
        # Plot feature points
        plot_2D_features(features, labels)

        w = np.array(self.weights)

        #   Find the min and max in x, y directions
        xmin = features.column_min(0) - .5
        xmax = features.column_max(0) + .5
        ymin = features.column_min(1) - .5
        ymax = features.column_max(1) + .5

        plot_hyperplane(w,xmin, xmax)

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.legend()

        plt.show()

def plot_2D_features(features, labels):
    """
    Make a scatterplot of the features, labeled according to the same labels
    """
    # Sort feature data by label
    label_map = get_label_map(labels)

    # Plot like labels
    for key in label_map.keys():
        like_ind = label_map[key]
        like_data = np.array([features[i] for i in like_ind])
        plt.scatter(like_data[:,0],like_data[:,1],label=key)

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
    import sys

    if len(sys.argv) < 2:
        raise ValueError("invalid arguments")


    if sys.argv[1] == "plot":
        print("Attempting to plot dividing line")

        arff = Arff(sys.argv[2])
        features = arff.get_features()
        labels = arff.get_labels()
        pl = PerceptronLearner()
        pl.train(features, labels)
        pl.graph_2D_features(features, labels)

    elif sys.argv[1] == "visualize_training":
        from matplotlib import animation
        print("Visualizing training")

        arff = Arff(sys.argv[2])
        features = arff.get_features()
        labels = arff.get_labels()

        pl = PerceptronLearner()

        if len(sys.argv) >= 4:
            if len(sys.argv) > 4:
                pl.train(features, labels, visualize=True, filename = sys.argv[3], lr=float(sys.argv[4]))
            else:
                pl.train(features, labels, visualize=True, filename=sys.argv[3])

        else:
            pl.train(features, labels, visualize=True)

        predictions = pl.predict_all(features)
        print("accuracy =", sum(predictions == labels.data)[0]/len(predictions))

    else:
        raise ValueError("invalid arguments")
