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
TOL = .02

class PerceptronLearner(SupervisedLearner):
    """docstring for PerceptronLearner."""

    def __init__(self, lr=LR, vis=False, file_name=None,verbose=False):
        self.lr = lr
        self.visualize = vis
        self.file_name = file_name
        self.verbose = verbose

    def train(self, features, labels):
        """
        Parameters:
            :type features: Matrix, a matrix of input values
            :type labels: Matrix, a matrix of expected output values
        """
        ### INITIALIZATION
        self.features = features
        self.labels = labels
        self.weights = 2 * np.random.random(features.features_count + 1) - 1
        self.accuracy = self.calc_accuracy(features, labels,return_scalar=True)
        self.weights_tracker = [self.weights]
        self.accuracy_tracker = [self.accuracy]

        ### TRAINING
        self.convergence_ticker = 5

        for iter in range(MAX_ITER):

            # Suffle the features before each new epoch
            features.shuffle(labels)

            for row in range(features.instance_count):
                input = np.concatenate((features[row],[1]))

                # Calculate the output using the recall function
                output = input @ self.weights > 0
                output = output.astype(np.int)
                # Update the weights
                self.weights = self.weights + self.lr * input * (labels[row] - output)
                self.weights_tracker.append(self.weights)

            #   Check for convergence, defined by change in overall accuracy
            current_accuracy = self.calc_accuracy(features, labels,return_scalar=True)
            self.accuracy_tracker.append(current_accuracy)

            if abs(self.accuracy - current_accuracy) < TOL:
                self.convergence_ticker -= 1

                if self.convergence_ticker == 0:
                    print("Converged in {} epochs".format(iter + 1))
                    break
            else:
                self.convergence_ticker = 5

            self.accuracy = current_accuracy

        if self.convergence_ticker != 0:
            print("Did not converge in {} iterations".format(MAX_ITER))

        ### VISUALIZATION
        if self.visualize:
            if features.features_count != 2:
                print(features.features_count)
                print("Must be a 2 dimensional feature set to visualize training")

            else:
                self.visualize_training()

    def predict_all(self, features):
        """
        Use the weights calculated by training to predict the output of features
        self.train must be called before this function

        :type features: [float]
        """
        predictions = features.data @ self.weights[:-1] > -self.weights[-1]

        return np.array(predictions).reshape((len(predictions),1))

    def visualize_training(self):
        """
        Show an animation of the training process
        """

        print("Visualizing training")
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Take out each feature type, one at a time
        label_map = get_label_map(self.labels)

        for key in label_map.keys():
            like_ind = label_map[key]
            like_data = np.array([self.features[i] for i in like_ind])

            plt.scatter(like_data[:,0],like_data[:,1],label=key)

        # get limits
        xmin = self.features.column_min(0) - .5
        xmax = self.features.column_max(0) + .5
        ymin = self.features.column_min(1) - .5
        ymax = self.features.column_max(1) + .5

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
            epoch = i//self.features.instance_count
            w = self.weights_tracker[i]
            a = self.accuracy_tracker[epoch]
            divider.set_data([xmin,xmax],[(-xmin * w[0] - w[2]) / w[1], (-xmax * w[0] - w[2]) / w[1]])
            epoch_tracker.set_text("{} {}".format(epoch + 1, a))

            # Keep a shadow of the hyperplane at the end of each epoch
            if i % self.features.instance_count == 0:
                plot_hyperplane(w,xmin,xmax,iter = i, alpha = .3, color='black',linestyle='dashed')

            return divider
        from matplotlib import animation
        ani = animation.FuncAnimation(fig, update, frames=range(len(self.weights_tracker)), interval=250,repeat=False)
        plt.legend()

        if self.file_name != None:
            ani.save(self.file_name, writer='imagemagick', fps=5)
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
