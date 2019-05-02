from .supervised_learner import SupervisedLearner
import numpy as np
from matplotlib import pyplot as plt

# MAX_ITER is the maximum number of epochs to run before termination
MAX_ITER = 100

# LR is the learning rate, which measures how large a step is taken at each update
LR = .1

# TOL is the convergence threshold. If the error between epochs does not change
#   more than TOL, we consider the training as converging
TOL = 1

class PerceptronLearner(SupervisedLearner):
    """docstring for PerceptronLearner."""

    def train(self, features, labels):
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
        convergence_ticker = 5
        old_error = np.array([np.inf]*labels.label_count)
        for iter in range(MAX_ITER):

            # train one epoch
            for row in range(features.instance_count):
                input = np.concatenate((features[row],[1]))

                # Calculate the output using the recall function
                output = np.array(input @ weights) > 0
                output = output.astype(np.int)

                error = output - labels[row]

                # Update the weights
                weights = weights - LR * np.outer(input, output - labels[row])


            # Check to see if error value is converging
            if np.linalg.norm(error - old_error) < TOL:
                convergence_ticker -= 1

                if convergence_ticker is 0:
                    self.weights = weights
                    print("--Training converged in {} epochs".format(iter + 1))
                    return
            else:
                convergence_ticker = 5

            old_error = error

        print("--Training did not converge")
        self.weights = weights


    def predict_all(self, features):
        """
        Use the weights calculated by training to predict the output of features
        self.train must be called before this function

        :type features: [float]
        """
        # if None == self.weights:
        #     raise NotImplementedError("Must call train before predict_all")

        predictions = features.data @ self.weights[:-1] > -self.weights[-1]

        return  predictions

    def graph_2D_features(self, features, labels):
        """
        Graphs the content of the arff, along with the calculated dividing line
        according to the calculated weights
        """
        # Check to make sure training has already happened
        # if None == self.weights:
        #     raise NotImplementedError("Must call train before graph_2D_features")

        # Sort feature data by label
        label_map = dict()
        for i,v in enumerate(np.ravel(labels.data)):
            if v in label_map.keys():
                label_map.get(v).append(i)
            else:
                label_map[v] = [i]

        # Plot like labels
        for key in label_map.keys():
            like_ind = label_map[key]
            like_data = np.array([features[i] for i in like_ind])
            plt.scatter(like_data[:,0],like_data[:,1],label=key)

        # Calculate separating hyperplane using weights
        # w vector is given by [w[0],w[1]], normalized to length of w[2]
        w = np.array([self.weights[0],self.weights[1]])

        #   Find the min and max in x, y directions
        xmin = features.column_min(0)
        xmax = features.column_max(0)
        ymin = features.column_min(1)
        ymax = features.column_max(1)

        #   Find the points on the dividing hyperplane
        #   using p[0]w[0] + p[1]w[1] = 0, or p[1] = -p[0]w[0]/w[1]
        x_points = [xmin, xmax]
        y_points = [-xmin * self.weights[0] / self.weights[1], -xmax * self.weights[0] / self.weights[1]]

        plt.xlim(xmin - .5, xmax + .5)
        plt.ylim(ymin - .5, ymax + .5)
        plt.legend()

        plt.plot(x_points, y_points, label="Decision line")

        plt.show()
