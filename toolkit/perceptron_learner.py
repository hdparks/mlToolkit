from .supervised_learner import SupervisedLearner
import numpy as np
# MAX_ITER is the maximum number of epochs to run before termination
MAX_ITER = 100

# LR is the learning rate, which measures how large a step is taken at each update
LR = .1

# TOL is the convergence threshold. If the weights between epochs do not change
#   more than TOL, we consider the training as converging
TOL = .5

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

        for iter in range(MAX_ITER):

            old_weights = weights.copy()

            # train one epoch
            for row in range(features.instance_count):
                input = np.concatenate((features[row],[1]))

                # Calculate the output using the recall function
                output = np.array(input @ weights) > 0
                output = output.astype(np.int)

                # Update the weights
                weights = weights - LR * np.outer(input, output - labels[row])


            # Check to see if weights are converging
            if np.linalg.norm(weights - old_weights) < TOL:
                convergence_ticker -= 1

                if convergence_ticker is 0:
                    self.weights = weights
                    print("--Training converged in {} epochs".format(iter + 1))
                    return
            else:
                convergence_ticker = 5


        print("--Training did not converge")
        self.weights = weights


    def predict_all(self, features):
        """
        Use the weights calculated by training to predict the output of features
        self.train must be called before this function

        :type features: [float]
        """
        if self.weights is None:
            raise NotImplementedError("Must call train before predict_all")

        predictions = features.data @ self.weights[:-1] > -self.weights[-1]

        return  predictions
