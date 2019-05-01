from .supervised_learner import SupervisedLearner
from .matrix import Matrix

# MAX_ITER is the maximum number of epochs to run before termination
MAX_ITER = 5

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
        weights = 2 * np.random.random(features.cols + 1,labels.cols) - 1

        ### TRAINING
        #   Iterate until
        #   a.) We've done MAX_ITER iterations
        #   b.) All the outputs are correct (Our model predicts sufficiently well)
        for _ in range(MAX_ITER):

            # train one epoch
            for row in range(features.rows):
                # Generate the input matrix, including the bias
                input = np.concatenate((features[row],[1]))

                # Calculate the output using the recall function
                output = input @ weights

                # Update the weights
                weights = weights - (output - labels[row]) * input



            pass
