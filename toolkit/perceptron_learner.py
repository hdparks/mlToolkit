from .supervised_learner import SupervisedLearner
from .matrix import Matrix

class PerceptronLearner(SupervisedLearner):
    """docstring for PerceptronLearner."""

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """

        #   Start with weights set to 1
        print("Features:")
        features.print()
        print("labels: ")
        labels.print()
