from .supervised_learner import SupervisedLearner
import numpy as np

class MultilayerPerceptronLearner(SupervisedLearner):

    MAX_EPOCH = 100
    DEFAULT_LR = .1

    def __init__(self, shape, momentum = 0, lr = DEFAULT_LR, max_epoch = MAX_EPOCH):

        self.shape = shape
        self.momentum = momentum
        self.weights = \
            [
                2 * np.array(np.random.random((shape[i]+1,shape[i+1]))) - 1
                for i in range(len(shape)-1)
            ]
        self.latest_weight_update = [ np.zeros_like(w) for w in self.weights ]
        self.lr = lr
        self.max_epoch = max_epoch
        self.normalize = True
        self.validation = False
        self.bssf = np.inf
        self.epochs = 0


    def set_validation(self, features, labels):
        self.validation = [features, labels]

    def set_training(self, features, labels):
        self.features = features
        self.labels = labels

    def train(self):

        for epoch in range(self.max_epoch):
            self.train_one_epoch()
            # Validation
            if self.check_for_convergence():
                return

    def train_one_epoch(self):
        self.features.shuffle(self.labels)
        for i in range(self.features.instance_count):
            self.train_one_instance(self.features.data[i], self.labels.data[i])
        self.epochs += 1

    def train_one_instance(self, input, expected):
        if self.normalize:
            expected = self.normalize_output(expected)

        # forward pass
        activations, output = self.forward_pass(input)

        # Get d
        d = ( expected - output ) * output * ( 1.0 - output )

        for layer in range(len(self.shape) - 2, -1, -1):

            a = self.concat_bias(activations[layer])
            update = self.momentum * self.latest_weight_update[layer] + self.lr * np.outer( a, d )

            self.latest_weight_update[layer] = update

            self.weights[layer] += update
            d = activations[layer] * (1.0 - activations[layer]) * (self.weights[layer][:-1] @ d)


    def check_for_convergence(self):
        if self.validation:
            features = self.validation[0]
            labels = self.validation[1]
        else:
            features = self.features.data
            labels = self.labels.data

        errors = []
        for i in range(len(features)):
            errors.append(self.get_error(features[i],labels[i]))

        mse = sum(errors)/len(errors)
        if mse <= self.bssf:
            self.bssf = mse
            self.epochs_since_bssf = 0

        else:
            self.epochs_since_bssf += 1

            if self.epochs_since_bssf >= 5:
                return True

        return False

    def get_mse(self, validation = False):
        if validation:
            features = self.validation[0]
            labels = self.validation[1]
        else:
            features = self.features.data
            labels = self.labels.data

        errors = []
        for i in range(len(features)):
            errors.append(self.get_error(features[i],labels[i]))

        return sum(errors)/len(errors)

    def get_error(self, input, expected):
        a, output = self.forward_pass(input)
        if self.normalize:
            expected = self.normalize_output(expected)

        i = (output - expected)**2
        return sum(i)

    def normalize_output(self, label):
        output = np.zeros(self.shape[-1])
        output[int(label)] = 1
        return output

    def forward_pass(self, input):
        forward = input.copy()
        activations = [forward]
        for layer in range(len(self.shape)-1):
            forward = self.sigmoid( self.concat_bias(forward) @ self.weights[layer] )
            activations.append(forward)

        return activations, forward

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def concat_bias(self, activation):
        return np.concatenate((activation,[1]))

    def get_accuracy(self, features, labels):
        total = []
        for i in range(len(features)):
            total.append( np.argmax(self.forward_pass(features[i])[1]) == np.argmax(self.normalize_output(labels[i]))  )

        return sum(total)/len(total)

    def get_test_mse(self, features, labels):
        errors = []
        for i in range(len(features)):
            errors.append(self.get_error(features[i],labels[i]))

        return sum(errors)/len(errors)

    def get_test_accuracy(self, features, labels):
        total = []

        for i in range(len(features)):
            total.append( np.argmax(self.forward_pass(features[i])[1]) == np.argmax(self.normalize_output(labels[i]))  )

        return sum(total)/len(total)
