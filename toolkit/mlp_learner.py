from .supervised_learner import SupervisedLearner
import numpy as np

class MultilayerPerceptronLearner(SupervisedLearner):

    MAX_EPOCH = 500
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

    def train(self,features, labels,v_features, v_labels):

        for epoch in range(self.max_epoch):
            self.train_one_epoch(features, labels)
            # Validation
            if self.check_for_convergence(v_features, v_labels):
                print("Converged in", self.epochs,'epochs')
                return
        print("Did not converge")

    def train_one_epoch(self,features, labels):
        features.shuffle(labels)
        for i in range(features.instance_count):
            self.train_one_instance(features.data[i], labels.data[i])
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


    def check_for_convergence(self,features, labels):
        errors = []
        for i in range(features.instance_count):
            errors.append(self.get_error(features.data[i],labels.data[i]))

        mse = sum(errors)/len(errors)
        print(mse)
        if mse <= self.bssf:
            self.bssf = mse
            self.epochs_since_bssf = 0

        else:
            self.epochs_since_bssf += 1

            if self.epochs_since_bssf >= 5:
                return True

        return False

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

    def get_mse(self, features, labels):

        errors = []
        for i in range(features.instance_count):
            errors.append(self.get_error(features.data[i],labels.data[i]))
        return sum(errors)/len(errors)

    def get_error(self, input, expected):
        a, output = self.forward_pass(input)
        if self.normalize:
            expected = self.normalize_output(expected)

        i = (output - expected)**2
        return sum(i)

    def get_accuracy(self, features, labels):
        total = []
        for i in range(features.instance_count):
            output = self.forward_pass(features.data[i])[1]
            expected = self.normalize_output(labels.data[i])
            total.append( np.argmax(output) == np.argmax(expected)  )
        return sum(total)/ len(total)


    def normalize_output(self, label):
        output = np.zeros(self.shape[-1])
        output[int(label)] = 1
        return output

    def zero_weights(self):
        for layer in range(len(self.weights)):
            self.weights[layer] = np.zeros_like(self.weights[layer])
