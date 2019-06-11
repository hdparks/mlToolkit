import numpy as np
from scipy import stats

class DecisionTreeLearner():
    def __init__(self):
        pass

    def train(self,features, labels):
        # Setup
        self.feature_names = features.get_attr_names()
        self.enum_to_str = features.enum_to_str
        self.str_to_enum = features.str_to_enum
        self.classes = labels.enum_to_str[0]
        default_class = labels.most_common_value(0)
        self.tree = self.make_tree(features, labels, default_class)

    def make_tree(self, features, labels, default_class):
        # If tree is empty, return a leaf
        if features.instance_count == 0 or len(features.get_attr_names()) == 0:
            return {'feature': 'class', 'class': self.classes[default_class] }

        # Get possible class types
        class_types = set(np.ravel(labels))

        # If not leaf, build next layer based on gains
        gains = [self.calc_slice_entropy(features.data[:,i], labels.data) for i in range(features.features_count)]
        best_split_idx = np.argmax(gains)
        subtree = {'feature' : features.attr_names[best_split_idx] }

        feature_instances = set(features.str_to_enum[best_split_idx].keys())
        splits = [np.where(features.data[:,best_split_idx] == features.str_to_enum[best_split_idx][f] )[0] for f in feature_instances]

        # Split features on best index
        splitlist = list(range(features.features_count))
        splitlist.pop(best_split_idx)
        features = features.create_subset_arff(col_idx = splitlist)

        for split, name in zip(splits, feature_instances):
            new_features = features.create_subset_arff(row_idx=split)
            new_labels = labels.create_subset_arff(row_idx=split)

            if self.num_unique_vals(new_labels.data) == 1:
                subtree[name] = {'feature' : 'class', 'class': self.classes[new_labels.data[0][0].astype(int)] }
            else:
                new_default = new_labels.most_common_value(0) if new_labels.instance_count > 0 else default_class
                subtree[name] = self.make_tree(new_features, new_labels, new_default)
        return subtree

    def num_unique_vals(self, data):
        return len(set(np.ravel(data)))

    def get_accuracy(self, features, labels):
        pred = self.predict_all(features)
        acc = pred == np.ravel(labels.data)
        return sum(acc)/len(acc)

    def predict(self, data):
        node = self.tree
        while(True):
            f = node['feature']
            if f == 'class':
                return node['class']
            i = self.feature_names.index(f)
            t = self.name_dict[i][int(data[i])]
            node = node[t]

    def predict_all(self, features):
        return np.array([self.predict(data) for data in features])

    def calc_entropy(self,classes):

        # Get possible types:
        class_types = set(np.ravel(classes))
        # Get percentages
        p = [sum(np.array(classes) == c_type)/len(classes) for c_type in class_types]

        # Return total entropy (sum of entropy values for each type)
        entropy = lambda p : p * np.log2(p) if p != 0 else 0
        return sum([entropy(p_) for p_ in p])

    def calc_slice_entropy(self, data, classes):
        """ Calculate the gain of splitting the data on the given feature
            data = the data
            classes = the output, what we check against (labels)
            feature = the column index for the given data

            Gain(data, feature) = Entropy(data) - sum_over_feature( % of feature density * entropy of data split on feature)
        """
        # Get total entropy
        total_entropy = self.calc_entropy(classes)

        entropy = 0

        # Get a list of feature instances

        feature_instances = set(data)

        # For each feature, calculate its density (percentage) and
        # the entropy of the resulting data set
        for instance in feature_instances:
            points = np.where(data == instance)[0]
            density = len(points) / len(data)
            split_entropy = self.calc_entropy(classes[points])
            entropy += density * split_entropy

        return entropy

    def prune(self, vf, vl):
        current_accuracy = self.get_accuracy(vf,vl)
        node = self.tree

        def recurse(node):
            # trim the tree
            old_node = node.copy()

            mcc = self.most_common_class(node)
            node = {'feature':'class', 'class': mcc }

            # Get validation accuracy
            new_acc = self.get_accuracy(vf, vl)

            if new_acc > current_accuracy:
                current_accuracy = new_acc
                # We are done on this branch
                return

            else:

                for child in node.values():
                    if type(child) == str:
                        continue



    def most_common_class(self, node):
        def recurse(node):
            if node['feature'] == 'class':
                return [node['class']]

            children = []
            for child in node.values():
                if type(child) == str:
                    continue
                children += recurse(child)
            return children

        classifications = recurse(node)
        return stats.mode(classifications)[0][0]
