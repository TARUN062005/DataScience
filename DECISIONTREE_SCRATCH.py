import random
import math

# Helper functions
def calculate_entropy(data):
    """
    Calculate the entropy of a dataset.
    data: List of target labels
    """
    total = len(data)
    if total == 0:
        return 0
    counts = {}
    for label in data:
        counts[label] = counts.get(label, 0) + 1
    entropy = 0
    for count in counts.values():
        prob = count / total
        entropy -= prob * math.log2(prob)
    return entropy

def split_data(dataset, feature_index):
    """
    Split the dataset based on a feature.
    dataset: List of lists where each inner list is a data point
    feature_index: Index of the feature to split on
    """
    splits = {}
    for row in dataset:
        key = row[feature_index]
        if key not in splits:
            splits[key] = []
        splits[key].append(row)
    return splits

def calculate_information_gain(dataset, feature_index, target_index):
    """
    Calculate the Information Gain for splitting on a specific feature.
    dataset: List of lists where each inner list is a data point
    feature_index: Index of the feature to split on
    target_index: Index of the target variable
    """
    total_entropy = calculate_entropy([row[target_index] for row in dataset])
    splits = split_data(dataset, feature_index)
    total_samples = len(dataset)
    
    weighted_entropy = 0
    for subset in splits.values():
        prob = len(subset) / total_samples
        subset_entropy = calculate_entropy([row[target_index] for row in subset])
        weighted_entropy += prob * subset_entropy
    
    information_gain = total_entropy - weighted_entropy
    return information_gain

# Decision Tree Classifier
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, dataset, features, target_index):
        self.tree = self._build_tree(dataset, features, target_index, depth=0)

    def _build_tree(self, dataset, features, target_index, depth):
        target_values = [row[target_index] for row in dataset]
        if len(set(target_values)) == 1:
            return target_values[0]
        if not features or (self.max_depth is not None and depth >= self.max_depth):
            return max(set(target_values), key=target_values.count)

        best_feature_index = -1
        best_gain = -float('inf')
        for i in range(len(features)):
            gain = calculate_information_gain(dataset, i, target_index)
            if gain > best_gain:
                best_gain = gain
                best_feature_index = i

        if best_gain == 0:
            return max(set(target_values), key=target_values.count)

        best_feature = features[best_feature_index]
        splits = split_data(dataset, best_feature_index)
        subtree = {}
        remaining_features = features[:best_feature_index] + features[best_feature_index + 1:]

        for value, subset in splits.items():
            subtree[value] = self._build_tree(subset, remaining_features, target_index, depth + 1)

        return {best_feature: subtree}

    def predict(self, row, feature_names):
        """
        Predict the class label for a single data point.
        row: List of feature values
        feature_names: List of feature names to map indices
        """
        node = self.tree
        while isinstance(node, dict):
            feature = list(node.keys())[0]
            feature_index = feature_names.index(feature)
            value = row[feature_index]
            node = node[feature].get(value, None)
            if node is None:
                return None
        return node

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, sample_ratio=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_ratio = sample_ratio
        self.trees = []

    def fit(self, dataset, features, target_index):
        for _ in range(self.n_trees):
            sampled_data = self._bootstrap_sample(dataset)
            sampled_features = self._random_feature_subset(features)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(sampled_data, sampled_features, target_index)
            self.trees.append((tree, sampled_features))

    def _bootstrap_sample(self, dataset):
        n_samples = int(len(dataset) * self.sample_ratio)
        return [random.choice(dataset) for _ in range(n_samples)]

    def _random_feature_subset(self, features):
        n_features = math.ceil(math.sqrt(len(features)))
        return random.sample(features, n_features)

    def predict(self, row):
        predictions = []
        for tree, features in self.trees:
            predictions.append(tree.predict(row, features))
        return max(set(predictions), key=predictions.count)

# Example Usage
dataset = [
    ['Sunny', 'Hot', 'High', 'No'],
    ['Sunny', 'Hot', 'High', 'No'],
    ['Overcast', 'Hot', 'High', 'Yes'],
    ['Rainy', 'Mild', 'High', 'Yes'],
    ['Rainy', 'Cool', 'Normal', 'Yes'],
    ['Rainy', 'Cool', 'Normal', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Yes'],
    ['Sunny', 'Mild', 'High', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Yes'],
    ['Rainy', 'Mild', 'Normal', 'Yes']
]

features = ['Outlook', 'Temperature', 'Humidity']
target_index = 3                                                                                                                                                         

forest = RandomForest(n_trees=5, max_depth=3, sample_ratio=0.8)
forest.fit(dataset, features, target_index)

# Predict an example
test_row = ['Sunny', 'Mild', 'High']  # Example row for prediction
prediction = forest.predict(test_row)
print("Prediction:", prediction)