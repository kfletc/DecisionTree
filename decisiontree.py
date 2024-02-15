# decisiontree.py
# represents a decision tree with a gain threshold stopping condition, has methods to train and test off of datasets

from decisionsplit import get_best_split
import treenode

class DecisionTree:
    def __init__(self, gain_threshold):
        self.gain_threshold = gain_threshold
        self.root_node = None
        self.node_count = 0
        self.leaf_node_count = 0

    # train model off ofa training set of data
    def train(self, train_set):
        self.root_node = treenode.TreeNode(train_set, True)
        self.node_count += 1
        self.split(self.root_node)

    # recursively builds a tree finding best splits until the gain is below the threshold
    def split(self, node):
        data = node.data_input
        attribute, rule, gain = get_best_split(data)
        if gain < self.gain_threshold:
            self.leaf_node_count += 1
            node.set_class()
        else:
            node.attribute = attribute
            node.rule = rule
            left_child, right_child = node.create_children()
            self.node_count += 2
            self.split(left_child)
            self.split(right_child)

    # runs a test set of data through the created tree to assign classes
    def test(self, test_set):
        return_set = test_set
        return_set['PredictedClass'] = "None"
        for index, row in test_set.iterrows():
            prediction = self.traverse_tree(row, self.root_node)
            return_set.at[index, 'PredictedClass'] = prediction
        return return_set

    # runs an instance of data through the tree recursively until it classifies it
    def traverse_tree(self, row, node):
        if node.is_leaf:
            return node.assigned_class
        if row[node.attribute] == node.rule:
            return self.traverse_tree(row, node.right_child)
        else:
            return self.traverse_tree(row, node.left_child)
