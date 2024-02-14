

class DecisionTree:
    def __init__(self, gain_threshold, node_max):
        self.gain_threshold = gain_threshold
        if node_max == 0:
            self.node_max == None
        else:
            self.node_max == node_max
        self.root_node = None

    def train(self, train_set):
        self.root_node = treenode.TreeNode(train_set)
        node_count = 1
        data_at_node = self.root_node.get_instances()
        while(True):
            #compute current gain
            #compute best split
            #check gain threshold


