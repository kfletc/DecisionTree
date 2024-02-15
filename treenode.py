# treenode.py
# represents a node in a binary tree that has associated data and a rule for splitting to left and right subtrees
# or if it doesn't have a rule it assigns a class and is a leaf node

class TreeNode:
    def __init__(self, data_input, root):
        self.data_input = data_input
        self.is_root = root
        self.is_leaf = False
        self.assigned_class = None
        self.right_child = None
        self.left_child = None
        self.attribute = None
        self.rule = None
        if self.is_leaf:
            self.assigned_class = self.__determine_class()

    # determine class of a leaf node
    def set_class(self):
        self.is_leaf = True
        classes = self.data_input['class'].drop_duplicates()
        max_count = 0
        prominent_class = None
        for c in classes:
            c_df = self.data_input[self.data_input['class'] == c]
            c_count = c_df.shape[0]
            if c_count > max_count:
                max_count = c_count
                prominent_class = c
        self.assigned_class = prominent_class

    # applies the rule of the node to the data and sends abiding data into the right subtree
    # and the rest into the left subtree
    def create_children(self):
        right_data = self.data_input[self.data_input[self.attribute] == self.rule]
        left_data = self.data_input[self.data_input[self.attribute] != self.rule]
        self.right_child = TreeNode(right_data, False)
        self.left_child = TreeNode(left_data, False)
        return self.right_child, self.left_child
