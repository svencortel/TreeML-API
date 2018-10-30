from DecisionTree.Node import *

class Tree:
    """The structure of the tree"""
    def __init__(self, criterion = "entropy", max_depth = None, random_feat=False):
        self.criterion = criterion
        self.max_depth = max_depth
        self.X_data = None
        self.y_data = None
        self.root_node = None
        self.random = random_feat

    def load(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def train(self, X_data = None, y_data = None):
        if X_data is None and y_data is None:
            if self.X_data is None:
                raise("No data loaded")
            X_data = self.X_data
            y_data = self.y_data

        # init root node
        self.root_node = Node(random_feat = self.random)
        self.root_node.train(X_data, y_data, self.max_depth)

    def predict(self, X_data):
        result = []
        for i in X_data:
            result.append(self.root_node.predictData(i))
        return np.array(result)

    def printTree(self):
        self.root_node.printNode()
