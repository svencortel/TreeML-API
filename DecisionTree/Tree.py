from DecisionTree.Node import *
from sklearn.metrics import accuracy_score

class Tree:
    """The structure of the tree"""
    def __init__(self, criterion = "entropy", max_depth = None,
                 random_feat=False, PFSRT=False, omega = 1.9, theta = 0.9):
        self.criterion = criterion
        self.max_depth = max_depth
        self.X_data = None
        self.y_data = None
        self.root_node = None
        self.random = random_feat
        self.nr_features = 0

        # PFSRT variables
        self.is_PFSRT = PFSRT
        self._best_accuracy = 0
        self._cur_accuracy = 0
        self.omega = omega # reward
        self.theta = theta # punish

        # Probabilistic Feature Selection Random Tree (PFSRT)
        # DS = Depth Score
        # PS = Prior Score
        self.DS = None
        self.PS = None

    def load(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        self.nr_features = X_data.shape[1]
        # updated data means resetting PFSRT
        self._best_accuracy = 0
        self._cur_accuracy = 0
        if self.is_PFSRT:
            self.DS = np.ones((self.nr_features, self.max_depth))
            self.PS = np.ones((self.nr_features, self.nr_features+1))
            print(self.DS)
            print(self.PS)
            print(self.DS.shape)
            print(self.PS.shape)
            print(self.nr_features)

    def train(self, X_data = None, y_data = None):
        if X_data is None and y_data is None:
            if self.X_data is None:
                raise("No data loaded")
        else:
            self.load(X_data, y_data)

        # init root node and start training
        self.root_node = Node(random_feat = self.random, tree=self)
        self.root_node.train(self.X_data, self.y_data, self.max_depth)

    def updatePFSRT(self):
        if not self.is_PFSRT:
            raise Exception("Must enable PFSRT=True")
        # test over the training data and update DS and PS
        y_pred = self.predict(self.X_data)
        self._cur_accuracy = accuracy_score(y_pred, self.y_data)
        # update DS and PS
        self.root_node.recursiveUpdatePFSRT()
        if self._cur_accuracy > self._best_accuracy:
            self._best_accuracy = self._cur_accuracy

    def predict(self, X_data):
        result = []
        for i in X_data:
            result.append(self.root_node.predictData(i))
        return np.array(result)

    def printTree(self):
        self.root_node.printNode()
