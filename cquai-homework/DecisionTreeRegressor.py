import numpy as np


class TreeNode:
    def __init__(
        self, feature=None, threshold=None, impurity=None, n_node=None, value=None
    ):
        self.left_child = self.right_child = -1
        self.feature = feature
        self.threshold = threshold
        self.impurity = impurity
        self.n_node = n_node
        self.value = value


class DecisionTreeRegressor:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self._nodes = []

    def _build_leaf(self, X, y, cur_depth, parent, is_left):
        self._nodes.append(
            TreeNode(
                impurity=np.mean(np.square(y)) - np.square(np.mean(y)),
                n_node=X.shape[0],
                value=np.mean(y),
            )
        )
        self._set_parent(parent, is_left, len(self._nodes) - 1)
        return

    def _set_parent(self, parent, is_left, child_ind):
        if parent is not None:
            if is_left:
                self._nodes[parent].left_child = len(self._nodes) - 1
            else:
                self._nodes[parent].right_child = len(self._nodes) - 1
        return

    def _build_tree(self, X, y, cur_depth, parent, is_left):
        if cur_depth == self.max_depth:
            self._build_leaf(X, y, cur_depth, parent, is_left)

        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left_ind = best_right_ind = None

        sum_all = np.sum(y)
        step = lambda x, y, a: (x + a, y - a)

        for i in range(X.shape[1]):  # for features
            sum_left, sum_right = 0, sum_all
            n_left = 0
            n_right = X.shape[0]
            ind = np.argsort(X[:, i])

            for j in range(ind.shape[0] - 1):  # for all sample
                # step by step
                sum_left, sum_right = step(sum_left, sum_right, y[ind[j]])
                n_left, n_right = step(n_left, n_right, 1)

                cur_gain = (
                    sum_left * sum_left / n_left + sum_right * sum_right / n_right
                )
                if cur_gain > best_gain:  # found better choice
                    best_gain = cur_gain
                    best_feature = i
                    best_threshold = X[ind[j], i]
                    best_left_ind, best_right_ind = ind[: j + 1], ind[j + 1 :]

        self._nodes.append(
            TreeNode(
                feature=best_feature,
                threshold=best_threshold,
                impurity=np.mean(np.square(y)) - np.square(np.mean(y)),
                n_node=X.shape[0],
                value=np.mean(y),
            )
        )
        cur_id = len(self._nodes) - 1
        self._set_parent(parent, is_left, cur_id)

        if cur_depth < self.max_depth:
            self._build_tree(
                X[best_left_ind], y[best_left_ind], cur_depth + 1, cur_id, True
            )
            self._build_tree(
                X[best_right_ind], y[best_right_ind], cur_depth + 1, cur_id, False
            )

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self._nodes = []
        self._build_tree(X, y, 0, None, None)
        return self

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            cur_node = 0
            while self._nodes[cur_node].left_child != -1 and self._nodes[cur_node].threshold is not None:  # search in tree
                if (
                    X[i][self._nodes[cur_node].feature]
                    <= self._nodes[cur_node].threshold
                ):
                    cur_node = self._nodes[cur_node].left_child
                else:
                    cur_node = self._nodes[cur_node].right_child
            pred[i] = self._nodes[cur_node].value
        return pred
