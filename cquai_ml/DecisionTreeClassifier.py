# python3
# -*- coding: utf-8 -*-
# @File    : DecisionTreeClassifier.py
# @Desc    :
# @Project : ML-Algorithm
# @Time    : 11/8/19 10:48 PM
# @Author  : Loopy
# @Contact : peter@mail.loopy.tech
# @License : CC BY-NC-SA 4.0 (subject to project license)

import numpy as np


class TreeNode:
    def __init__(
            self, feature=None, threshold=None, impurity=None, n_node=None, value=None
    ):
        self.left_child = -1
        self.right_child = -1
        self.feature = feature
        self.threshold = threshold
        self.impurity = impurity
        self.n_node = n_node
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth

    def _entropy(self, y_cnt):
        prob = y_cnt / np.sum(y_cnt)
        prob = prob[prob > 0]
        return -np.sum(prob * np.log(prob))

    @staticmethod
    def _inti(X, f):
        _, counts = np.unique(X[:, f], return_counts=True)
        ave = counts / counts.sum()
        return (np.log(ave) * ave).sum()

    def _build_leaf(self, X, y, cur_depth, parent, is_left):
        value = np.bincount(y, minlength=self.n_classes_)
        self._nodes.append(TreeNode(n_node=X.shape[0], value=value, impurity=value))
        self._set_parent(parent, is_left, len(self._nodes) - 1)
        return

    def _set_parent(self, parent, is_left, child_ind):
        if parent is not None:
            if is_left:
                self._nodes[parent].left_child = child_ind
            else:
                self._nodes[parent].right_child = child_ind
        return

    def _build_tree(self, X, y, cur_depth, parent, is_left):
        if cur_depth == self.max_depth:
            self._build_leaf(X, y, cur_depth, parent, is_left)
            return

        best_improvement = -np.inf
        best_feature = None
        best_threshold = None
        best_left_ind = None
        best_right_ind = None

        y_cnt = np.bincount(y, minlength=self.n_classes_)
        for i in range(X.shape[1]):
            inti = self._inti(X, i)
            ind = np.argsort(X[:, i])
            y_cnt_left, y_cnt_right = np.bincount([], minlength=self.n_classes_), y_cnt.copy()
            n_left, n_right = 0, X.shape[0]

            for j in range(ind.shape[0] - 1):
                y_cnt_left[y[ind[j]]] += 1
                y_cnt_right[y[ind[j]]] -= 1
                n_left += 1
                n_right -= 1
                if j + 1 < ind.shape[0] - 1 and np.isclose(
                        X[ind[j], i], X[ind[j + 1], i]
                ):
                    continue
                cur_improvement = (n_left * self._entropy(y_cnt_left) + n_right * self._entropy(y_cnt_right)) / inti

                if cur_improvement > best_improvement:
                    best_improvement = cur_improvement
                    best_feature = i
                    best_threshold = X[ind[j], i]
                    best_left_ind = ind[: j + 1]
                    best_right_ind = ind[j + 1:]

        self._nodes.append(
            TreeNode(
                feature=best_feature,
                threshold=best_threshold,
                n_node=X.shape[0],
                value=y_cnt,
                impurity=self._entropy(y_cnt),
            )
        )

        self._set_parent(parent, is_left, len(self._nodes) - 1)
        if cur_depth < self.max_depth:
            self._build_tree(
                X[best_left_ind],
                y[best_left_ind],
                cur_depth + 1,
                len(self._nodes) - 1,
                True,
            )
            self._build_tree(
                X[best_right_ind],
                y[best_right_ind],
                cur_depth + 1,
                len(self._nodes) - 1,
                False,
            )

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.classes_, y_train = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self._nodes = []
        self._build_tree(X, y_train, 0, None, None)
        return self

    def predict(self, X):
        pred = np.zeros(X.shape[0], dtype=np.int)
        for i in range(X.shape[0]):
            cur_node = 0
            while self._nodes[cur_node].left_child != -1:
                if (
                        X[i][self._nodes[cur_node].feature]
                        <= self._nodes[cur_node].threshold
                ):
                    cur_node = self._nodes[cur_node].left_child
                else:
                    cur_node = self._nodes[cur_node].right_child
            pred[i] = cur_node
        return np.array([self.classes_[np.argmax(self._nodes[p].value)] for p in pred])


from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier as skDecisionTreeClassifier

if __name__ == '__main__':
    X, y = load_breast_cancer(return_X_y=True)
    clf1 = DecisionTreeClassifier(max_depth=1).fit(X, y)
    clf2 = skDecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)

    pred1 = clf1.predict(X)
    pred2 = clf2.predict(X)
    print(pred1)
