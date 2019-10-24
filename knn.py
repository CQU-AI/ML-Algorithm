import scipy
from sklearn.neighbors import BallTree, KDTree


class KNeighborsClassifier:
    """ 一个仿scikit-learn接口的微型knn实现。
    Ｄependency
    ----------
    `scipy` : 考虑到性能，使用scipy来完成底层运算
        - `scipy.unique`
        - `scipy.spatial.distance.cdist`
        - `scipy.argsort`
        - `scipy.stats.mode`
    `sklearn.neighbors` ： 由于时间有限暂时还没有实现的数据结构(TODO: implement of BallTree & KDTree)
        - `sklearn.neighbors.BallTree`
        - `sklearn.neighbors.KDTree`
    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    algorithm : {'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.
    Methods
    --------
    __init__(n_neighbors=5, algorithm=’brute’, leaf_size=30) : Init
    fit(self.X,y) : Fit the model using X as training data and y as target values
    predict(self, X) : Predict the class labels for the provided data
    Examples
    --------
    >>> X = [[0], [1], [2], [3]]
    >>> y = [0, 0, 1, 1]
    >>> neigh = KNeighborsClassifier(n_neighbors=3)
    >>> neigh.fit(X, y)
    KNeighborsClassifier(...)
    >>> print(neigh.predict([[1.1]]))
    [0]
    >>> print(neigh.predict_proba([[0.9]]))
    [[0.66666667 0.33333333]]
    """

    class KNeighborsClassifier:
        def __init__(self, n_neighbors=5, algorithm="brute", leaf_size=30):
            self.n_neighbors = n_neighbors
            self.algorithm = algorithm
            self.leaf_size = leaf_size

        def fit(self, X, y):
            self._fit_X = X
            self.classes_, self._fit_y = scipy.unique(y, return_inverse=True)

            if self.algorithm == "brute":
                pass
            elif self.algorithm == "kd_tree":
                self.tree = KDTree(X, leaf_size=self.leaf_size)
            elif self.algorithm == "ball_tree":
                self.tree = BallTree(X, leaf_size=self.leaf_size)
            else:
                raise ValueError("unrecognized algorithm: ", str(self.algorithm))
            return self

        def predict(self, X):
            if self.algorithm == "brute":
                dist_mat = scipy.spatial.distance.cdist(X, self._fit_X)
                neighbors_ind = scipy.argsort(dist_mat, axis=1)[:, : self.n_neighbors]
            elif self.algorithm in ["kd_tree", "ball_tree"]:
                dist_mat, neighbors_ind = self.tree.query(X, k=self.n_neighbors)
            else:
                raise ValueError("unrecognized algorithm: ", str(self.algorithm))
            neighbors_y = scipy.stats.mode(self._fit_y[neighbors_ind], axis=1)[
                0
            ].flatten()
            return self.classes_[neighbors_y]
