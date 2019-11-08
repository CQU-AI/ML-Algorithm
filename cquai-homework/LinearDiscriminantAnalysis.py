import numpy as np


class LinearDiscriminantAnalysis:
    """ Linear Discriminant Analysis
    Dependency
    ----------
    numpy:
        - numpy.dot
        - numpy.mean
        - numpy.unique
        - numpy.linalg.inv

    Parameters
    ----------
    None
    Methods
    --------
    fit(self, X, y) : Fit the model using X as training data and y as target values
    predict(self, X) : Predict the class labels for the provided data
    """

    def fit(self, X, y):
        self._class, y = np.unique(y, return_inverse=True)
        X_p, X_n = X[y == 1], X[y == 0]
        X_p_mean, X_n_mean = X_p.mean(), X_n.mean()

        # 类内散度
        sw = np.array(
            (X_p - X_p_mean).T.dot(X_p - X_p_mean)
            + (X_n - X_n_mean).T.dot(X_n - X_n_mean)
        )

        # 权重判别
        self.w_ = np.dot((X_n_mean - X_p_mean), (np.linalg.inv(sw)))

        # 样本集中心
        self.X_mean_ = (
            np.dot((np.dot(X_p_mean - X_n_mean, np.mat(sw).I)), (X_p_mean + X_n_mean))
            * 0.5
        )

        return self

    def predict(self, X):
        y_pred = ((np.dot(X, self.w_) + self.X_mean_) < 0).astype(int).T
        return self._class[y_pred]
