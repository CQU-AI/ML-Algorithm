import numpy as np
from scipy.optimize import minimize


class LinearRegression:
    """ Linear Regressor
    Dependency
    ----------
    scipy :
        - scipy.optimize.minimize
    numpy:
        - numpy.dot
        - numpy.mean
        - numpy.unique
        - numpy.zeros

    Parameters
    ----------
    None
    Methods
    --------
    fit(self, X, y) : Fit the model using X as training data and y as target values
    predict(self, X) : Predict the class labels for the provided data
    """

    @staticmethod
    def cost_gradient(theta, X, y):
        h = np.dot(X, theta) - y
        J = np.dot(h, h) / (2 * X.shape[0])
        grad = np.dot(X.T, h) / X.shape[0]
        return J, grad

    def fit(self, X, y):
        self._class, y = np.unique(y, return_inverse=True)
        y = y * 2 - 1  # map to(-1,1)
        X_mean, y_mean = np.mean(X, axis=0), np.mean(y)
        X_train, y_train = X - X_mean, y - y_mean

        res = minimize(
            fun=LogisticRegression.cost_gradient,
            jac=True,
            x0=np.zeros(X_train.shape[1]),
            args=(X_train, y_train),
            method="L-BFGS-B",
        )

        self.coef_ = res.x
        self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        return self

    def predict(self, X):
        y_pred = (
            (1 / (1 + np.exp(-np.dot(X, self.coef_) - self.intercept_))) > 0.5
        ).astype(int)
        return self._class[y_pred]
