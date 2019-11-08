import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class LogisticRegression:
    """ Binary Logistic regression based on gradient descent (l-bfgs-b) without penalty
    Dependency
    ----------
    scipy :
        - scipy.optimize.minimize
        - scipy.special.expit
    numpy:
        - numpy.dot
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
    def cost_gradient(w, X, y):
        yz = y * (np.dot(X, w[:-1]) + w[-1])
        cost = -np.sum(
            np.vectorize(
                lambda x: -np.log(1 + np.exp(-x))
                if x > 0
                else x - np.log(1 + np.exp(x))
            )(yz)
        ) + 0.5 * np.dot(w[:-1], w[:-1])
        grad = np.zeros(len(w))
        t = (expit(yz) - 1) * y
        grad[:-1] = np.dot(X.T, t) + w[:-1]
        grad[-1] = np.sum(t)
        return cost, grad

    def fit(self, X, y):
        self._class, y = np.unique(y, return_inverse=True)
        y = y * 2 - 1  # map to(-1,1)

        res = minimize(
            fun=LogisticRegression.cost_gradient,
            jac=True,
            x0=np.zeros(X.shape[1] + 1),
            args=(X, y),
            method="L-BFGS-B",
        )
        self.coef_, self.intercept_ = res.x[:-1], res.x[-1]
        return self

    def predict(self, X):
        y_pred = ((np.dot(X, self.coef_) + self.intercept_) > 0).astype(int)
        return self._class[y_pred]
