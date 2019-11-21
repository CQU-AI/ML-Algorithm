import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))


def nn_bpa(
        X, y, learning_rate=0.1, stop_loss=0.001, max_iteration=1000, num_of_layer=2
):
    V = np.random.rand(X.shape[1], num_of_layer)
    W = np.random.rand(num_of_layer, y.shape[0])
    t0 = np.random.rand(1, num_of_layer)
    t1 = np.random.rand(1, y.shape[0])

    for i in range(max_iteration):
        b = sigmoid(X.dot(V) - t0)
        Y0 = sigmoid(b.dot(W) - t1)
        loss = np.sum((Y0 - y) ** 2) / X.shape[0]
        if loss < stop_loss or i > max_iteration:
            break

        g = Y0 * (1 - Y0) * (y - Y0)
        e = b * (1 - b) * g.dot(W.T)
        W += learning_rate * b.T.dot(g)
        t1 -= learning_rate * g.sum(axis=0)
        V += learning_rate * X.T.dot(e)
        t0 -= learning_rate * e.sum(axis=0)

        if i % 50 == 0:
            print("Iteration : {i} loss = {loss}".format(i=i, loss=loss))

    return V, W, t0, t1


def nn_bps(
        X, y, learning_rate=0.1, stop_loss=0.001, max_iteration=1000, num_of_layer=2
):
    V = np.random.rand(X.shape[1], num_of_layer)
    W = np.random.rand(num_of_layer, y.shape[0])
    t0 = np.random.rand(1, num_of_layer)
    t1 = np.random.rand(1, y.shape[0])

    for i in range(max_iteration):
        for k in range(X.shape[0]):
            b = sigmoid(X.dot(V) - t0)
            Y0 = sigmoid(b.dot(W) - t1)
            loss = np.sum((Y0 - y) ** 2) / X.shape[0]
            if loss < stop_loss or i > max_iteration:
                return V, W, t0, t1

            g = Y0[k] * (1 - Y0[k]) * (y[k] - Y0[k])
            g = g.reshape(1, g.size)
            b = b[k]
            b = b.reshape(1, b.size)
            e = b * (1 - b) * g.dot(W.T)
            W += learning_rate * b.T.dot(g)
            t1 -= learning_rate * g
            V += learning_rate * X[k].reshape(1, X[k].size).T.dot(e)
            t0 -= learning_rate * e

        if i % 50 == 0:
            print("Iteration : {i} loss = {loss}".format(i=i, loss=loss))


if __name__ == "__main__":
    from cquai_ml import load_melon

    X, y = load_melon(return_X_y=True, return_array=True)
    nn_bpa(X, y)
    nn_bps(X, y)


