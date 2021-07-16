from tinyml.core import Backend as np


def polynomial_regression(X, y, degree=1):
    N = len(X)
    matX = [np.ones(N)]
    for power in range(degree):
        matX.append(np.power(X, power + 1))
    matX = np.column_stack(matX)
    A = np.linalg.pinv(matX.T @ matX)
    D = A @ matX.T
    return D @ y


def polynomial_regression_predict(degree, coeff, X):
    y_pred = 0
    for power in range(degree):
        y_pred += coeff[power + 1] * np.power(X, power + 1)
    y_pred + coeff[0]
    return y_pred
