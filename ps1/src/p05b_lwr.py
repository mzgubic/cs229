import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau=tau)
    model.fit(x_train, y_train)
    # Get MSE value on the validation set
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_valid)
    mse = np.sum((y_pred - y_valid)**2)
    print(mse)
    # Plot validation predictions on top of training set
    fig, ax = plt.subplots()
    ax.scatter(x_train[:,1], y_train, color='b', marker='x')
    ax.scatter(x_valid[:,1], y_pred, color='r', marker='o')
    plt.show()
    # No need to save anything
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        x_mean = np.mean(x[:,1])
        weights = np.e**(-(x[:,1] - x_mean)**2 / (2 * self.tau**2))
        weights = np.diag(weights)

        inv_matrix1 = np.linalg.inv(np.matmul(x.T, np.matmul(weights, x)))
        matrix2 = np.matmul(x.T, np.matmul(weights, y))
        self.theta = np.matmul(inv_matrix1, matrix2)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return np.matmul(x, self.theta)
        # *** END CODE HERE ***

