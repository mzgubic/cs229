import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    m, n = x_train.shape

    # *** START CODE HERE ***
    theta_0 = np.zeros(shape=(n, ))
    model = PoissonRegression(theta_0=theta_0, step_size=lr)

    # Fit a Poisson Regression model
    model.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_eval, y_eval= util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)

    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape

        # keep track of old theta to track convergence
        old_theta = self.theta - 1.0

        # run until convergence
        while np.linalg.norm(old_theta - self.theta) > self.eps:

            #######################
            # the loop way
            want_slow = False
            if want_slow:
                delta_loss = np.zeros(n)
                for i in range(m):
                    x_i = x[i]
                    y_i = y[i]
                    eta_i = np.e**(np.dot(self.theta, x_i))
                    delta_loss += 1.0/m * x_i * (y_i - eta_i)
            #######################

            #######################
            # the vectorised way
            diff = (y - np.e**np.matmul(x, self.theta))
            delta_loss = 1.0/m * np.sum(np.multiply(diff[:,np.newaxis], x), axis=0)
            #######################

            # update theta
            old_theta = self.theta
            self.theta = self.theta + self.step_size * delta_loss 

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.e**(np.matmul(x, self.theta))

        # *** END CODE HERE ***
