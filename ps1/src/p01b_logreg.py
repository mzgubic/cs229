import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # initial guess of parameters
    theta_0 = np.zeros(shape=(3, ))

    # get the model
    model = LogisticRegression(theta_0=theta_0)
    model.fit(x_train, y_train)

    # predict using the trained model
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)

    # Plot decision boundary on top of validation set set
    util.plot(x_eval, y_eval, model.theta, 'output/{ds}_log_reg.pdf'.format(ds=eval_path.split('/')[-1]))

    # Use np.savetxt to save predictions on eval set to pred_path
    np.savetxt(pred_path, y_pred)


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        m = x.shape[0]
        n = x.shape[1]

        # keep track of old theta to track convergence
        old_theta = self.theta - 1.0

        # run until convergence
        while np.linalg.norm(old_theta - self.theta) > self.eps:

            # compute the d_loss 
            g = self.sigmoid(np.matmul(x, self.theta))
            d_loss = np.sum(np.multiply(x.T, -1.0/m * (y * (1.0-g) - (1.0-y) * g)), axis=1)

            # compute the hessian
            stack_of_columns = x[:, :, np.newaxis]
            stack_of_rows = x[:, np.newaxis, :]
            stack_of_outer_products = np.matmul(stack_of_columns, stack_of_rows)
            col = 1.0/m * g * (1.0-g)
            hessian_vector = np.multiply(col[:, np.newaxis, np.newaxis], stack_of_outer_products)
            hessian = np.sum(hessian_vector, axis=0)

            # update thetas
            inv_hessian = np.linalg.inv(hessian)
            old_theta = self.theta
            self.theta = self.theta + self.step_size * np.matmul(inv_hessian, d_loss)


    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return self.sigmoid(np.matmul(x, self.theta)) 
        # *** END CODE HERE ***



