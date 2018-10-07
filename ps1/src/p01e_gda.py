import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # get the model
    model = GDA()
    model.fit(x_train, y_train)

    # predict using the trained model
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    y_pred = model.predict(x_eval)

    # Plot decision boundary on top of validation set set
    theta = list(model.theta)
    theta_0 = [model.theta_0]
    util.plot(x_eval, y_eval, theta_0+theta, 'output/{ds}_GDA.pdf'.format(ds=eval_path.split('/')[-1]))

    # Use np.savetxt to save predictions on eval set to pred_path
    np.savetxt(pred_path, y_pred)


class GDA(LinearModel):
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        m = x.shape[0]
        # Find phi, mu_0, mu_1, and sigma
        # phi
        phi = np.sum(y==1) / m
        # mu0, mu1
        mu_0 = np.sum(x[y==0, :], axis=0) / np.sum(y==0)
        mu_1 = np.sum(x[y==1, :], axis=0) / np.sum(y==1)
        # sigma
        mu_y = np.zeros(shape=x.shape)
        mu_y[y==0] = mu_0
        mu_y[y==1] = mu_1
        diff = x - mu_y
        stack_of_columns = diff[:, :, np.newaxis]
        stack_of_rows = diff[:, np.newaxis, :]
        stack_of_outer_products = np.matmul(stack_of_columns, stack_of_rows)
        sigma = 1.0/m * np.sum(stack_of_outer_products, axis=0) 
        inv_sigma = np.linalg.inv(sigma)

        # Write theta in terms of the parameters
        self.theta = np.matmul(inv_sigma, mu_1-mu_0)
        self.theta_0 = 0.5 * ( np.dot(mu_0, np.dot(inv_sigma, mu_0)) - np.dot(mu_1, np.dot(inv_sigma, mu_1)) )
        self.theta_0 -= np.log((1.0-phi) / phi)

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        theta_x = np.matmul(x, self.theta)
        return self.sigmoid(theta_x + self.theta_0)







