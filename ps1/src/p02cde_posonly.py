import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    #######################
    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    theta_0 = np.zeros(shape=(3, ))
    model_t = LogisticRegression(theta_0=theta_0)
    model_t.fit(x_train, t_train)

    # predict using the trained model
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    t_pred = model_t.predict(x_test)

    # Plot decision boundary on top of test set
    util.plot(x_test, t_test, model_t.theta, 'output/2c_{ds}.pdf'.format(ds=test_path.split('/')[-1]))

    # Use np.savetxt to save predictions on eval set to pred_path
    np.savetxt(pred_path, t_pred)

    #######################
    # Part (d): Train on y-labels and test on true labels
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    model_y = LogisticRegression(theta_0=theta_0)
    model_y.fit(x_train, y_train)

    # predict using the trained model
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)
    t_pred = model_t.predict(x_test)

    # Plot decision boundary on top of test set
    util.plot(x_test, y_test, model_y.theta, 'output/2d_{ds}.pdf'.format(ds=test_path.split('/')[-1]))

    #######################
    # Part (e): Apply correction factor using validation set and test on true labels
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    pred_y_valid = model_y.predict(x_valid)
    V_plus_mask = (y_valid == 1)
    alpha = np.mean(pred_y_valid[V_plus_mask])
    
    # Plot decision boundary on top of test set
    util.plot(x_test, y_test, model_y.theta, 'output/2e_{ds}.pdf'.format(ds=test_path.split('/')[-1]), correction=alpha)

    # Plot and use np.savetxt to save outputs
    # *** END CODE HERE



