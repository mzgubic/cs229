import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    mses = {}
    models = {}
    y_preds = {}
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_valid)
        models[tau] = model
        y_preds[tau] = y_pred
        mses[tau] = np.sum((y_pred - y_valid)**2)

    best_tau = min(mses, key=lambda x: mses[x])
    best_valid_mse = mses[best_tau]

    # Fit a LWR model with the best tau value
    
    # Run on the test set to get the MSE value
    y_pred_test = models[best_tau].predict(x_test)
    test_mse = np.sum((y_pred_test - y_test)**2)
    print(best_valid_mse, test_mse)

    # Save test set predictions to pred_path
    np.savetxt(pred_path, y_pred_test)

    # Plot data
    fig, ax = plt.subplots()
    ax.scatter(x_train[:,1], y_train, color='b', marker='x')
    markers = dict(zip(tau_values, ['+', 'o', '*', 'v', 'x', '1']))
    for tau in tau_values:
        ax.scatter(x_valid[:,1], y_preds[tau], color='r', marker=markers[tau], label=tau)
    ax.legend(loc='best')
    plt.show()


    # *** END CODE HERE ***


