import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

# logistic function
def logisticFun(x):
    x[x < -500] = -500
    return 1 / (1 + np.exp(-x))

def transformFeature(x, standardize_mean, standardize_std):
    # standardize
    x = (x - standardize_mean) / standardize_std
    # add bias term
    ones_row = np.ones((1, x.shape[1]))
    x = np.vstack((ones_row, x))
    return x

def loadDataSet():
    # read data from file
    df = pd.read_csv('titanic_data.csv')
    # split features and label
    X = df.drop(['Survived'], axis=1)
    y = df['Survived']
    # transform to numpy
    X = X.to_numpy().T
    y = y.to_numpy().reshape((y.shape[0], 1))
    # standardize
    standardize_mean = np.mean(X, axis=1, keepdims=True)
    standardize_std = np.std(X, axis=1, keepdims=True)
    X = (X - standardize_mean) / standardize_std 
    # add bias term
    ones_row = np.ones((1, X.shape[1]))
    X = np.vstack((ones_row, X))
    # split train and test
    sample_number = X.shape[1]
    train_number = int(sample_number * 0.8)
    X_test = X[:, train_number:]
    X_train = X[:, :train_number]
    y_test = y[train_number:]
    y_train = y[:train_number]
    return X, X_train, X_test, y, y_train, y_test, standardize_mean, standardize_std

def computeGradient(X, y, theta):
    # compute gradient
    gradient = X @ (y.T - logisticFun(theta.T @ X)).T
    return gradient

def learningTheta(X_train, X_test, y_train, y_test):
    theta = np.ones(X_train.shape[0]).reshape((X_train.shape[0],1))
    # learning rate
    step = 0.1
    # learning rate's decay rate
    decay_rate = 0.0001
    # convergence condition
    epsilon = 0.0001
    # max iterations
    max_iterations = 10000
    for i in range(max_iterations):
        # print('Iteration', i)
        gradient = computeGradient(X_train, y_train, theta)
        theta = theta + step * gradient
        # print('gradient: ', gradient)
        # print('theta: ', theta)
        if np.linalg.norm(gradient) < epsilon:
            print('Model converge at iteration', i)
            test_result = logisticFun(theta.T @ X_test).reshape((y_test.shape[0], 1))
            test_result[test_result > 0.5] = 1
            test_result[test_result <= 0.5] = 0
            test_accuracy = np.sum(test_result == y_test) / y_test.shape[0]
            print("Test Accuracy: ", test_accuracy)
            break

        step = step / (1 + decay_rate * i)
    
    return theta

def computeLogLikelihood(X, y, theta):
    log_likelihood =  np.log(logisticFun(theta.T @ X)) @ y + np.log(logisticFun(-theta.T @ X)) @ (1 - y)
    return log_likelihood

def predict(x, theta):
    # predict
    log_odds = theta.T @ x
    p = logisticFun(log_odds)
    return p, log_odds

def computeCovarianceOfTheta(X, theta):
    # matrix method
    cov_theta = np.linalg.inv(np.exp(-theta.T @ X) / (1 + np.exp(-theta.T @ X))**2 * X @ X.T)

    # loop method
    # cov_theta = np.zeros((X.shape[0], X.shape[0]))
    # for i in range(X.shape[1]):
    #     vector = X[:, i].reshape((X.shape[0], 1))
    #     cov_theta += np.exp(-theta.T @ vector) / (1 + np.exp(-theta.T @ vector))**2 * vector @ vector.T
    # cov_theta = np.linalg.inv(cov_theta)

    return cov_theta

def confidenceIntervalOfLogOdds(x, X, theta):
    # get the covariance of theta
    cov_theta = computeCovarianceOfTheta(X, theta)
    var = x.T @ cov_theta @ x

    alpha = 0.05
    tau = -norm.ppf(alpha/2, scale=np.sqrt(var))
    return tau

def significanceTest(X, theta):
    cov_theta = computeCovarianceOfTheta(X, theta)
    vector_vj2 = np.diag(cov_theta).reshape((cov_theta.shape[0], 1))

    alpha = 0.05
    chi_value = chi2.ppf(1-alpha, df=1)
    
    return vector_vj2, chi_value

def run():
    # load data
    print("\nProblem 3.1")
    X, X_train, X_test, y, y_train, y_test, standardize_mean, standardize_std = loadDataSet()

    # learning theta
    theta = learningTheta(X_train, X_test, y_train, y_test)
    print('theta: ')
    print(theta)

    # compute log likelihood
    log_likelihood = computeLogLikelihood(X, y, theta)
    print('log_likelihood:\n', log_likelihood)

    # predict
    print("\nProblem 3.3")
    x = np.array([3, 0, 23, 0, 0, 7.75]).reshape((6, 1))
    x = transformFeature(x, standardize_mean, standardize_std)
    probability, log_odds = predict(x, theta)
    print('log odds: ', log_odds)
    print('probability: ', probability)

    # 95% confidence interval
    tau = confidenceIntervalOfLogOdds(x, X, theta)
    print("tau: ", tau)
    print("95% confidence interval of tau: ", (log_odds-tau, log_odds+tau))
    print("95% confidence interval of probability: ", (logisticFun(log_odds-tau), logisticFun(log_odds+tau)))

    # significance test
    print("\nProblem 3.4")
    vector_vj2, chi_value = significanceTest(X, theta)
    print('vector_vj^2:')
    print(vector_vj2)
    print("chi2_value: ")
    print(chi_value)
    print("whether significant:")
    print(theta**2 > chi_value * vector_vj2)
    print("significant level:")
    print(theta**2 - chi_value * vector_vj2)

    # change the most significant feature and test
    new_x = np.array([3, 1, 23, 0, 0, 7.75]).reshape((6, 1))
    new_x = transformFeature(new_x, standardize_mean, standardize_std)
    new_probability, new_log_odds = predict(new_x, theta)
    print('new log odds: ', new_log_odds)
    print('new_probability: ', new_probability)

    # 95% confidence interval
    tau = confidenceIntervalOfLogOdds(new_x, X, theta)
    print("tau: ", tau)
    print("95% confidence interval of log_odds: ", (new_log_odds-tau, new_log_odds+tau))
    print("95% confidence interval of probability: ", (logisticFun(new_log_odds-tau), logisticFun(new_log_odds+tau)))

if __name__ == '__main__':
    run()