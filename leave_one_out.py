import pandas as pd
import numpy as np


class RhoLogisticRegression:

    def __init__(self, X_train, y_train, lamd: float, rho, eps: float, max_iterations=1000):

        self.X_train = X_train
        self.y_train = y_train
        self.lamd = lamd
        self.rho = rho
        self.max_iterations = max_iterations
        self.eps = eps

    def train(self, init_beta=None):
        self.d_train, self.n_train = self.X_train.shape
        self.init_learning_rate = self.__compute_init_learning_rate__()
        self.beta, self.cost_history_fastgrad, self.beta_history = self.fast_grad_descent(init_beta)

    def obj(self, beta):
        a = np.exp(-self.rho * self.y_train * (self.X_train.T.dot(beta)))
        b = np.sum(np.log(1 + a))
        c = self.lamd * np.square(np.linalg.norm(beta))
        cost = (b / (self.n_train * self.rho) + c)
        return cost.squeeze()

    def compute_grad(self, beta):
        x = self.rho * self.y_train * (self.X_train.T.dot(beta))
        a = (1 / (1 + np.exp(x)))
        p = np.diag(np.array(a).reshape(self.n_train, ))
        grad = -1 / self.n_train * (self.X_train.dot(p).dot(self.y_train)) + 2 * self.lamd * beta
        return grad

    def __compute_init_learning_rate__(self):
        eigenvalues, eigenvectors = np.linalg.eigh(1 / self.n_train * self.X_train.T.dot(self.X_train))
        lipschitz = max(eigenvalues) + self.lamd
        return 1 / lipschitz

    def bt_line_search(self, beta, alpha=0.5, gamma=0.8, max_iter=100):
        learning_rate = self.init_learning_rate
        grad = self.compute_grad(beta)
        z = beta - learning_rate * grad
        lhs = self.obj(z)
        rhs = self.obj(beta) - alpha * learning_rate * np.square(np.linalg.norm(grad))
        i = 0
        while rhs < lhs and i < max_iter:
            learning_rate *= gamma
            z = beta - learning_rate * grad
            lhs = self.obj(z)
            rhs = self.obj(beta) - alpha * learning_rate * np.square(np.linalg.norm(grad))
            i += 1
        return learning_rate

    def fast_grad_descent(self, init_beta=None):

        if init_beta is None:
            beta = np.zeros((self.d_train, 1))
            theta = np.zeros((self.d_train, 1))

        else:
            beta = init_beta
            theta = init_beta

        cost_history_fastgrad = []
        error_train = []
        error_val = []
        beta_history = np.array(beta)

        for it in range(self.max_iterations):
            theta_grad = self.compute_grad(theta)
            error = np.linalg.norm(theta_grad)
            if error > self.eps:
                cost_history_fastgrad.append(self.obj(beta))
                t = self.bt_line_search(beta)
                beta = theta - (t * theta_grad)
                theta = beta + it / (it + 3) * (beta - beta_history[:, (it)].reshape(self.d_train, 1))
                beta_history = np.append(beta_history, beta, axis=1)
            else:
                break

        return beta, cost_history_fastgrad, beta_history

    def predict(self, X, beta):
        pred = 1 / (1 + np.exp(-X.T.dot(beta))) > 0.5
        predictions = pred * 2 - 1
        return predictions

    def classification_error(self, X, y, beta):
        predictions = self.predict(X, beta)
        error = np.mean(predictions != y) * 100
        return error


from concurrent.futures import ProcessPoolExecutor


def get_spam_data():
    spam_data = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data', sep=" ", header=None)
    predictors = spam_data.drop(columns=57)
    response = spam_data.iloc[:, 57]
    response = np.where(response == 0, -1, 1)

    return predictors, response


def __run_sample__(rho_logistic_model, init_beta, counter, X_test, y_test):
    rho_logistic_model.train()
    error_i = rho_logistic_model.classification_error(X_test, y_test, rho_logistic_model.beta)
    print('Done training iteration {}'.format(counter))
    return error_i


def get_leave_one_out_first_itr(X, y, i, d, n):
    X = np.asarray(X)
    y = np.asarray(y).reshape(n, 1)
    X_train = np.delete(X, i, axis=1)
    y_train = np.delete(y, i, axis=0)
    X_test = X[:, i].reshape(d, 1)
    y_test = y[i, :].reshape(1, 1)
    return X_train, y_train, X_test, y_test


def leave_one_out(X, y):
    X = np.asarray(X)
    y = np.asarray(y)

    d, n = X.shape
    lambda_list = [0.003,0.007,0.02,0.05,0.08,0.1,0.2,0.5,2.0,3.0]

    misclassification_error = {}
    executor = ProcessPoolExecutor(max_workers=10)
    beta_li = None

    for li, p in enumerate(lambda_list):

        print('Iteration for lambda = ' + str(p)

        misclassification_errors_futures = []

        X_train, y_train, X_test, y_test = get_leave_one_out_first_itr(X, y, 0, d, n)
        model = RhoLogisticRegression(X_train, y_train, lamd=lamd, rho=2, eps=0.01)
        if beta_li is None:
            model.train()

        else:
            model.train(beta_li)

        beta_li = model.beta

        for i in range(1, n):
            X_train, y_train, X_test, y_test = get_leave_one_out_first_itr(X, y, i, d, n)
            model = RhoLogisticRegression(X_train, y_train, lamd=lamd, rho=2, eps=0.01)
            future_misclass_error_i = executor.submit(__run_sample__, model, beta_li, i, X_test, y_test)
            misclassification_errors_futures.append(future_misclass_error_i)

        misclassification_errors_lambd = []
        for f in misclassification_errors_futures:
            error_i = f.result()
            misclassification_errors_lambd.append(error_i)

        misclassification_error[lamd] = np.mean(misclassification_errors_lambd)

    executor.shutdown()

    return misclassification_error


if __name__ == '__main__':
    X, y = get_spam_data()
    X = (X - np.mean(X)) / (np.std(X) + 0.001)
    X = X.T
    X = np.asarray(X)
    y = np.asarray(y)
    X = X[0:500]
    y = y[0:500]
    lambda_error = leave_one_out(X, y)
    print(lambda_error)
