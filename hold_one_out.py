from concurrent.futures import ProcessPoolExecutor
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

    def train(self):
        self.d_train, self.n_train = self.X_train.shape
        self.init_learning_rate = self.__compute_init_learning_rate__()
        self.beta, self.cost_history_fastgrad, self.beta_history = self.fast_grad_descent()

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

    def fast_grad_descent(self):

        beta = np.zeros((self.d_train, 1))
        theta = np.zeros((self.d_train, 1))
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


def __run_sample__(rho_logistic_model,lambd,counter, X_test, y_test):
    rho_logistic_model.train()
    error_i = rho_logistic_model.classification_error(X_test, y_test, rho_logistic_model.beta)
    print('Done training iteration {}'.format(counter))
    return lambd, np.squeeze(error_i)


def hold_out(X_train, y_train, X_test, y_test):
    lamda = np.linspace(-4, 4, 10)

    executor = ProcessPoolExecutor (max_workers = 40)

    misclassfication_error_futures = []

    for i, lamda in enumerate(lamda):

        lam = np.power(10, lamda).squeeze()

        model = RhoLogisticRegression(X_train, y_train, lamd = lam, rho=2, eps=0.001)
        missclassfication_error_futures_i = executor.submit(__run_sample__,model,lam,i,X_test,y_test)
        misclassfication_error_futures.append(missclassfication_error_futures_i)

    executor.shutdown()

    misclassfication_errors_lambd = {}
    for f in misclassfication_error_futures:
        *lambduh ,error = f.result()
        print(f.result())
        misclassfication_errors_lambd[lambduh] = error
    return misclassfication_errors_lambd

def create_binary_classes(train):
    pairs_of_classes = []
    for i in range(0, 10):
        for j in range(i + 1, 10):
            d = str(i)
            e = str(j)
            train_d_e = train.loc[(train['Labels'] == i) | (train['Labels'] == j)]
            pairs_of_classes.append((train_d_e.drop('Labels', axis=1),
                                     np.where(train_d_e[['Labels']] == i, -1, 1),
                                     (i, j)))

    return pairs_of_classes



if __name__ == '__main__':
    train_features = np.load("/data_competition1_files/train_features.npy")
    train_labels = np.load("/data_competition1_files/train_labels.npy")
    
    
    
    train_features_df = pd.DataFrame(train_features)
    train_labels_df = pd.DataFrame(train_labels)
    train_labels_df.columns = ['Labels']
    
    frames = [train_labels_df, train_features_df]
    train = pd.concat(frames, axis=1)
    
    pairs = create_binary_classes(train)
    lambda_error_pair = {}
    for p in pairs:

        X_trainval, y_trainval, pair_name = p
        X_train, y_train, X_val, y_val = train_test_split(X_trainval, y_trainval , test_size=0.20 , random_state=42)
        print("Choosing pair: ",pair_name)
        misclassfication_errors_lambd = hold_out( X_train.T, y_train, X_val.T, y_val)
        print(misclassfication_errors_lambd)
        lambda_error_pair[pair_name] = misclassfication_errors_lambd

    print(lambda_error_pair)
