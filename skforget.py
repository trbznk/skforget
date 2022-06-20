# TODO: Polynomial Regression
# TODO: Ridge Regression
# TODO: Lasse Regression
# TODO: Elastic Net

import random
import math


def mse(Y_true, Y_pred):
    n = len(Y_true)
    loss = 0
    for y_true, y_pred in zip(Y_true, Y_pred):
        loss += (y_true-y_pred)**2
    loss = (1/n)*loss
    return loss


def mean(l):
    return sum(l)/len(l)


def std(l):
    return math.sqrt(var(l))


def var(l):
    mu = mean(l)
    return mean([(x-mu)**2 for x in l])


def transpose(l):
    lt = []
    for col in range(shape(l)[1]):
        lt.append([l[i][col] for i in range(len(l))])
    return lt


def standardize(l):
    if len(shape(l)) == 1:
        mu = mean(l)
        s = std(l)
        return [(x-mu)/(s+1e-64) for x in l]
    elif len(shape(l)) == 2:
        lt = transpose(l)
        for i in range(len(lt)):
            lt[i] = standardize(lt[i])
        return transpose(lt)


def linear_function(x, a):
    y = a[0]
    for i in range(len(x)):
        y += a[i+1]*x[i]
    return y


def shape(l):
    s = []
    temp_l = l
    while isinstance(temp_l, list):
        s.append(len(temp_l))
        temp_l = temp_l[0]
    return tuple(s)


def make_regression(n_samples=100, n_features=1, noise_std=5):
    a = [random.uniform(-100, 100) for _ in range(n_features+1)] # +1 for bias
    X, Y = [], []
    for sample_number in range(n_samples):
        x = [random.gauss(0, 1) for _ in range(n_features)]
        y = linear_function(x, a)
        noise = random.gauss(0, noise_std)
        y += noise
        X.append(x)
        Y.append(y)
    return X, Y


def data_split(X, Y, sizes):
    if sum(sizes) != 1:
        raise ValueError(f"Sum of the sizes is {sum(sizes)} must be 1")
    n = len(X)
    indices = [i for i in range(len(X))]
    splits_indices = []
    random.shuffle(indices)
    for i, size in enumerate(sizes):
        samples = int(n*size)
        if i == len(sizes)-1:
            split_indices = indices
        else:
            split_indices = indices[:samples]
            del indices[:samples]
        splits_indices.append(split_indices)

    x_splits = []
    y_splits = []
    for indices in splits_indices:
        x_split, y_split = [], []
        for idx in indices:
            x_split.append(X[idx])
            y_split.append(Y[idx])
        x_splits.append(x_split)
        y_splits.append(y_split)

    return x_splits+y_splits


class LinearRegression:
    def __init__(self, lr=0.1):
        self.lr = lr

    def fit(self, X, Y, epochs=100):
        n = len(X)
        n_features = len(X[0])
        self.a_ = [random.gauss(0, 1) for _ in range(n_features+1)]
        
        last_loss = None
        for epoch in range(epochs):
            ga = [0]*len(self.a_)
            running_loss = 0
            for x, y in zip(X, Y):
                y_pred = linear_function(x, self.a_)
                running_loss += (y_pred-y)**2
                ga[0] += y_pred-y
                for i in range(1, len(ga)):
                    ga[i] += (y_pred-y)*x[i-1]

            for i in range(len(self.a_)):
                self.a_[i] = self.a_[i]-self.lr*(2/n)*ga[i]
            running_loss = running_loss/n
            print(f"({epoch+1}) loss: {running_loss:.2f}")

    def predict(self, X):
        Y = [linear_function(x, self.a_) for x in X]
        return Y


"""X, Y = make_regression(n_samples=1000, n_features=10)
X_train, X_test, Y_train, Y_test = data_split(X, Y, [0.8, 0.2])
print(shape(X_train), shape(X_test), shape(Y_train), shape(Y_test))
reg = LinearRegression()
reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_train)
loss = mse(Y_train, Y_pred)
print(f"MSE: {loss:.2f}")"""
