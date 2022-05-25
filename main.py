import random


def mse(Y_true, Y_pred):
    n = len(Y_true)
    loss = 0
    for y_true, y_pred in zip(Y_true, Y_pred):
        loss += (y_true-y_pred)**2
    loss = (1/n)*loss
    return loss


def make_regression(n=100) -> tuple[list[float], list[float]]:
    a0 = random.random()
    a1 = random.random()
    h = lambda x: a1*x+a0
    X = [random.random() for _ in range(n)]
    Y = [h(x) for x in X]
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

    def _h(self, x):
        return self.a1_*x+self.a0_

    def fit(self, X, Y, epochs=10):
        n = len(X)
        self.a0_ = random.random()
        self.a1_ = random.random()

        for epoch in range(epochs):
            ga0 = 0
            ga1 = 0
            running_loss = 0
            for x, y in zip(X, Y):
                y_pred = self._h(x)
                running_loss += (y_pred-y)**2
                ga0 += y_pred-y
                ga1 += (y_pred-y)*x
            self.a0_ = self.a0_-self.lr*(2/n)*ga0
            self.a1_ = self.a1_-self.lr*(2/n)*ga1
            running_loss = running_loss/n
            print(f"({epoch+1}) loss: {running_loss:.2f}")

    def predict(self, X):
        Y = [self._h(x) for x in X]
        return Y


X, Y = make_regression()
X_train, X_test, Y_train, Y_test = data_split(X, Y, [0.8, 0.2])
print(len(X_train), len(X_test), len(Y_train), len(Y_test))
reg = LinearRegression()
reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_train)
loss = mse(Y_train, Y_pred)
print(loss)
