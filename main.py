import random


def make_regression(n=100) -> tuple[list[float], list[float]]:
    a1 = random.random()
    a0 = random.random()
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
    def __init__(self):
        pass

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass


X, Y = make_regression()
X_train, X_test, Y_train, Y_test = data_split(X, Y, [0.8, 0.2])
print(len(X_train), len(X_test), len(Y_train), len(Y_test))
reg = LinearRegression()
reg.fit(X, Y)
