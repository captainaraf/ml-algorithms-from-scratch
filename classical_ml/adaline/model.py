import numpy as np 

class AdalineGD:
    def __init__(self, learning_rate=1e-3, n_iter=100, random_state=42):
        self.learning_rate = learning_rate
        self.n_iter = n_iter 
        self.random_state = random_state
        self.random_generator = np.random.RandomState(self.random_state)
        
    def fit(self, X, y):
        self.w_ = self.random_generator.normal(
            loc=0.0, scale=0.1, size=X.shape[1]
        )
        self.b_ = 0.0

        for _ in range(self.n_iter):
            prediction = self.net_input(X)
            outputs = self.activation(prediction)
            errors = (y - outputs)
            self.w_ += self.learning_rate * 2.0 * (X.T @ errors) / X.shape[0]
            self.b_ += self.learning_rate * 2.0 * errors.mean()

    def activation(self, X):
        return X

    def net_input(self, X): 
        return X @ self.w_ + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.5, 1, 0)

class AdalineSGD:
    def __init__(self, learning_rate=1e-3, n_iter=100, random_state=42, shuffle=False):
        self.learning_rate = learning_rate
        self.n_iter = n_iter 
        self.random_state = random_state
        self.random_generator = np.random.RandomState(self.random_state)
        self.w_initialized = False
        self.shuffle = shuffle
        
    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            for xi, target in zip(X, y):
                self._update_weights(xi, target)

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])

        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        
        else:
            self._update_weights(X, y)

    def activation(self, X):
        return X

    def net_input(self, X): 
        return X @ self.w_ + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.5, 1, 0)

    def _initialize_weights(self, m):
        self.w_ = self.random_generator.normal(
            loc=0.0, scale=0.1, size=m
        )
        self.b_ = 0.0
        self.w_initialized = True
    
    def _shuffle(self, X, y):
        r = self.random_generator.permutation(len(y))
        return X[r], y[r]

    def _update_weights(self, X, y):
        prediction = self.net_input(X)
        outputs = self.activation(prediction)
        error = (y - outputs)
        self.w_ += self.learning_rate * 2.0 * error * X
        self.b_ += self.learning_rate * 2.0 * error
