import numpy as np 

class Perceptron:
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
            for Xi, y in zip(X, y):
                prediction = self.predict(Xi)
                update = self.learning_rate * (y - prediction)
                self.w_ += update * Xi
                self.b_ += update

    def predict(self, X):
        prediction = X @ self.w_ + self.b_ 
        return np.where(prediction >= 0.0, 1, 0)