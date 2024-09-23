import numpy as np  
class LogisticRegression:
    def __init__(self, lr = 0.001, n_iter = 1000) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z)) 
    def fit(self, X,y):
        if not np.all(np.isin(y, [0, 1])):
            print("Array contains elements other than 0 or 1.")
            return -1
       

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            # if i % 100 == 0:  # Print every 100 iterations
            #         print(f"Iteration {i}: Weights: {self.weights}, Bias: {self.bias}, dw: {dw}, db: {db}")
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls
  
        