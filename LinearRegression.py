import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iter=10000):
       self.lr = lr
       self.n_iter = n_iter
       self.weights = None
       self.bias = None
    def fit(self, X, y):
        print(np.isnan(X).sum()) 
        print(np.isinf(X).sum())  
        print(np.isnan(y).sum()) 
        print(np.isinf(y).sum())  

        
        
        
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iter):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if i % 100 == 0:  # Print every 100 iterations
                print(f"Iteration {i}: Weights: {self.weights}, Bias: {self.bias}, dw: {dw}, db: {db}")

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted