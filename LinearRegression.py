import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

class LinearRegression:
    def __init__(self, lr=0.001, n_iter=10000):
       self.lr = lr
       self.n_iter = n_iter
       self.weights = None
       self.bias = None

    def clean(self, X, z_score_thresh=2):
        ## remove outliers
        z_score = np.abs(stats.zscore(X))
        x = X[(np.abs( z_score < z_score_thresh)).all(axis=1)]
        return x
    def drop_target(self, X, target_name):
        ## remove target
        y = X[target_name]
        x = X.drop(columns=[target_name])
        return x,y
    def scale(self, X):
        ## remove target

        ## scale
        scaler = StandardScaler()
        x = scaler.fit_transform(X)
        return x
    
    def fit_ordinary_least_squares(self, X, y):
        '''
        - Implementation of ordinary least squares
        - calculates the weights and bias
        '''
        n_samples, n_features = X.shape
        X_transpose = X.T   
        self.weights = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
        self.bias = np.mean(y) - np.dot(X, self.weights)
        print(f"Weights: {self.weights}, Bias: {self.bias}")
        
    def fit(self, X, y):
        '''
        - Implementation of gradient descent
        - runs for n_iter iterations
        - calcualtes the predicted values using the current weights and bias
        - dw is the gradient of the loss function with respect to the weights
        - db is the gradient of the loss function with respect to the bias
        - updates the weights and bias using the gradients
        '''

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iter):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # if i % 100 == 0:  # Print every 100 iterations
            #     print(f"Iteration {i}: Weights: {self.weights}, Bias: {self.bias}, dw: {dw}, db: {db}")

    def RSS(self, y, y_predicted):
            return np.sum((y - y_predicted) ** 2)
    def TSS(self, y):
        return np.sum((y - np.mean(y)) ** 2)
    def r2_score(self, y, y_predicted):
        return 1 - self.RSS(y, y_predicted) / self.TSS(y)

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias

        return y_predicted
    
    def residuals(self, y, y_predicted):
        return y - y_predicted
    