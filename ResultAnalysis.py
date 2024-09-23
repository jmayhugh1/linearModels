import numpy as np
import LogisticRegression as lgr
import LinearRegression as lr
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

'''
Perform a 5-fold cross validation.
Compute accuracy, precision, recall, and F1 score for each validation set across 5 folds. Report
the average and standard deviation of these metrics. Do you see a big change across different
folds?
'''
def k_fold_cross_validation(X, y, k=5):
    model = lgr.LogisticRegression()

    # Initialize 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # To store the results of each fold
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    # Perform cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train the model on the training set
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1)
        recall = recall_score(y_test, y_pred, zero_division=1)
        f1 = f1_score(y_test, y_pred, zero_division=1)
        
        # Append metrics for each fold
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # Report the average and standard deviation for each metric
    print(f"Accuracy: Mean = {np.mean(accuracy_list):.4f}, Std = {np.std(accuracy_list):.4f}")
    print(f"Precision: Mean = {np.mean(precision_list):.4f}, Std = {np.std(precision_list):.4f}")
    print(f"Recall: Mean = {np.mean(recall_list):.4f}, Std = {np.std(recall_list):.4f}")
    print(f"F1 Score: Mean = {np.mean(f1_list):.4f}, Std = {np.std(f1_list):.4f}")
'''
Perform a 5-fold cross validation.
Compute RMSE for each validation set across 5 folds. Report average and standard deviation
of RMSE values. Do you see a big change across different folds? How can you use the coefficient
of this model to find the most informative features?
'''
def k_fold_cross_validation_RMSE(X, y, k=5):
    model = lr.LinearRegression()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmse_list = []
    feature_importance = np.zeros(X.shape[1])  # To store sum of absolute coefficients
    
    for train_index, test_index in kf.split(X):
        # Use normal array indexing since X and y are numpy arrays
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Fit the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        rmse_list.append(rmse)
        
        # Sum absolute values of weights (coefficients)
        feature_importance += np.abs(model.weights)
    
    # Average feature importance across folds
    feature_importance /= k
    
    # Report RMSE mean and std
    print(f"RMSE: Mean = {np.mean(rmse_list):.4f}, Std = {np.std(rmse_list):.4f}")
    
    # Report most informative features
    feature_ranking = np.argsort(-feature_importance)  # Sort in descending order
    #print(f"Most informative features (by importance): {feature_ranking}")
    # most important feature
    names = X.columns
    for i in range(5):  
        print(f"{names[feature_ranking[i]]}: {feature_importance[feature_ranking[i]]:.4f}")