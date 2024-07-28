
"""
Import the required libraries. We will use only basic libraries of numpy, pandas for this implementation from scratch
"""

import pandas as pd
import numpy as np


"""Define the sigmoid function. Will be called by predict function later"""
def sigmoid(z):
    sigmoid=1.0/(1.0 + np.exp(-z))
    return sigmoid

"""Define the predict function. Will take X and w as input and compute dot product and then apply sigmoid function on it to generate prediction for the current w. 
X is a np array of m*n; w is np array of n*1 """
def predict(X,w):
    # dot product of X and w. the bias term is also part of X and w np arrays
    z = np.dot(X, w)
    # Apply sigmoid on above dot product to generate probabilites
    pred_prob = sigmoid(z)
    return pred_prob

"""Define the Loss function. Will take X and w as input and call predict() to generate prediction for the current w. Then, the Loss value will be computed for current w by 
using the actual y labels and predicted probabilities
This function is only needed to print the Loss values at various iterations as the gradient is directly computed using the values of X, y and predicted probabilites
X is a np array of m*n; w is np array of n*1; y is np array of m*1
Scalar value (Loss_value) is returned by the function
"""

def loss(X, y, w):

    pred_prob = predict(X, w)

    # Loss_value computation
    Loss_value = -1 * (np.mean((y*np.log(pred_prob) + (1-y)*np.log(1 - pred_prob))) )

    return Loss_value

"""Define the function to compute gradient. Will take X and w as input and call predict() to generate prediction for the current w. 
Then, the gradient of Loss function is directly computed for current w by using the values of X, y and predicted probabilites 
X is a np array of m*n; w is np array of n*1; y is np array of m*1
n*1 np array is returned by the function"""

def gradient(X, y, w):

    pred_prob = predict(X, w)

    # Gradient (1/n)*(p-y)*x
    grad = (np.dot(X.T, (pred_prob - y))) / X.shape[0]
    return grad

"""Define the function to perform gradient descent.

Gradient descent is performed for the input hyperparameter values of learning rate and epochs.
In each epoch of gradient descent:
     computes the gradient by calling gradient function. Gradient function is called by passing X, y, and weight values (initially w as 0s). 
     Then, update weights using teh gradient and learning rate

Separately, Loss values are appended to keep track of Loss values during each epoch.
"""

def gradient_descent(X, y, lr = 0.25, epochs = 50):

    w = np.zeros((X.shape[1], 1))

    loss_list = []

    for _ in range(epochs):

        # Append Loss values for each epoch
        Loss_value = loss(X, y, w)
        loss_list.append(Loss_value)

        # Compute the gradient by calling the function
        grad = gradient(X, y, w)

        # Compute the updated weight values
        w = w - lr*grad

    return w, loss_list


"""
The scratch implementation of Logistic Regression model is completed in previous functions
"""




"""
Now, we will use above functions to make predictions. Also, we will compare the results of predictions from above scratch implementation vs
sklearn Logistic implementation
"""


"""
Create data using datasets from sklearn and perform validations for usefulness of above defined functions
"""

# Import libraries to create custom dataset and to perform model validations
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc,confusion_matrix, precision_score, recall_score

## Create custom dataset
X, y = make_classification(n_samples=800, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=1)

# Convert y from row vector to column vector
y = y.reshape(-1, 1)

# Add bias term to X
ones= np.ones((800,1))
X_ = np.hstack((ones, X))


"""
Train-test split of dataset
"""
X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.25, random_state=1)

"""Train the Scratch Logistic model"""
# Train using the custom functions defined earlier
final_weights, loss_list = gradient_descent(X_train, y_train,lr = 0.01, epochs = 50000)

print("*************************Scratch Logistic Results**************************************")
#Print final weights obtained from Scratch Logistic model. First weight is for bias term.
print("Model intercept using Scratch Logistic implementation is:",np.round(final_weights[0],2))
print("Model coefficient using Scratch Logistic implementation are:",np.round(final_weights[1:],2))

"""Performance assessment on test dataset for Scratch Logistic model"""
# Predict on test data
pred_prob_test_scratch_logistic=predict(X_test,final_weights)

# Convert probabilities to class predictions (using a threshold of 0.5)
pred_test_scratch_logistic = (pred_prob_test_scratch_logistic >= 0.5).astype(int)

# Evaluate the custom Logistic model
test_accuracy_scratch_logistic = accuracy_score(y_test, pred_test_scratch_logistic)
test_precision_scratch_logistic = precision_score(y_test, pred_test_scratch_logistic)
test_recall_scratch_logistic = recall_score(y_test, pred_test_scratch_logistic)

print("Testing Accuracy of Scratch Logistic model:", round(100*test_accuracy_scratch_logistic,2),"%")
print("Testing Precision of Scratch Logistic model:", round(100*test_precision_scratch_logistic,2),"%")
print("Testing Recall of Scratch Logistic model:", round(100*test_recall_scratch_logistic,2),"%")
print("***************************************************************")

"""Train the sklearn Logistic model"""

########
# obtain sklearn Logistic implementation weights and predictions
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# bias term column not needed seperately in sklean; so, X_train[:, 1:] used
# For y_train, Convert to 1-dimensional array before passing
model.fit(X_train[:, 1:], y_train.ravel())
sklearn_y_test_predictions=model.predict(X_test[:, 1:])
print("*************************sklearn Logistic Results**************************************")
print("Model intercept using sklearn Logistic implementation is:",np.round(model.intercept_,2))
print("Model coefficient using sklearn Logistic implementation are:",np.round(model.coef_,2))


"""Performance assessment on test dataset for sklearn Logistic model"""
test_accuracy_sklearn = accuracy_score(y_test, sklearn_y_test_predictions)
test_precision_sklearn = precision_score(y_test, sklearn_y_test_predictions)
test_recall_sklearn = recall_score(y_test, sklearn_y_test_predictions)

print("Testing Accuracy of sklearn model:", round(100*test_accuracy_sklearn,2),"%")
print("Testing Precision of sklearn model:", round(100*test_precision_sklearn,2),"%")
print("Testing Recall of sklearn model:", round(100*test_recall_sklearn,2),"%")
print("***************************************************************")


