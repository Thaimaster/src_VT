import pickle

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# from memory_profiler import profile
# from typing import Tuple
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import psutil

# data_path = "/home/viettel_May23/ml_algorithms/my_data/my_data_2.csv"
# data = pd.read_csv(data_path)
# X = data.iloc[:, 0:-1].values
# y = data.iloc[:, -1].values


# Make data
def make_data(NUM_SAMPLES, NUM_FEATURES):
    X, y = make_regression(n_samples=NUM_SAMPLES, n_features=NUM_FEATURES,
                           n_informative=NUM_FEATURES, noise=0.5)
    data = pd.DataFrame(X, columns=['feature' + str(i) for i in range(1, NUM_FEATURES + 1)], dtype=np.float16)
    data['output'] = np.array(y, dtype=np.float16)
    return data

# Test/Train
def test_train(data, NUM_SAMPLES):
    X_train, y_train = data.iloc[:int(NUM_SAMPLES / 2)].drop(['output'], axis=1), data.iloc[:int(NUM_SAMPLES / 2)]['output']
    X_test, y_test = data.iloc[int(NUM_SAMPLES / 2):].drop(['output'], axis=1), data.iloc[int(NUM_SAMPLES / 2):]['output']
    return (X_train, y_train, X_test, y_test)


# Fitting
def fitting(X_train, y_train):
    # lm = LogisticRegression(n_jobs=1)
    # lm = LinearRegression(n_jobs=1)
    lm = SVR(C=1.0, epsilon=0.2)
    lm.fit(X_train, y_train)
    del X_train
    del y_train
    return lm

# Saving model
def save(lm):
    with open('LinearModel.sav', mode='wb') as f:
        pickle.dump(lm, f)

# Inference
def model_run(model, testfile):
    """
    Loads and runs a sklearn linear model
    """
    lm = pickle.load(open(model, 'rb'))
    X_test = pd.read_csv(testfile)
    _ = lm.predict(X_test)
    return None

NUM_SAMPLES, NUM_FEATURES = 10000, 10000
data = make_data(NUM_SAMPLES, NUM_FEATURES)
X_train, y_train = data.iloc[0:].drop(['output'], axis=1), data.iloc[0:]['output']
# X_train, y_train, X_test, y_test = test_train(data, NUM_SAMPLES)
# X_test.to_csv("Test.csv", index=False)
print("Training model")
lm = fitting(X_train, y_train)
# print("Saving and Inference model")
# save(lm)
# model_run('SvrModel.sav', 'Test.csv')