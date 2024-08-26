import numpy as np
from LinearClassifiers.logisticregression import LogisticRegression
import pytest
from Utils.utils import csv_to_np, get_logistic_model

# file name
file_name = "lending_data.txt"

# create file path
file_path = f"../Data/{file_name}"

data = csv_to_np(file_path)

def test_init():
    logistic_test = get_logistic_model(data)
    assert (logistic_test.x.shape[1]) == data.shape[1]
    iterator = iter(logistic_test.x)
    assert isinstance(next(iterator)[1], float)
    assert (logistic_test.x[:, 0] == 1).all()
    assert logistic_test.n_components == data.shape[1]
    #logistic_regression_test = LogisticRegression()

def test_pk():
    logistic_test = get_logistic_model(data)

    # initialize a beta vector
    beta = np.ones(logistic_test.n_components)
    x_k = np.ones(logistic_test.n_components)

    test_p = logistic_test.pk(x_k, beta)

    assert isinstance(test_p, float)
    assert test_p < 1

def test_F_x():
    # initialize a test case
    logistic_test = get_logistic_model(data)

    # initialize a test beta vector
    beta = np.ones(logistic_test.n_components)

    F_x: np.ndarray = logistic_test.F_x(beta)

    assert isinstance(F_x, np.ndarray)
    assert len(F_x) == logistic_test.n_components

def test_H_x():
    # initialize a test case
    logistic_test = get_logistic_model(data)

    # initialize a test beta vector
    beta = np.ones(logistic_test.n_components)

    H_x: np.ndarray = logistic_test.H_x(beta)

    assert isinstance(H_x, np.ndarray)
    assert H_x.shape[0] == logistic_test.n_components
    assert H_x.shape[1] == logistic_test.n_components

def test_compute_beta():
    # initialize a test case
    logistic_test = get_logistic_model(data)
    # initialize a test beta vector
    x0 = np.ones(logistic_test.n_components)

    beta = logistic_test.compute_beta(x0)

    assert isinstance(beta, np.ndarray)

