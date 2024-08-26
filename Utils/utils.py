import os
import csv
import numpy as np
from LinearClassifiers.logisticregression import LogisticRegression

def csv_to_np(file_path: str) -> np.ndarray:

    #open the file
    with open(file_path) as csvfile:

        # initialize a reader object
        reader = csv.reader(csvfile)

        # skip the header
        header = next(reader)

        # create a list of rows
        data_list = [row for row in reader]

        # return the list as a numpy array
        return np.array(data_list)


def get_logistic_model(data: np.ndarray):
    x = data[:, :-1]
    y = data[:, -1]
    return LogisticRegression(x, y)
