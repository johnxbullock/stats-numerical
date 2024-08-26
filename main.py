import numpy as np
from LinearClassifiers.logisticregression import LogisticRegression
from Utils.utils import get_logistic_model, csv_to_np

# file name
file_name = "lending_data.txt"

# create file path
file_path = f"./Data/{file_name}"

data = csv_to_np(file_path)

logistic_model = get_logistic_model(data)

x0 = np.ones(logistic_model.n_components)

beta = logistic_model.compute_beta(x0)
