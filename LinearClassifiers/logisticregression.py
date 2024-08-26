import numpy as np

"""
@brief class to fit a logistic regression model for a binary classification problem.
"""


class LogisticRegression:
    def __init__(self, x: np.ndarray or list, y=None):
        self.params = []

        # if features and target have been split
        if y is not None:

            # initialize the x
            if isinstance(x, np.ndarray):
                self.x: np.ndarray = x.astype(float)
            elif isinstance(x, list):
                self.x: np.ndarray = np.array(x).astype(float)
            else:
                raise TypeError('x must be an ndarray or list')

            # initialize the y
            if isinstance(y, np.ndarray):
                self.y: np.ndarray = y.astype(float)
            elif isinstance(y, list):
                self.y: np.ndarray = np.array(y).astype(float)

        else:  # if features and target have not been split

            # initialize the x
            if isinstance(x, np.ndarray):
                # keep only the first k - 1 cols
                self.x: np.ndarray = x[:, :-1].astype(float)
                self.y: np.ndarray = x[:, -1].astype(float)
            elif isinstance(x, list):
                x_array = np.array(x)
                self.x: np.ndarray = x_array[:, :-1].astype(float)
                self.y: np.ndarray = x_array[:, -1].astype(float)
            else:
                raise TypeError('x must be an ndarray or list')

        # temp array to hold a column of one's for adding the constant.
        ones = np.ones((self.x.shape[0], 1))

        self.x = np.hstack((ones, self.x))

        # number of features for an observation
        self.n_components: int = self.x.shape[1]

        # number of components in the dataset
        self.N: int = x.shape[0]

        self.beta_hat: np.ndarray or None = None

    def F_x(self, beta_k: np.ndarray) -> np.ndarray:
        f_x: list = [sum(self.x[i][j] * (self.y[i] - self.pk(self.x[i], beta_k)) for i in range(self.N)) for j in
                     range(self.n_components)]
        return np.array(f_x)

    #compute the hessian matrix with respect to beta
    def H_x(self, beta_k: np.ndarray) -> np.ndarray:
        # i summation over observations
        # j the first partial
        # k the second partial

        pki = [self.pk(self.x[i], beta_k) for i in range(self.N)]

        J_x: list = [[sum(self.x[i][j] * self.x[i][k] * pki[i] * (1 - pki[i])
            for i in range(self.N))
            for k in range(self.n_components)]
            for j in range(self.n_components)]
        return np.array(J_x)

    """
    @brief computes the probability G = k conditioned on x and theta
    """

    def pk(self, x: np.ndarray, beta: np.ndarray) -> float:
        #return np.exp(x @ beta) / (1 + np.exp(x @ beta))
        z = x @ beta
        if z >= 0:
            return 1 / (1 + np.exp(-z))
        else:
            exp_z = np.exp(z)
            return exp_z / (1 + exp_z)

    def compute_beta(self, x0: np.ndarray) -> np.ndarray:
        N = 10000
        epsilon = 0.001
        x = x0

        k = 1

        while k <= N:
            print(f"Iteration: {k}")

            # initial approximation to F(x)
            F_x: np.ndarray = self.F_x(x)

            # H(x)
            H_x: np.ndarray = self.H_x(x)

            # compute y
            y: np.ndarray = np.linalg.solve(H_x, -1 * F_x)

            # update beta_hat
            beta_hat = x + y

            #compute the norm of y
            y_norm: float = np.linalg.norm(y)

            print(f"y_norm: {y_norm}")

            if y_norm < epsilon:
                print("Success.")
                return beta_hat
            else:
                k += 1

    def fit(self):
        x0 = np.ones(8)
        self.beta_hat = self.compute_beta(x0)
