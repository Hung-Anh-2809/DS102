import numpy as np  # noqa: I001

#f(x) = x1^4 + x2^2 + 2*x1*x2 + 1
def f(x):
    return x[0]**4 + x[1]**2 + 2*x[0]*x[1] + 1

def grad_func(x):
    return np.array([
        4*x[0]**3 + 2*x[1], 
        2*x[1] + 2*x[0]
    ])

def hessian_func(x):
    return np.array([
        [12*x[0]**2, 2],
        [2, 2]
    ])