import copy
import numpy as np

def newton_optimizer(grad_func, hessian_func, x_0, tol=1e-8, max_iter=100):
    history = [x_0]
    x = copy.deepcopy(x_0)
    for _ in range(max_iter):
        grad = grad_func(x)
        if np.linalg.norm(grad) < tol:
            break
        hessian = hessian_func(x)
        a = np.linalg.solve(hessian, grad)
        x = x - a
        history.append(copy.deepcopy(x))
    return history, x
