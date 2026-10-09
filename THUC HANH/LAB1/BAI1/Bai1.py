import copy

import numpy as np

#f(x) = x1^4 + x2^2 + 2x1x2 + 1

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
        
x_0 = np.array([-76.0, 67.0])

history, x_opt = newton_optimizer(grad_func, hessian_func, x_0)

print("Optimization history:", history)
print("Optimal solution:", x_opt)
print("Function value at optimal solution:", f(x_opt))