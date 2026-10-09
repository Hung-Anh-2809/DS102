import numpy as np
from functions import f, grad_func, hessian_func
from optimizer import newton_optimizer

#f(x) = x1^4 + x2^2 + 2*x1*x2 + 1
x_0 = np.array([6, 7])

history, x_opt = newton_optimizer(grad_func, hessian_func, x_0, tol=1e-8, max_iter=100)

print("Lịch sử tối ưu:", history)
print("Điểm cực trị:", x_opt)
print("Giá trị tại điểm cực trị:", f(x_opt))
