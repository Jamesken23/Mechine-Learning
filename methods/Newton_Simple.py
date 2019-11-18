"""
Newton法求二元函数
"""
import numpy as np
from sympy import symbols, diff


# 首先定义二维向量内的元素
x1 = symbols("x1")
x2 = symbols("x2")


def jacobian(f, x):
    """
    求函数一阶导
    :param f: 原函数
    :param x: 初始值
    :return: 函数一阶导的值
    """
    grandient = np.array([diff(f, x1).subs(x1, x[0]).subs(x2, x[1]), diff(f, x2).subs(x1, x[0]).subs(x2, x[1])], dtype=float)
    return grandient


def hessian(f, x):
    """
    求函数二阶导，即海森矩阵
    :param f: 原函数
    :param x: 初始值
    :return: 函数二阶导的值
    """
    hessian_obj = np.array([[diff(f, x1, 2), diff(diff(f, x1), x2)], [diff(diff(f, x2), x1), diff(f, x2, 2)]], dtype=float)
    return hessian_obj


def newton(f, x, iters):
    """
    实现牛顿法
    :param f: 原函数
    :param x: 初始值
    :param iters: 遍历的最大epoch
    :return:
    """
    Hessian_T = np.linalg.inv(hessian(f, x))
    H_G = np.matmul(Hessian_T, jacobian(f, x))
    x_new = x - H_G
    print("第1次迭代后的结果为:", x_new)
    for i in range(1, iters):
        Hessian_T = np.linalg.inv(hessian(f, x_new))
        H_G = np.matmul(Hessian_T, jacobian(f, x_new))
        x_new = x_new - H_G
        print("第"+str(i+1)+"次迭代后的结果为:", x_new)
    return x_new


if __name__ == "__main__":
    x = np.array([100, 10], dtype=float)
    f = x1**2 + 3*x1 - 3*x1*x2 + 2*x2**2 + 4*x2
    print(newton(f, x, 1000))