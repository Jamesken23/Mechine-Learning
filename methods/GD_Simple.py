"""
一维问题的梯度下降法示例
"""
from scipy.misc import derivative


def func_1d(x):
    """
    目标函数
    :param x: 自变量，标量
    :return: 因变量，标量
    """
    return x ** 2 + 1


def grad_1d(x):
    """
    目标函数的梯度
    :param x: 自变量，标量
    :return: 因变量，标量
    """
    # 求一阶导
    return derivative(func_1d, x, dx=1e-6, n=1)
    # 求二阶导
    # return derivative(func_1d, x, dx=1e-6, n=2)


def gradient_descent_ld(grad, cur_x=0.1, learning_rate=0.01, precision=0.0001, max_iters=1000):
    """
    一维问题的梯度下降法
    :param grad: 目标函数的梯度
    :param cur_x: 当前x值，通过参数可以提供初始值
    :param learning_rate: 学习率，也相当于设置的步长
    :param precision: 设置收敛精度
    :param max_iters: 最大迭代次数
    :return: 局部最小值x
    """
    for i in range(max_iters):
        grad_cur = grad(cur_x)
        # 当梯度趋近于0时，视为收敛
        if abs(grad_cur) < precision:
            break
        cur_x = cur_x - grad_cur * learning_rate
        print("第%d次迭代的x值为：%f" % (i, cur_x))
    print("局部最小值 x=", cur_x)
    return cur_x


def gradient_descent_ld_decay(grad, cur_x=0.1, learning_rate=0.01, precision=0.0001, max_iters=1000, decay=0.5):
    """
    一维问题的梯度下降法，变步长
    :param grad: 目标函数的梯度
    :param cur_x: 当前x值，通过参数可以提供初始值
    :param learning_rate: 学习率，也相当于设置的步长
    :param precision: 设置收敛精度
    :param max_iters: 最大迭代次数
    :param decay: 学习率衰减因子
    :return:
    """
    for i in range(max_iters):
        # 新的步长
        learning_rate = learning_rate * 1.0 / (1.0 + decay * i)
        grad_cur = grad(cur_x)
        # 当梯度趋近于0时，视为收敛
        if abs(grad_cur) < precision:
            break
        cur_x = cur_x - grad_cur * learning_rate
        print("第%d次迭代的x值为：%f" % (i, cur_x))
    print("局部最小值 x=", cur_x)
    return cur_x


if __name__ == "__main__":
    # gradient_descent_ld(grad_1d)
    gradient_descent_ld_decay(grad_1d)