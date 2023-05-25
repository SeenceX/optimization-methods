import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def newtonMethod(f, grad_f, H, x0, e1, e2, M):
    print('\n[ШАГ 2]\nk = 0\n')
    k = 0  # шаг 2
    x = x0
    f_values = [f(x)]  # список для значений функции на каждой итерации
    x_values = [x]
    print(f'x0: {x}')
    while True:
        print(f'\nk = {k} | x = {x} | grad(x{k}) = {grad_f(x)}\n')
        grad = grad_f(x)  # шаг 3
        print(f'\n[ШАГ 3_{k}]\ngrad(x{k}) = {grad}')
        print(f'\n[ШАГ 4_{k}]\n||grad(x{k})|| <= e1? : ||grad(x{k})|| = {np.linalg.norm(grad)}, e1 = {e1}')
        if np.linalg.norm(grad) <= e1:
            print(f'Да. Расчет окончен. x* = x{k} = {x}')
            graph(x, len(f_values))
            return x
        print(f'Нет.\n\n[ШАГ 5_{k}]\nk >= M? : k = {k}, M = {M}')
        if k >= M:
            print(f'Да. Расчет окончен. x* = x{k} = {x}')
            graph(x, len(f_values))
            return x
        print(f'Нет.\n\n[ШАГ 6_{k}]\nВычислить H(x{k}) : {H(f,x)}\n[ШАГ 7_{k}]\nВычислить H^-1(x{k})\n')
        H_inverse = np.linalg.inv(H(f, x))  # шаг 6-7 (H(x), H^-1(x)
        print(f'\n[ШАГ 8_{k}]\n Проверить выполнение условия H^-1(x{k} > 0 : H^-1(x{k} = {H_inverse}\n')
        tk = 1
        d = -H_inverse.dot(grad)
        if np.all(H_inverse) > 0:  # шаг 8 - проверка, что все элементы матрицы > 0
            tk = 1
            xk = x + tk * d  # шаг 10 при d = -H^-1(xk)*grad(xk)
            x_new = xk
            print(f'Да. Переходим к шагу 9\n\n[ШАГ 9_{k}]\nd{k} = {d}\n\n[ШАГ 10_{k}]\nx{k+1} = {xk}, tk = {tk}')
        else:
            xk = x - tk * grad_f(x)  # шаг 10 при d = -H^-1(xk)*grad(xk)
            while f(xk) < f(x):
                tk /= 2
            xk = x + tk * d
            x_new = xk
            print(f'Да. Переходим к шагу 9\n\n[ШАГ 9_{k}]\nd{k} = {d}\n\n[ШАГ 10_{k}]\nx{k + 1} = {xk}, tk = {tk}')
        print(f'\n[ШАГ 11_{k}]\nПроверка выполнения условий: ||x{k+1}|| < e2, |f(x{k+1} - f(x{k}| < e2')
        print(f'{np.linalg.norm(grad)} < {e1}?, {np.linalg.norm(x_new - x)} < {e2}?')
        if np.linalg.norm(grad) < e1 and np.linalg.norm(x_new - x) < e2:
            break
        x = x_new
        k += 1
        f_values.append(f(x))
        x_values.append(x)

    graph(x_new, len(f_values))
    return x_new


def f(x):
    return (x[0]) ** 3 + 0.6 * x[0] * x[1] + 6 * (x[1]) ** 2


def grad_f(x):
    return np.array([3 * (x[0]) ** 2 + 0.6 * x[1], 0.6 * x[0] + 12 * x[1]])


def H(f, x):
    """
    Computes the Hessian matrix of function f at point x using central difference.
    """
    n = x.shape[0]
    hessian = np.zeros((n, n))
    eps = np.sqrt(np.finfo(float).eps) # machine epsilon

    for i in range(n):
        for j in range(i, n):
            if i == j:
                # diagonal elements
                hessian[i][j] = (f(x + eps*np.eye(n)[i]) - 2*f(x) + f(x - eps*np.eye(n)[i])) / (eps**2)
            else:
                # off-diagonal elements
                hessian[i][j] = hessian[j][i] = (f(x + eps*np.array([1 if k==i or k==j else 0 for k in range(n)])) \
                                                - f(x + eps*np.array([1 if k==i else 0 for k in range(n)])) \
                                                - f(x + eps*np.array([1 if k==j else 0 for k in range(n)])) \
                                                + f(x)) / (eps**2)
    return hessian


def graph(x, k):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Создаем точку и добавляем ее в 3D-пространство
    z = lambda x_1, x_2: x_1 ** 3 + 0.6 * x_1 * x_2 + 6 * x_2 ** 2
    ax.scatter(x[0], x[1], z(x[0], x[1]), color='red')
    ax.text(x[0], x[1], z(x[0], x[1]), str(x), color='red', zdir='z')

    # Строим поверхность
    for i in range(k):
        x1_vals = np.linspace(0, 5, 100)
        x2_vals = np.linspace(0, 5, 100)
        x1, x2 = np.meshgrid(x1_vals, x2_vals)
        ax.plot_surface(x1, x2, z(x1, x2), color='green')

    plt.show()


print('[ШАГ 1]\n')
# x0 = np.array([float(input('x1:')), float(input('x2:'))])
x0 = np.array([1.5, 0.5])
e1, e2 = 0.15, 0.20
M = 10
print(f'x0 = {x0}, e1 = {e1}, e2 = {e2}, M = {M}')

result = newtonMethod(f, grad_f, H, x0, e1, e2, M)
print("Минимум функции:", result)
