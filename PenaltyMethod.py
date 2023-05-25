import numpy as np
from matplotlib import pyplot as plt


def newtonMethod(f, grad_f, H, x0, e1, e2, M):
    print('\n[ШАГ 2]\nk = 0\n')
    k = 0  # шаг 2
    x = x0
    f_values = [f(x)]  # список для значений функции на каждой итерации
    x_values = [x]
    print(f'x0: {x}')
    tk = 1
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
        print(f'Нет.\n\n[ШАГ 6_{k}]\nВычислить H(x{k}) : {H(func,x, eps)}\n[ШАГ 7_{k}]\nВычислить H^-1(x{k})\n')
        H_inverse = np.linalg.inv(H(func,x, eps))  # шаг 6-7 (H(x), H^-1(x)
        print(f'\n[ШАГ 8_{k}]\n Проверить выполнение условия H^-1(x{k} > 0 : H^-1(x{k} = {H_inverse}\n')
        tk = 1
        if np.all(H_inverse) > 0:  # шаг 8 - проверка, что все элементы матрицы > 0
            d = -H_inverse.dot(grad)  # шаг 9 - умножаем матрицу H на вектор grad
            tk = 1
            xk = x + tk * d  # шаг 10 при d = -H^-1(xk)*grad(xk)
            x_new = xk
            print(f'Да. Переходим к шагу 9\n\n[ШАГ 9_{k}]\nd{k} = {d}\n\n[ШАГ 10_{k}]\nx{k + 1} = {xk}, tk = {tk}')
        else:
            d = -grad  # шаг 10 при d = -grad(xk)
            xk = x + tk * grad_f(x)  # шаг 10 при d = -H^-1(xk)*grad(xk)
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



def grad_f(x):
    return np.array([2*x[0], 2*x[1]])


def H(f, x, eps):
    """
    Computes the Hessian matrix of function f at point x using central difference.
    """
    print('x:',x)
    n = len(x)
    hessian = np.zeros((n, n))

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
    print("---------------------GRAPH---------------------")
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


def func(x):
    return x[0] ** 2 + x[1] ** 2


def gj1(x):
    return x[0] + 2 * x[1] - 3


def F(x, r):
    return func(x) + r * gj1(x)


def penalty_method(func, g, x0, r0, C, eps):
    k = 0
    xk = x0

    while True:
        rk = r0 * C ** k
        x_star = newtonMethod(func, grad_f, H, xk, 0.05, 0.05, 10)
        P = max([max(0, gj1(x_star))]) ** 2 + max([max(0, -gj1(x_star))]) ** 2
        print(f'x_star: {x_star} | P: {P}')

        if P <= eps:
            return x_star, func(x_star)
        else:
            xk = x_star
            k += 1


#g = gj1(x0)
x0 = np.array([1.5, 0.5])
r0 = 1
C = 5
eps = 0.05

x_min, f_min = penalty_method(func, gj1, x0, r0, C, eps)

print("Минимальное значение: ", f_min)
print("Точка минимума: ", x_min)
