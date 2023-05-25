import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


class NewtonMethod:
    def __init__(self, f, grad_f, H, x0, e1, e2, M):
        self.f = f
        self.grad_f = grad_f
        self.H = H
        self.x0 = x0
        self.e1 = e1
        self.e2 = e2
        self.M = M
        self.f_values = [f(x0)]
        self.x_values = [x0]

    def run(self):
        print('\n[ШАГ 2]\nk = 0\n')
        k = 0  # шаг 2
        x = self.x0
        print(f'x0: {x}')
        while True:
            print(f'\nk = {k} | x = {x} | grad(x{k}) = {self.grad_f(x)}\n')
            grad = self.grad_f(x)  # шаг 3
            print(f'\n[ШАГ 3_{k}]\ngrad(x{k}) = {grad}')
            print(f'\n[ШАГ 4_{k}]\n||grad(x{k})|| <= e1? : ||grad(x{k})|| = {np.linalg.norm(grad)}, e1 = {self.e1}')
            if np.linalg.norm(grad) <= self.e1:
                print(f'Да. Расчет окончен. x* = x{k} = {x}')
                self.graph(x, len(self.f_values))
                return x
            print(f'Нет.\n\n[ШАГ 5_{k}]\nk >= M? : k = {k}, M = {self.M}')
            if k >= self.M:
                print(f'Да. Расчет окончен. x* = x{k} = {x}')
                self.graph(x, len(self.f_values))
                return x
            print(f'Нет.\n\n[ШАГ 6_{k}]\nВычислить H(x{k}) : {self.H(x)}\n[ШАГ 7_{k}]\nВычислить H^-1(x{k}\n')
            H_inverse = np.linalg.inv(self.H(x))  # шаг 6-7 (H(x), H^-1(x)
            print(f'\n[ШАГ 8_{k}]\n Проверить выполнение условия H^-1(x{k} > 0 : H^-1(x{k} = {H_inverse}\n')
            if np.all(H_inverse) > 0:  # шаг 8 - проверка, что все элементы матрицы > 0
                d = -H_inverse.dot(grad)  # шаг 9 - умножаем матрицу H на вектор grad
                tk = 1
                xk = x + tk * d  # шаг 10 при d = -H^-1(xk)*grad(xk)
                x_new = xk
                print(f'Да. Переходим к шагу 9\n\n[ШАГ 9_{k}]\nd{k} = {d}\n\n[ШАГ 10_{k}]\nx{k + 1} = {xk}, tk = {tk}')
            else:
                tk = 1
                d = -grad  # шаг 10 при d = -grad(xk)
                xk = x + tk * d  # шаг 10 при d = -H^-1(xk)*grad(xk)
                while xk < x:
                    tk /= 2
                x_new = xk
                print(f'Да. Переходим к шагу 9\n\n[ШАГ 9_{k}]\nd{k} = {d}\n\n[ШАГ 10_{k}]\nx{k + 1} = {xk}, tk = {tk}')
            print(f'\n[ШАГ 11_{k}]\nПроверка выполнения условий: ||x{k + 1}|| < e2, |f(x{k + 1} - f(x{k}| < e')
            print(f'{np.linalg.norm(grad)} < {self.e1}?, {np.linalg.norm(x_new - x)} < {self.e2}?')
            if np.linalg.norm(grad) < self.e1 and np.linalg.norm(x_new - x) < self.e2:
                break
        x = x_new
        k += 1
        self.f_values.append(self.f(x))
        self.x_values.append(x)

        self.graph(x_new, len(self.f_values))
        return x_new


    def graph(self, x, k):
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
