class Mo:
    def __init__(self, c1, c2, c3, c4, e):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4

    def func(self, x):
        return (self.c1 * (x * x * x) + self.c2 * (x * x) + c3 * x + c4)

    def HalfDivision(self):
        print(f"f(x) = {c1}x^3", end='')
        if c2 < 0:
            print(f"{c2}x^2", end='')
        else:
            print(f"+{c2}x^2", end='')
        if c3 < 0:
            print(f"{c3}x", end='')
        else:
            print(f"+{c3}x", end='')
        if c4 < 0:
            print(f"{c4}")
        else:
            print(f"+{c4}")

        print("Начальный интервал:")
        a = int(input())
        b = int(input())
        x = (a + b) / 2
        l = b - a
        fx = self.func(x)
        k = -1

        while l > e:
            k += 1
            y = a + (1 / 4)
            z = b - (1 / 4)
            fx = self.func(x)
            fy = self.func(y)
            fz = self.func(z)
            print(f"При k = {k}")
            print(f"x = {x}; y = {y}; z = {z}; a = {a}; b = {b}; l = {l}")
            print(f"f(x) = {fx}; f(y) = {fy}; f(z) = {fz}")
            if fy < fx:
                b = x
                x = y
            else:
                if fz < fx:
                    a = x
                    x = z
                else:
                    a = y
                    b = z
            l = abs(a - b)


print("Введите коэф. уравнения:")
print("c1:", end="")
c1 = int(input())
print("c2:", end="")
c2 = int(input())
print("c3:", end="")
c3 = int(input())
print("c4:", end="")
c4 = int(input())
print("Требуемая точноесть e: ", end="")
e = float(input())
mo = Mo(c1, c2, c3, c4, e)

mo.HalfDivision()