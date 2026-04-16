
def get_input():
    print("Введите коэффициенты A, B, C, D, E, F:")
    A = float(input("A = "))
    B = float(input("B = "))
    C = float(input("C = "))
    D = float(input("D = "))
    E = float(input("E = "))
    F = float(input("F = "))
    return A, B, C, D, E, F


def build_function(A, B, C, D, E, F):
    def f(x, y):
        return 0.5 * A * x**2 + B * x * y + 0.5 * C * y**2 - D * x - E * y + F

    def grad(x, y):
        return (
            A * x + B * y - D,
            B * x + C * y - E
        )

    return f, grad


# Аналитическое решение
def solve_exact(A, B, C, D, E, F, f):
    det = A * C - B**2

    if det <= 0:
        raise ValueError("Матрица не положительно определена")

    x = (C * D - B * E) / det
    y = (A * E - B * D) / det

    return x, y, f(x, y)


# Градиентный спуск
def gradient_method(f, grad, x0, y0, lr=0.1, eps=1e-6, max_iter=1000):
    x, y = x0, y0
    prev = f(x, y)

    for i in range(1, max_iter + 1):
        gx, gy = grad(x, y)

        x -= lr * gx
        y -= lr * gy

        curr = f(x, y)

        if abs(curr - prev) < eps:
            return x, y, curr, i

        prev = curr

    return x, y, curr, max_iter


# Покоординатный спуск
def coordinate_method(f, x0, y0, lr=0.1, eps=1e-6, max_iter=1000):
    x, y = x0, y0
    prev = f(x, y)
    h = 1e-5

    for i in range(1, max_iter + 1):
        dfx = (f(x + h, y) - f(x - h, y)) / (2 * h)
        x -= lr * dfx

        dfy = (f(x, y + h) - f(x, y - h)) / (2 * h)
        y -= lr * dfy

        curr = f(x, y)

        if abs(curr - prev) < eps:
            return x, y, curr, i

        prev = curr

    return x, y, curr, max_iter


# Главная функция
def main():
    A, B, C, D, E, F = get_input()

    f, grad = build_function(A, B, C, D, E, F)

    x0, y0 = 0.0, 0.0

    # аналитика
    x_exact, y_exact, f_exact = solve_exact(A, B, C, D, E, F, f)

    # численные методы
    x_gd, y_gd, f_gd, it_gd = gradient_method(f, grad, x0, y0)
    x_cd, y_cd, f_cd, it_cd = coordinate_method(f, x0, y0)

    # вывод
    print("\nРезультаты")

    print("\nАналитическое решение:")
    print(f"x = {x_exact:.6f}, y = {y_exact:.6f}, f = {f_exact:.6f}")

    print("\nГрадиентный спуск:")
    print(f"x = {x_gd:.6f}, y = {y_gd:.6f}, f = {f_gd:.6f}, итераций = {it_gd}")

    print("\nПокоординатный спуск:")
    print(f"x = {x_cd:.6f}, y = {y_cd:.6f}, f = {f_cd:.6f}, итераций = {it_cd}")


if __name__ == "__main__":
    main()