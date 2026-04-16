import numpy as np
import matplotlib.pyplot as plt


def compute_ode(f, y_init, interval, step, scheme='all'):
    """
    Численное решение ОДУ y' = f(x, y)
    """
    x_start, x_end = interval
    steps = int((x_end - x_start) / step)
    grid = np.linspace(x_start, x_end, steps + 1)

    sol_rk = None
    sol_abm = None

    # --- Runge-Kutta 4 ---
    if scheme in ('rk', 'all'):
        sol_rk = np.empty(steps + 1)
        sol_rk[0] = y_init

        for j in range(steps):
            xj, yj = grid[j], sol_rk[j]

            k1 = f(xj, yj)
            k2 = f(xj + step / 2, yj + step * k1 / 2)
            k3 = f(xj + step / 2, yj + step * k2 / 2)
            k4 = f(xj + step, yj + step * k3)

            sol_rk[j + 1] = yj + (step / 6) * (k1 + 2*k2 + 2*k3 + k4)

    # --- Adams-Bashforth-Moulton ---
    if scheme in ('abm', 'all'):
        if steps < 4:
            raise RuntimeError("Недостаточно шагов для метода Адамса")

        sol_abm = np.empty(steps + 1)
        sol_abm[0] = y_init

        # старт через RK4
        for j in range(4):
            xj, yj = grid[j], sol_abm[j]

            k1 = f(xj, yj)
            k2 = f(xj + step / 2, yj + step * k1 / 2)
            k3 = f(xj + step / 2, yj + step * k2 / 2)
            k4 = f(xj + step, yj + step * k3)

            sol_abm[j + 1] = yj + (step / 6) * (k1 + 2*k2 + 2*k3 + k4)

        f_cache = np.zeros(steps + 1)
        for j in range(5):
            f_cache[j] = f(grid[j], sol_abm[j])

        for j in range(4, steps):
            # predictor
            y_tmp = sol_abm[j] + step / 720 * (
                1901*f_cache[j] - 2774*f_cache[j-1]
                + 2616*f_cache[j-2] - 1274*f_cache[j-3]
                + 251*f_cache[j-4]
            )

            f_tmp = f(grid[j+1], y_tmp)

            # corrector
            sol_abm[j+1] = sol_abm[j] + step / 720 * (
                251*f_tmp + 646*f_cache[j]
                - 264*f_cache[j-1] + 106*f_cache[j-2]
                - 19*f_cache[j-3]
            )

            f_cache[j+1] = f(grid[j+1], sol_abm[j+1])

    return grid, sol_rk, sol_abm


def visualize(grid, rk, abm, exact, caption="ODE solution comparison"):
    """
    Более современная и аккуратная визуализация
    """

    plt.style.use('seaborn-v0_8')

    # --- Основной график ---
    fig, ax = plt.subplots(figsize=(11, 6))

    if rk is not None:
        ax.plot(grid, rk, linewidth=2.5, label='Runge-Kutta 4')

    if abm is not None:
        ax.plot(grid, abm, linestyle='--', linewidth=2.5, label='Adams method')

    ax.scatter(grid, exact, s=20, alpha=0.7, label='Exact solution')

    ax.set_title(caption, fontsize=14, pad=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y(x)', fontsize=12)

    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    # --- Ошибки ---
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Абсолютная
    if rk is not None:
        axes[0].plot(grid, np.abs(rk - exact), linewidth=2, label='RK4')

    if abm is not None:
        axes[0].plot(grid, np.abs(abm - exact), linestyle='--', linewidth=2, label='Adams')

    axes[0].set_yscale('log')
    axes[0].set_title('Absolute error', fontsize=13)
    axes[0].set_ylabel('Error')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Относительная
    eps = 1e-14

    if rk is not None:
        axes[1].plot(grid, np.abs((rk - exact) / (exact + eps)), linewidth=2, label='RK4')

    if abm is not None:
        axes[1].plot(grid, np.abs((abm - exact) / (exact + eps)), linestyle='--', linewidth=2, label='Adams')

    axes[1].set_yscale('log')
    axes[1].set_title('Relative error', fontsize=13)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Relative error')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def run_demo():
    """
    Демонстрационный запуск
    """

    def rhs(x, y):
        return y

    y_init = 1.0
    domain = (0.0, 5.0)
    step = 0.05

    def exact_fn(x):
        return np.exp(x)

    x, rk_sol, abm_sol = compute_ode(rhs, y_init, domain, step, scheme='all')
    exact_vals = exact_fn(x)

    visualize(x, rk_sol, abm_sol, exact_vals)


if __name__ == "__main__":
    run_demo()