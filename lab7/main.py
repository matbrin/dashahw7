import numpy as np


def integrand(x_vals, y_vals, A, B, C):
    return A * x_vals**2 + B * y_vals**2 + C


def analytical_solution(A, B, C):
    return np.pi * ((A + B) / 4 + C)


def mc_average_method(A, B, C, samples):
    x_rand = np.random.uniform(-1, 1, samples)
    y_rand = np.random.uniform(-1, 1, samples)

    mask = x_rand**2 + y_rand**2 <= 1
    points_inside = np.count_nonzero(mask)

    if points_inside == 0:
        return 0.0, 0

    values = integrand(x_rand[mask], y_rand[mask], A, B, C)
    estimate = (np.pi / points_inside) * np.sum(values)

    return estimate, points_inside


def mc_volume_method(A, B, C, samples):
    max_val = max(A, B) + C

    x_rand = np.random.uniform(-1, 1, samples)
    y_rand = np.random.uniform(-1, 1, samples)
    z_rand = np.random.uniform(0, max_val, samples)

    mask = (x_rand**2 + y_rand**2 <= 1) & \
           (z_rand <= integrand(x_rand, y_rand, A, B, C))

    hits = np.count_nonzero(mask)
    volume = 4 * max_val

    estimate = volume * hits / samples
    return estimate, hits


def run_experiments(A, B, C, runs, samples):
    true_val = analytical_solution(A, B, C)

    res_avg = []
    res_vol = []

    for _ in range(runs):
        val1, _ = mc_average_method(A, B, C, samples)
        val2, _ = mc_volume_method(A, B, C, samples)

        res_avg.append(val1)
        res_vol.append(val2)

    res_avg = np.array(res_avg)
    res_vol = np.array(res_vol)

    return {
        "exact": true_val,
        "avg_mean": res_avg.mean(),
        "avg_std": res_avg.std(),
        "avg_err": abs(res_avg.mean() - true_val),
        "vol_mean": res_vol.mean(),
        "vol_std": res_vol.std(),
        "vol_err": abs(res_vol.mean() - true_val),
    }


def main():
    np.random.seed(123)

    # === ВВОД ДАННЫХ ===
    A = float(input("Введите A: "))
    B = float(input("Введите B: "))
    C = float(input("Введите C: "))

    runs = int(input("Введите количество испытаний: "))
    samples = int(input("Введите количество точек: "))

    print("\n--- Параметры ---")
    print(f"A={A}, B={B}, C={C}")
    print(f"Испытаний: {runs}, Точек: {samples}")

    stats = run_experiments(A, B, C, runs, samples)

    print("\n--- Результаты ---")
    print(f"Точное значение: {stats['exact']:.6f}")

    print("\nМетод среднего:")
    print(f"  Среднее: {stats['avg_mean']:.6f}")
    print(f"  Std: {stats['avg_std']:.6f}")
    print(f"  Ошибка: {stats['avg_err']:.6f}")

    print("\nМетод объема:")
    print(f"  Среднее: {stats['vol_mean']:.6f}")
    print(f"  Std: {stats['vol_std']:.6f}")
    print(f"  Ошибка: {stats['vol_err']:.6f}")


if __name__ == "__main__":
    main()