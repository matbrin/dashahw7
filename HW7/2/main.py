import numpy as np
import matplotlib.pyplot as plt

import fdtd
import fdtd.backend as bd

fdtd.set_backend("numpy")

# ПАРАМЕТРЫ МОДЕЛИ
LAMBDA = 1550e-9
C0 = 299_792_458.0

DX = 0.1 * LAMBDA
DOMAIN_X = 25e-6
DOMAIN_Y = 15e-6
DOMAIN_Z = 1

NSTEPS = 180


def simulate(case_label, obj_size_x, obj_size_y):
    """
    Один расчёт:
    создаётся сетка, добавляется объект, источник и детектор
    """

    # СЕТК
    g = fdtd.Grid(
        (DOMAIN_X, DOMAIN_Y, DOMAIN_Z),
        grid_spacing=DX,
        permittivity=1.0,
        permeability=1.0,
    )

    # ГРАНИЦЫ
    pml = 10

    g[0:pml, :, :] = fdtd.PML(name="левая_граница")
    g[-pml:, :, :] = fdtd.PML(name="правая_граница")

    g[:, 0:pml, :] = fdtd.PML(name="нижняя_граница")
    g[:, -pml:, :] = fdtd.PML(name="верхняя_граница")

    g[:, :, 0] = fdtd.PeriodicBoundary(name="граница_z")

    # ИСТОЧНИК
    g[45:50, 65:70, 0] = fdtd.LineSource(
        period=LAMBDA / C0,
        name="источник"
    )

    # ДЕТЕКТОР
    g[18e-6, :, 0] = fdtd.LineDetector(name="детектор")

    # ОБЪЕКТ
    cx = 105
    cy = 70

    xs = cx - obj_size_x // 2
    xe = xs + obj_size_x

    ys = cy - obj_size_y // 2
    ye = ys + obj_size_y

    g[xs:xe, ys:ye, 0:1] = fdtd.AnisotropicObject(
        permittivity=2.5,
        name="объект"
    )

    # РАСЧЁТ
    det_trace = []

    for i in range(NSTEPS):
        g.step()
        det_trace.append(g.E[120, 75, 0, 2])

    det_trace = np.array(det_trace)

    # ПОЛЯ
    ex = bd.numpy(g.E[:, :, 0, 0])
    ey = bd.numpy(g.E[:, :, 0, 1])
    ez = bd.numpy(g.E[:, :, 0, 2])

    power = ex**2 + ey**2 + ez**2

    eps_inv = bd.numpy(g.inverse_permittivity[:, :, 0, 0])
    eps_map = 1.0 / eps_inv

    return {
        "label": case_label,
        "ez": ez,
        "power": power,
        "eps": eps_map,
        "signal": det_trace
    }


def show_maps(data):
    """
    Визуализация результатов одного расчёта
    """

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))

    # Поле Ez
    a = ax[0, 0]
    f = data["ez"]
    lim = max(abs(f.min()), abs(f.max()))
    im = a.imshow(f.T, origin="lower", cmap="RdBu", vmin=-lim, vmax=lim)
    a.set_title(f'Поле Ez ({data["label"]})')
    a.set_xlabel("x")
    a.set_ylabel("y")
    plt.colorbar(im, ax=a)

    # Интенсивность
    a = ax[0, 1]
    im = a.imshow(data["power"].T, origin="lower", cmap="magma")
    a.set_title("Интенсивность поля |E|^2")
    a.set_xlabel("x")
    a.set_ylabel("y")
    plt.colorbar(im, ax=a)

    # Диэлектрическая проницаемость
    a = ax[1, 0]
    im = a.imshow(data["eps"].T, origin="lower", cmap="viridis")
    a.set_title("Распределение диэлектрической проницаемости")
    a.set_xlabel("x")
    a.set_ylabel("y")
    plt.colorbar(im, ax=a)

    # Сигнал детектора
    a = ax[1, 1]
    sig = np.real(data["signal"])
    a.plot(sig)
    a.set_title("Сигнал на детекторе")
    a.set_xlabel("шаг по времени")
    a.set_ylabel("Ez")
    a.grid(True)

    plt.tight_layout()
    plt.show()


def compare(a, b):
    """
    Сравнение сигналов двух расчётов
    """

    plt.figure(figsize=(9, 4))
    plt.plot(np.real(a["signal"]), label=a["label"])
    plt.plot(np.real(b["signal"]), label=b["label"])
    plt.title("Сравнение сигналов на детекторе")
    plt.xlabel("шаг по времени")
    plt.ylabel("Ez")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """
    Меняем размер объекта и смотрим, как меняется результат
    """

    # маленький объект
    case_a = simulate(
        case_label="маленький объект",
        obj_size_x=18,
        obj_size_y=18
    )

    # большой объект
    case_b = simulate(
        case_label="большой объект",
        obj_size_x=36,
        obj_size_y=36
    )

    show_maps(case_a)
    show_maps(case_b)
    compare(case_a, case_b)

    max_a = np.max(np.abs(np.real(case_a["signal"])))
    max_b = np.max(np.abs(np.real(case_b["signal"])))

    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print(f"{case_a['label']}: максимальный сигнал = {max_a:.6e}")
    print(f"{case_b['label']}: максимальный сигнал = {max_b:.6e}")

    print("При увеличении размера объекта волна сильнее искажается.")
    print("Это видно по изменению картины поля и сигнала на детекторе.")


if __name__ == "__main__":
    main()