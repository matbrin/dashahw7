import numpy as np
import matplotlib.pyplot as plt


def run_fdtd_un_split_pml():
    """
    1D FDTD симуляция с аддитивным источником и un-split PML на двух границах.
    Сравниваются три случая:
    1) без PML
    2) тонкий PML
    3) толстый PML
    """

    # ОБЩИЕ ПАРАМЕТРЫ СИМУЛЯЦИИ
    SIZE = 400                 # Количество узлов сетки
    MAX_TIME = 700             # Количество временных шагов
    IMP0 = 377.0               # Волновое сопротивление свободного пространства

    SOURCE_POSITION = 120      # Положение источника
    SOURCE_CENTER = 60         # Центр гауссова импульса
    SOURCE_WIDTH = 18          # Ширина гауссова импульса

    DETECTOR_POSITION = 140    # Точка, где смотрим отражение

    # Временные шаги, в которые сохраняем снимки поля
    SNAPSHOT_TIMES = [120, 180, 240, 320]

    # После этого времени считаем, что на детектор возвращается отражённая волна
    REFLECTION_START_TIME = 220

    # ФУНКЦИЯ ИСТОЧНИКА
    def additive_source(q_time):
        """
        Гауссов импульс.
        """
        t = float(q_time)
        center = float(SOURCE_CENTER)
        width = float(SOURCE_WIDTH)
        return np.exp(-((t - center) ** 2) / (width ** 2))

    # СОЗДАНИЕ ПРОФИЛЯ PML
    def build_pml_arrays(pml_thickness, sigma_max, kappa_max, alpha_max, grading_order):
        """
        Создаёт массивы коэффициентов PML
        """

        # Если PML нет, возвращаем обычные коэффициенты
        if pml_thickness == 0:
            kappa_e = np.ones(SIZE)
            kappa_h = np.ones(SIZE)
            b_e = np.zeros(SIZE)
            b_h = np.zeros(SIZE)
            c_e = np.zeros(SIZE)
            c_h = np.zeros(SIZE)
            return kappa_e, kappa_h, b_e, b_h, c_e, c_h

        # Массивы для электрического и магнитного поля
        sigma_e = np.zeros(SIZE)
        sigma_h = np.zeros(SIZE)

        kappa_e = np.ones(SIZE)
        kappa_h = np.ones(SIZE)

        alpha_e = np.zeros(SIZE)
        alpha_h = np.zeros(SIZE)

        # Профиль для Ez
        for mm in range(SIZE):
            # Левая граница
            if mm < pml_thickness:
                dist = (pml_thickness - mm) / pml_thickness
                sigma_e[mm] = sigma_max * (dist ** grading_order)
                kappa_e[mm] = 1.0 + (kappa_max - 1.0) * (dist ** grading_order)
                alpha_e[mm] = alpha_max * (1.0 - dist)

            # Правая граница
            elif mm >= SIZE - pml_thickness:
                dist = (mm - (SIZE - pml_thickness - 1)) / pml_thickness
                sigma_e[mm] = sigma_max * (dist ** grading_order)
                kappa_e[mm] = 1.0 + (kappa_max - 1.0) * (dist ** grading_order)
                alpha_e[mm] = alpha_max * (1.0 - dist)

        # Профиль для Hy
        # Здесь берём похожий профиль, этого достаточно для 1D задачи
        for mm in range(SIZE):
            # Левая граница
            if mm < pml_thickness:
                dist = (pml_thickness - mm - 0.5) / pml_thickness
                if dist < 0:
                    dist = 0.0
                sigma_h[mm] = sigma_max * (dist ** grading_order)
                kappa_h[mm] = 1.0 + (kappa_max - 1.0) * (dist ** grading_order)
                alpha_h[mm] = alpha_max * (1.0 - dist)

            # Правая граница
            elif mm >= SIZE - pml_thickness:
                dist = (mm - (SIZE - pml_thickness) + 0.5) / pml_thickness
                if dist < 0:
                    dist = 0.0
                sigma_h[mm] = sigma_max * (dist ** grading_order)
                kappa_h[mm] = 1.0 + (kappa_max - 1.0) * (dist ** grading_order)
                alpha_h[mm] = alpha_max * (1.0 - dist)

        # Коэффициенты рекурсивной свёртки
        b_e = np.zeros(SIZE)
        b_h = np.zeros(SIZE)
        c_e = np.zeros(SIZE)
        c_h = np.zeros(SIZE)

        for mm in range(SIZE):
            if sigma_e[mm] > 0.0 or alpha_e[mm] > 0.0:
                decay_e = alpha_e[mm] + sigma_e[mm] / kappa_e[mm]
                b_e[mm] = np.exp(-decay_e)
                denom_e = sigma_e[mm] * kappa_e[mm] + (kappa_e[mm] ** 2) * alpha_e[mm]
                if denom_e != 0.0:
                    c_e[mm] = sigma_e[mm] * (b_e[mm] - 1.0) / denom_e

            if sigma_h[mm] > 0.0 or alpha_h[mm] > 0.0:
                decay_h = alpha_h[mm] + sigma_h[mm] / kappa_h[mm]
                b_h[mm] = np.exp(-decay_h)
                denom_h = sigma_h[mm] * kappa_h[mm] + (kappa_h[mm] ** 2) * alpha_h[mm]
                if denom_h != 0.0:
                    c_h[mm] = sigma_h[mm] * (b_h[mm] - 1.0) / denom_h

        return kappa_e, kappa_h, b_e, b_h, c_e, c_h

    # ОДИН ЗАПУСК СИМУЛЯЦИИ
    def run_single_case(case_name, pml_thickness, sigma_max, kappa_max, alpha_max, grading_order):
        """
        Выполняет один запуск FDTD.
        """

        # ИНИЦИАЛИЗАЦИЯ ПОЛЕЙ
        ez = np.zeros(SIZE)
        hy = np.zeros(SIZE)

        # Вспомогательные переменные PML
        psi_e = np.zeros(SIZE)
        psi_h = np.zeros(SIZE)

        # Получаем коэффициенты PML
        kappa_e, kappa_h, b_e, b_h, c_e, c_h = build_pml_arrays(
            pml_thickness, sigma_max, kappa_max, alpha_max, grading_order
        )

        # Для хранения результатов
        detector_trace = []
        snapshots = []

        # ОСНОВНОЙ ЦИКЛ FDTD
        for q_time in range(MAX_TIME):

            # Обновление магнитного поля H-
            for mm in range(SIZE - 1):
                d_e = ez[mm + 1] - ez[mm]

                # Обновление памяти PML для H
                psi_h[mm] = c_h[mm] * d_e + b_h[mm] * psi_h[mm]

                # Обычное обновление + добавка PML
                hy[mm] += (d_e / (IMP0 * kappa_h[mm])) + psi_h[mm] / IMP0

            # Обновление электрического поля E
            for mm in range(1, SIZE):
                d_h = hy[mm] - hy[mm - 1]

                # Обновление памяти PML для E
                psi_e[mm] = c_e[mm] * d_h + b_e[mm] * psi_e[mm]

                # Обычное обновление + добавка PML
                ez[mm] += (d_h * IMP0 / kappa_e[mm]) + psi_e[mm] * IMP0

            # Аддитивный источник
            ez[SOURCE_POSITION] += additive_source(q_time)

            # Сохраняем сигнал детектора
            detector_trace.append(ez[DETECTOR_POSITION])

            # Сохраняем снимки поля
            if q_time in SNAPSHOT_TIMES:
                snapshots.append((q_time, ez.copy()))

        # МЕТРИКА ОТРАЖЕНИЯ
        incident_peak = np.max(np.abs(detector_trace[:REFLECTION_START_TIME]))
        reflected_peak = np.max(np.abs(detector_trace[REFLECTION_START_TIME:]))

        if incident_peak > 0.0:
            reflection_metric = reflected_peak / incident_peak
        else:
            reflection_metric = 0.0

        # Возвращаем всё, что нужно
        return {
            "case_name": case_name,
            "ez_final": ez.copy(),
            "hy_final": hy.copy(),
            "snapshots": snapshots,
            "detector_trace": np.array(detector_trace),
            "reflection_metric": reflection_metric,
            "pml_thickness": pml_thickness,
            "kappa_e": kappa_e,
            "kappa_h": kappa_h,
            "b_e": b_e,
            "b_h": b_h,
            "c_e": c_e,
            "c_h": c_h
        }

    # НАСТРОЙКИ ТРЁХ СЛУЧАЕВ
    cases = [
        {
            "case_name": "Без PML",
            "pml_thickness": 0,
            "sigma_max": 0.0,
            "kappa_max": 1.0,
            "alpha_max": 0.0,
            "grading_order": 2
        },
        {
            "case_name": "Тонкий PML",
            "pml_thickness": 20,
            "sigma_max": 0.08,
            "kappa_max": 2.5,
            "alpha_max": 0.02,
            "grading_order": 3
        },
        {
            "case_name": "Толстый PML",
            "pml_thickness": 40,
            "sigma_max": 0.08,
            "kappa_max": 3.5,
            "alpha_max": 0.02,
            "grading_order": 3
        }
    ]

    # ЗАПУСКАЕМ ВСЕ СЛУЧАИ
    results = []

    for case in cases:
        result = run_single_case(
            case_name=case["case_name"],
            pml_thickness=case["pml_thickness"],
            sigma_max=case["sigma_max"],
            kappa_max=case["kappa_max"],
            alpha_max=case["alpha_max"],
            grading_order=case["grading_order"]
        )
        results.append(result)

    # ВЫВОД В КОНСОЛЬ
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ PML")

    for result in results:
        print(f'Случай: {result["case_name"]}')
        print(f'Толщина PML: {result["pml_thickness"]}')
        print(f'Метрика отражения: {result["reflection_metric"]:.6e}')
        print("-" * 60)

    best_case = min(results[1:], key=lambda x: x["reflection_metric"])
    print(f'Лучший результат даёт: {best_case["case_name"]}')
    print()

    # ГРАФИК 1: СИГНАЛ НА ДЕТЕКТОРЕ
    plt.figure(figsize=(12, 6))

    for result in results:
        plt.plot(result["detector_trace"], linewidth=2, label=result["case_name"])

    plt.axvline(REFLECTION_START_TIME, color='black', linestyle='--',
                label='Начало области отражения')
    plt.xlabel('Временной шаг')
    plt.ylabel('Ez в точке детектора')
    plt.title('Сравнение сигналов на детекторе')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ГРАФИК 2: ФИНАЛЬНЫЙ ПРОФИЛЬ ЭЛЕКТРИЧЕСКОГО ПОЛЯ
    plt.figure(figsize=(12, 6))

    for result in results:
        plt.plot(result["ez_final"], linewidth=2, label=result["case_name"])

        pml = result["pml_thickness"]
        if pml > 0:
            plt.axvline(pml, color='gray', linestyle=':', alpha=0.5)
            plt.axvline(SIZE - pml, color='gray', linestyle=':', alpha=0.5)

    plt.axvline(SOURCE_POSITION, color='red', linestyle='--', label='Источник')
    plt.axvline(DETECTOR_POSITION, color='green', linestyle='--', label='Детектор')
    plt.xlabel('Индекс узла x')
    plt.ylabel('Ez')
    plt.title('Финальный профиль электрического поля')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ГРАФИКИ 3 И 4: СНИМКИ ПОЛЯ ДЛЯ ДВУХ PML
    for result in results[1:]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for idx, (t, field) in enumerate(result["snapshots"]):
            ax = axes[idx]

            ax.plot(field, 'b-', linewidth=1.5, label='Ez поле')
            ax.fill_between(range(len(field)), field, alpha=0.25)

            # Границы PML
            pml = result["pml_thickness"]
            ax.axvline(pml, color='gray', linestyle=':', linewidth=2, label='Граница PML')
            ax.axvline(SIZE - pml, color='gray', linestyle=':', linewidth=2)

            # Источник и линия нуля
            ax.axvline(SOURCE_POSITION, color='red', linestyle='--', linewidth=1.5, label='Источник')
            ax.axhline(0, color='black', linewidth=0.5)

            ax.set_xlim(0, SIZE - 1)
            ax.set_xlabel('Индекс узла (x)')
            ax.set_ylabel('Ez')
            ax.set_title(f'{result["case_name"]}, t = {t}')
            ax.grid(True, alpha=0.3)

            y_max = np.max(np.abs(field))
            if y_max > 0:
                ax.set_ylim(-1.15 * y_max, 1.15 * y_max)

            if idx == 0:
                ax.legend(loc='upper right')

        fig.suptitle(f'Снимки поля: {result["case_name"]}', fontsize=14)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    run_fdtd_un_split_pml()