# При попытке установки библиотеки EMEPy возникла ошибка,
# связанная с ограничением длины пути к файлам в операционной системе Windows.


from dataclasses import dataclass
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from bpm.core import run_bpm
from bpm.mode_solver import slab_mode_source
from bpm.refractive_index import generate_MMI_n_r2


SAVE_DIR = Path("bpm_results")
SAVE_DIR.mkdir(exist_ok=True)


@dataclass
class DeviceConfig:
    # параметры материала
    lam_um: float = 1.55
    n_bg: float = 1.444
    n_core: float = 3.47

    # геометрия входа / выхода / MMI
    input_wg_width_um: float = 0.50
    output_pitch_um: float = 2.00
    mmi_section_width_um: float = 8.00

    # длины секций
    lead_in_um: float = 10.0
    mmi_um: float = 60.0
    lead_out_um: float = 10.0

    # численная сетка
    step_x_um: float = 0.02
    step_z_um: float = 0.002

    # параметры поглощающих областей
    absorber_um: float = 1.0
    absorber_strength: float = 3.0
    absorber_order: int = 2

    @property
    def full_length_um(self) -> float:
        return self.lead_in_um + self.mmi_um + self.lead_out_um

    @property
    def simulation_width_um(self) -> float:
        # добавляем запас вокруг MMI + две поглощающие области
        return self.mmi_section_width_um + 2 * self.absorber_um + 2.0


def make_transverse_grid(cfg: DeviceConfig) -> np.ndarray:
    half_width = cfg.simulation_width_um / 2
    return np.arange(-half_width, half_width + cfg.step_x_um, cfg.step_x_um)


def make_longitudinal_grid(cfg: DeviceConfig) -> np.ndarray:
    return np.arange(0.0, cfg.full_length_um + cfg.step_z_um, cfg.step_z_um)


def build_absorber_profile(x_axis: np.ndarray, thickness_um: float, peak: float, degree: int) -> np.ndarray:
    """
    Возвращает одномерный профиль затухания у левой и правой границ области расчёта.
    Профиль строится плавно, чтобы уменьшить численные отражения.
    """
    sigma = np.zeros_like(x_axis, dtype=float)

    x_left = x_axis[0]
    x_right = x_axis[-1]

    active_left = x_left + thickness_um
    active_right = x_right - thickness_um

    for idx, x_val in enumerate(x_axis):
        if x_val < active_left:
            rel = (active_left - x_val) / thickness_um
            sigma[idx] = peak * rel**degree
        elif x_val > active_right:
            rel = (x_val - active_right) / thickness_um
            sigma[idx] = peak * rel**degree

    return sigma


def create_index_landscape(cfg: DeviceConfig, x_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    """
    Формирует карту n^2 для структуры:
    входной волновод -> MMI секция -> два выходных канала.
    """
    return generate_MMI_n_r2(
        x=x_axis,
        z=z_axis,
        z_MMI_start=cfg.lead_in_um,
        L_MMI=cfg.mmi_um,
        w_MMI=cfg.mmi_section_width_um,
        w_wg=cfg.input_wg_width_um,
        d=cfg.output_pitch_um,
        n_WG=cfg.n_core,
        n_MMI=cfg.n_core,
        n0=cfg.n_bg,
    )


def make_input_field(cfg: DeviceConfig, x_axis: np.ndarray) -> np.ndarray:
    """
    Возбуждаем фундаментальную моду узкого входного волновода в центре.
    """
    return slab_mode_source(
        x=x_axis,
        w=cfg.input_wg_width_um,
        n_WG=cfg.n_core,
        n0=cfg.n_bg,
        wavelength=cfg.lam_um,
        ind_m=0,
        x0=0.0,
    )


def initialize_field_array(nx: int, nz: int, source_profile: np.ndarray) -> np.ndarray:
    field = np.zeros((nx, nz), dtype=np.complex128)
    field[:, 0] = source_profile
    return field


def propagate_field(
    cfg: DeviceConfig,
    field_0: np.ndarray,
    n_sq_map: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    sigma_x: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Запуск BPM-распространения.
    """
    t_start = time.perf_counter()

    field_z = run_bpm(
        E=field_0,
        n_r2=n_sq_map,
        x=x_axis,
        z=z_axis,
        dx=cfg.step_x_um,
        dz=cfg.step_z_um,
        n0=cfg.n_bg,
        sigma_x=sigma_x,
        wavelength=cfg.lam_um,
    )

    elapsed = time.perf_counter() - t_start
    return field_z, elapsed


def estimate_output_split(
    intensity_map: np.ndarray,
    x_axis: np.ndarray,
    waveguide_width_um: float,
    center_distance_um: float,
) -> tuple[float, float, float, float]:
    """
    Оценивает мощность в верхнем и нижнем выходных каналах на последнем срезе.
    """
    last_slice = -1
    shift = center_distance_um / 2

    upper_window = np.abs(x_axis - shift) <= waveguide_width_um
    lower_window = np.abs(x_axis + shift) <= waveguide_width_um

    p_upper = np.trapezoid(intensity_map[upper_window, last_slice], x_axis[upper_window])
    p_lower = np.trapezoid(intensity_map[lower_window, last_slice], x_axis[lower_window])

    total = p_upper + p_lower + 1e-30
    ratio_upper = p_upper / total
    ratio_lower = p_lower / total

    return p_upper, p_lower, ratio_upper, ratio_lower


def save_report(runtime_s: float, p_up: float, p_down: float, r_up: float, r_down: float) -> None:
    report_path = SAVE_DIR / "summary_alt.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Результаты BPM-моделирования MMI\n")
        f.write(f"Время расчёта, с: {runtime_s:.6f}\n")
        f.write(f"Мощность в верхнем выходе: {p_up:.6e}\n")
        f.write(f"Мощность в нижнем выходе: {p_down:.6e}\n")
        f.write(f"Доля мощности в верхнем канале: {r_up:.6f}\n")
        f.write(f"Доля мощности в нижнем канале: {r_down:.6f}\n")


def plot_index_map(n_sq_map: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray, cfg: DeviceConfig) -> None:
    plt.figure(figsize=(9, 4.5))
    plt.imshow(
        n_sq_map,
        origin="lower",
        aspect="auto",
        extent=[z_axis[0], z_axis[-1], x_axis[0], x_axis[-1]],
        cmap="cividis",
        interpolation="nearest",
    )
    plt.colorbar(label=r"$n^2$")
    plt.axvline(cfg.lead_in_um, color="cyan", linestyle=":", linewidth=1.5, label="Начало MMI")
    plt.axvline(cfg.lead_in_um + cfg.mmi_um, color="magenta", linestyle="--", linewidth=1.5, label="Конец MMI")
    plt.xlabel("Координата z, мкм")
    plt.ylabel("Координата x, мкм")
    plt.title("Карта квадрата показателя преломления")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "index_map_alt.png", dpi=220)
    plt.close()


def plot_intensity_map(intensity_map: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray, cfg: DeviceConfig) -> None:
    plt.figure(figsize=(9, 4.5))
    plt.imshow(
        intensity_map,
        origin="lower",
        aspect="auto",
        extent=[z_axis[0], z_axis[-1], x_axis[0], x_axis[-1]],
        cmap="magma",
        interpolation="bilinear",
    )
    plt.colorbar(label=r"$|E|^2$")
    plt.axvline(cfg.lead_in_um, color="white", linestyle=":", linewidth=1.2)
    plt.axvline(cfg.lead_in_um + cfg.mmi_um, color="white", linestyle="--", linewidth=1.2)
    plt.xlabel("Координата z, мкм")
    plt.ylabel("Координата x, мкм")
    plt.title("Распределение интенсивности в MMI-структуре")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "intensity_map_alt.png", dpi=220)
    plt.close()


def plot_output_profile(intensity_map: np.ndarray, x_axis: np.ndarray, cfg: DeviceConfig) -> None:
    """
    Дополнительный график: поперечный профиль интенсивности на выходе.
    """
    plt.figure(figsize=(7, 4))
    plt.plot(x_axis, intensity_map[:, -1], linewidth=2)
    plt.axvline(+cfg.output_pitch_um / 2, linestyle="--", linewidth=1.2, label="Верхний канал")
    plt.axvline(-cfg.output_pitch_um / 2, linestyle="--", linewidth=1.2, label="Нижний канал")
    plt.xlabel("Координата x, мкм")
    plt.ylabel(r"Интенсивность $|E|^2$")
    plt.title("Поперечный профиль на выходной грани")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "output_profile_alt.png", dpi=220)
    plt.close()


def run_simulation() -> None:
    cfg = DeviceConfig()

    print("[BPM] Подготовка сетки...")
    x_axis = make_transverse_grid(cfg)
    z_axis = make_longitudinal_grid(cfg)

    print("[BPM] Построение структуры...")
    n_sq_map = create_index_landscape(cfg, x_axis, z_axis)

    print("[BPM] Формирование входной моды...")
    launch_field = make_input_field(cfg, x_axis)

    print("[BPM] Инициализация массива поля...")
    field = initialize_field_array(len(x_axis), len(z_axis), launch_field)

    print("[BPM] Построение поглощающего профиля по краям области...")
    absorber = build_absorber_profile(
        x_axis=x_axis,
        thickness_um=cfg.absorber_um,
        peak=cfg.absorber_strength,
        degree=cfg.absorber_order,
    )

    print("[BPM] Запуск распространения...")
    field, runtime_s = propagate_field(
        cfg=cfg,
        field_0=field,
        n_sq_map=n_sq_map,
        x_axis=x_axis,
        z_axis=z_axis,
        sigma_x=absorber,
    )

    intensity = np.abs(field) ** 2

    print("[BPM] Расчёт выходных мощностей...")
    p_up, p_down, r_up, r_down = estimate_output_split(
        intensity_map=intensity,
        x_axis=x_axis,
        waveguide_width_um=cfg.input_wg_width_um,
        center_distance_um=cfg.output_pitch_um,
    )

    print("[BPM] Сохранение графиков и отчёта...")
    plot_index_map(n_sq_map, x_axis, z_axis, cfg)
    plot_intensity_map(intensity, x_axis, z_axis, cfg)
    plot_output_profile(intensity, x_axis, cfg)
    save_report(runtime_s, p_up, p_down, r_up, r_down)

    print("[BPM] Готово.")
    print(f"[BPM] Время расчёта: {runtime_s:.3f} с")
    print(f"[BPM] Верхний выход: {p_up:.6e}")
    print(f"[BPM] Нижний выход: {p_down:.6e}")
    print(f"[BPM] Деление мощности: верхний = {r_up:.4f}, нижний = {r_down:.4f}")


if __name__ == "__main__":
    run_simulation()