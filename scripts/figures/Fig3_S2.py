# %%
import pathlib
import typing
from collections.abc import Callable
import math
import csv
import os
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.colors as colors

from uplink_qkd import finite_key, rates, loss_parse

params_no_fibre = [7.761328918344407, 7.542259886343475, 0.0335677812551474, 0.02531375770896907, 10826895.017621633, 4e-10]
params_10km_fibre = [9.47893858965945, 9.89101987420694, 0.037228590019520497, 0.03834132110856429, 6135831.8248959305, 1e-09]

cpu_count = os.cpu_count()
if cpu_count is None:
    max_workers = 1
else:
    max_workers = cpu_count - 2

DATA_DIR = pathlib.Path.cwd().joinpath('data').resolve()

SAVE_DATA_POINTS = True
SAVE_FIGURE = True

FIBRE = True
POWER_RANGE = np.linspace(0.1, 10, 50)

DPI  = 300
FONTSIZE = 16
TICK_FONTSIZE = 14

if FIBRE:
    loss_profile_dir = DATA_DIR.joinpath('550000m_0m_0.25m')
    max_elevation_range = range(30,151,1)
    params = params_10km_fibre
    DATA_POINTS_FILENAME = 'Fig3.csv'
    FIGURE_FILENAME = 'paper_fig_3.pdf'
else:
    loss_profile_dir = DATA_DIR.joinpath('15deg_550000m_0m_0.25m')
    max_elevation_range = range(15,151,1)
    params = params_no_fibre
    DATA_POINTS_FILENAME = 'FigS2.csv'
    FIGURE_FILENAME = 'paper_fig_S2.pdf'

loss_profiles = loss_parse.get_loss_profiles(directory=loss_profile_dir)


# %%
def _akl_worker(
        total_loss: list[float],
        angle: int,
        params: list[float],
        power: float
) -> typing.Optional[tuple[ int, float, float]]:
    ps = params.copy()
    ps[4] = params[4] * power

    qber, qx, m = rates.raw_overpass(
        params=ps,
        loss_profile=total_loss
    )
    akl = (1-1.19*finite_key.h(qber)-finite_key.h(qx))/2 * m

    if akl > 0:
        return angle, power, akl
    return None

def _skl_worker(
        total_loss: list[float],
        angle: int,
        params: list[float],
        power: float
) -> typing.Optional[tuple[ int, float, float]]:
    ps = params.copy()
    ps[4] = params[4] * power

    qber, qx, m = rates.raw_overpass(
        params=ps,
        loss_profile=total_loss
    )
    m /= 2.0
    delta = (qber + qx) / 2.0

    skl = finite_key.smart_optimise(
        m=m,
        delta=delta,
        eps_qkd=1e-6,
        t=math.log2(10**8),
        f=1.19
    ) * m

    if skl > 0:
        return angle, power, skl
    return None

# %%
def power_kl_max_elev_data(
        loss_profiles: list[loss_parse.LossProfile],
        params: list[float],
        max_elevation_range: range,
        power_range: np.ndarray,
        func: Callable
) -> tuple[list[int], list[int], list[int]]:
    filtered = []
    for prof in loss_profiles:
        angle = int(prof.name.split(' ')[-1].split('°')[0])
        if angle in max_elevation_range:
            filtered.append((prof.total_loss, angle))

    jobs = [
        (total_loss, angle, params, power)
        for (total_loss, angle) in filtered
        for power in power_range
    ]

    angles: list[int] = []
    pwrs: list[int] = []
    kls: list[int] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(func, *job) for job in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            angle, power, kl = res
            angles.append(angle)
            pwrs.append(power)
            kls.append(kl)

    return angles, pwrs, kls

# %%
akl_angles, akl_pwrs, akls = power_kl_max_elev_data(
    loss_profiles=loss_profiles,
    params=params,
    max_elevation_range=max_elevation_range,
    power_range=POWER_RANGE,
    func=_akl_worker
)

# %%
skl_angles, skl_pwrs, skls = power_kl_max_elev_data(
    loss_profiles=loss_profiles,
    params=params,
    max_elevation_range=max_elevation_range,
    power_range=POWER_RANGE,
    func=_skl_worker
)

if SAVE_DATA_POINTS:
    with open(DATA_POINTS_FILENAME, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Purple contour', 'x', 'y', 'z', '', 'Blue contour', 'x', 'y', 'z'])

        for akl_angle, akl_pwr, akl, skl_angle, skl_pwr, skl in zip(akl_angles, akl_pwrs, akls, skl_angles, skl_pwrs, skls):
            writer.writerow(['', akl_angle, akl_pwr, akl, '', '', skl_angle, skl_pwr, skl])

# %%
if FIBRE:
    levels = [1e1, 1e2, 3.3e2, 1e3, 3.16e3, 5.62e3, 1e4, 1.78e4, 3.16e4]
else:
    levels = [1e1, 1e2, 1e3, 1e4, 1.78e4, 3.16e4, 5.62e4, 1e5, 1.78e5]

def power_akl_max_elev(
        angles: list[int],
        pwrs: list[int],
        akls: list[int],
        fontsize,
        tick_fontsize,
        ax: typing.Optional[axes.Axes] = None
) -> None:
    if ax is None:
        fig, ax = plt.subplots()

    ang_opt, pow_opt, akl_opt = loss_parse.optimal_power_curve(
        angles=angles,
        pwrs=pwrs,
        values=akls
    )
    cs = ax.tricontourf(
        angles,
        pwrs,
        akls,
        levels=levels,
        norm=colors.LogNorm(),
        cmap='Purples'
    )
    ax.plot(
        ang_opt,
        pow_opt,
        linewidth=2,
        marker='.',
        linestyle='',
        color='red'
    )
    ax.tick_params(labelsize=tick_fontsize)
    ax.set_yticks(ticks=range(0, 11, 2))
    ax.set_xticks(ticks=range(15, 91, 15))
    ax.set_xlim(15,90)
    ax.grid(visible=True)

    cb = ax.figure.colorbar(
        mappable=cs,
        ax=ax,
        location='top',
        orientation='horizontal',
    )
    cb.ax.tick_params(labelsize=tick_fontsize)
    cb.ax.xaxis.set_ticks_position(position='top')
    cb.set_label(
        label='AKL (bits)',
        fontsize=fontsize
    )
    cb.ax.xaxis.set_label_position(position='top')

    last_x = ang_opt[-1]
    last_y = pow_opt[-1]
    ax.scatter(
        last_x,
        last_y,
        linewidth=24,
        marker='.',
        linestyle='',
        color='red',
        clip_on=False,
        zorder=10
    )
    ax.annotate(
        r'$P^\text{finite}_\text{opt}$',
        (last_x, last_y),
        xytext=(-40, 5),
        textcoords='offset points',
        color='red',
        fontsize=fontsize,
        ha='left',
        va='bottom'
    )

def power_skl_max_elev(
        angles: list[int],
        pwrs: list[int],
        skls: list[int],
        ax: typing.Optional[axes.Axes] = None,
        fontsize=16,
        tick_fontsize=12
    ) -> None:
    if ax is None:
        fig, ax = plt.subplots()
    
    ang_opt, pow_opt, skl_opt = loss_parse.optimal_power_curve(
        angles=angles,
        pwrs=pwrs,
        values=skls
    )
    cs = ax.tricontourf(
        angles,
        pwrs,
        skls,
        levels=levels,
        norm=colors.LogNorm(),
        cmap='Blues'
    )
    ax.plot(
        ang_opt,
        pow_opt,
        linewidth=2,
        marker='.',
        linestyle='',
        color='red'
    )
    ax.tick_params(labelsize=tick_fontsize)
    ax.set_yticks(ticks=range(0, 11, 2))
    ax.set_xticks(ticks=range(15, 91, 15))
    ax.set_xlim(15,90)
    ax.grid(visible=True)
    ax.invert_xaxis()

    cb = ax.figure.colorbar(
        mappable=cs,
        ax=ax,
        location='top',
        orientation='horizontal',
    )
    cb.ax.tick_params(labelsize=tick_fontsize)
    cb.ax.xaxis.set_ticks_position(position='top')
    cb.set_label(
        label='SKL (bits)',
        fontsize=fontsize
    )
    cb.ax.xaxis.set_label_position(position='top')

    last_x = ang_opt[-1]
    last_y = pow_opt[-1]
    ax.scatter(
        last_x,
        last_y,
        linewidth=24,
        marker='.',
        linestyle='',
        color='red',
        clip_on=False,
        zorder=10
    )
    ax.annotate(
        r'$P^\text{asym}_\text{opt}$',
        (last_x, last_y),
        xytext=(0, 5),
        textcoords='offset points',
        color='red',
        fontsize=fontsize,
        ha='left',
        va='bottom'
    )

fig, ax = plt.subplots(
    nrows=1,ncols=2,
    dpi=DPI,
    constrained_layout=True
)
fig.supxlabel(
    t=r'$\phi_\text{max}$ (°)',
    fontsize=FONTSIZE
)
fig.supylabel(
    t=r'Power, $P$ (mW)',
    fontsize=FONTSIZE
)

power_akl_max_elev(
    angles=akl_angles,
    pwrs=akl_pwrs,
    akls=akls,
    fontsize=FONTSIZE,
    tick_fontsize=TICK_FONTSIZE,
    ax=ax[0]
)
power_skl_max_elev(
    angles=skl_angles,
    pwrs=skl_pwrs,
    skls=skls,
    fontsize=FONTSIZE,
    tick_fontsize=TICK_FONTSIZE,
    ax=ax[1]
)

if SAVE_FIGURE:
    fig.savefig(fname=FIGURE_FILENAME, dpi=DPI)
