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

loss_profile_dir = DATA_DIR.joinpath('550000m_0m_0.25m')
loss_profiles = loss_parse.get_loss_profiles(directory=loss_profile_dir)
loss_profile = loss_parse.get_loss_profile(
    max_elevation=90,
    loss_profiles=loss_profiles
)

SAVE_DATA_POINTS = True
SAVE_FIGURE = True

FIBRE = True
POWER_RANGE = np.linspace(0, 10, 100)

DPI = 300
FONTSIZE = 16
TICK_FONTSIZE = 14

if FIBRE:
    dc_range = range(0,401,10)
    params = params_10km_fibre
    FIGURE_FILENAME = 'paper_fig_4.pdf'
    DATA_POINTS_FILENAME = 'Fig4.csv'
else:
    dc_range = range(200,1401,55)
    params = params_no_fibre
    FIGURE_FILENAME = 'paper_fig_S3.pdf'
    DATA_POINTS_FILENAME = 'FigS3.csv'
    
# %%
def _akl_worker(
        total_loss: list[float],
        dc: int,
        params: list[float],
        power: int
) -> typing.Optional[tuple[ int, int, int]]:
    ps = params.copy()
    ps[4] = params[4] * power

    qber, qx, m = rates.raw_overpass(
        params=ps,
        loss_profile=total_loss,
        DC_B=dc
    )
    akl = (1-1.19*finite_key.h(qber)-finite_key.h(qx))/2 * m

    if akl > 0:
        return dc, power, akl
    return None

def _skl_worker(
        total_loss: list[float],
        dc: int,
        params: list[float],
        power: int
) -> typing.Optional[tuple[ int, int, int]]:
    ps = params.copy()
    ps[4] = params[4] * power

    qber, qx, m = rates.raw_overpass(
        params=ps,
        loss_profile=total_loss,
        DC_B=dc
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
        return dc, power, skl
    return None

# %%
def power_kl_dc_data(
        loss_profile: loss_parse.LossProfile,
        params: list[float],
        dc_range: range,
        power_range: np.ndarray,
        func: Callable
) -> tuple[list[int], list[int], list[int]]:
    jobs = []
    for dc in dc_range:
        for power in power_range:
            jobs.append((loss_profile.total_loss, dc, params, power))

    dcs, pwrs, kls = [], [], []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(func, *job) for job in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            dc, power, kl = res
            dcs.append(dc)
            pwrs.append(power)
            kls.append(kl)

    return dcs, pwrs, kls

# %%
akl_dcs, akl_pwrs, akls = power_kl_dc_data(
    loss_profile=loss_profile,
    params=params,
    dc_range=dc_range,
    power_range=POWER_RANGE,
    func=_akl_worker
)

# %%
skl_dcs, skl_pwrs, skls = power_kl_dc_data(
    loss_profile=loss_profile,
    params=params,
    dc_range=dc_range,
    power_range=POWER_RANGE,
    func=_skl_worker
)

if SAVE_DATA_POINTS:
    with open(DATA_POINTS_FILENAME, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Purple contour', 'x', 'y', 'z', '', 'Blue contour', 'x', 'y', 'z'])

        for akl_dc, akl_pwr, akl, skl_dc, skl_pwr, skl in zip(akl_dcs, akl_pwrs, akls, skl_dcs, skl_pwrs, skls):
            writer.writerow(['', akl_dc, akl_pwr, akl, '', '', skl_dc, skl_pwr, skl])


# %%
levels = [1e1, 1e2, 1e3, 3.16e3, 1e4, 1.78e4, 3.16e4, 5.62e4, 1e5]
def power_akl_dc(
        dcs: list[int],
        pwrs: list[int],
        akls: list[int],
        fontsize,
        tick_fontsize,
        ax: typing.Optional[axes.Axes] = None
) -> None:
    if ax is None:
        fig, ax = plt.subplots()

    cs = ax.tricontourf(
        dcs,
        pwrs,
        akls,
        levels=levels,
        norm=colors.LogNorm(),
        cmap='Purples'
    )

    ax.tick_params(labelsize=tick_fontsize)
    ax.set_yticks(ticks=range(0, 11, 2))

    if FIBRE:
        ax.set_xticks(ticks=range(0, 401, 100))
        ax.set_xlim(0,400)
    else:
        ax.set_xticks(ticks=range(200, 1401, 400))
        ax.set_xlim(200,1400)

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

def power_skl_dc(
        dcs: list[int],
        pwrs: list[int],
        skls: list[int],
        ax: typing.Optional[axes.Axes] = None,
        fontsize=16,
        tick_fontsize=12
    ) -> None:
    if ax is None:
        fig, ax = plt.subplots()
    
    cs = ax.tricontourf(
        dcs,
        pwrs,
        skls,
        levels=levels,
        norm=colors.LogNorm(),
        cmap='Blues'
    )

    ax.tick_params(labelsize=tick_fontsize)
    ax.set_yticks(ticks=range(0, 11, 2))

    if FIBRE:
        ax.set_xticks(ticks=range(0, 401, 100))
        ax.set_xlim(0,400)
    else:
        ax.set_xticks(ticks=range(200, 1401, 400))
        ax.set_xlim(200,1400)

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

fig, ax = plt.subplots(
    nrows=1,ncols=2,
    dpi=DPI,
    constrained_layout=True
)
fig.supxlabel(
    t='DC (cps)',
    fontsize=FONTSIZE
)
fig.supylabel(
    t=r'Power, $P$ (mW)',
    fontsize=FONTSIZE
)

power_akl_dc(
    dcs=akl_dcs,
    pwrs=akl_pwrs,
    akls=akls,
    fontsize=FONTSIZE,
    tick_fontsize=TICK_FONTSIZE,
    ax=ax[0]
)
power_skl_dc(
    dcs=skl_dcs,
    pwrs=skl_pwrs,
    skls=skls,
    fontsize=FONTSIZE,
    tick_fontsize=TICK_FONTSIZE,
    ax=ax[1]
)

if SAVE_FIGURE:
    fig.savefig(fname=FIGURE_FILENAME, dpi=DPI)
