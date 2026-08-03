import pathlib
import typing
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.axes as axes

from uplink_qkd import rates
from uplink_qkd.loss_parse import LossProfile, get_loss_profiles

DATA_DIR = pathlib.Path.cwd().joinpath('data').resolve()
params_10km_fibre = [9.47893858965945, 9.89101987420694, 0.037228590019520497, 0.03834132110856429, 6135831.8248959305, 1e-09]
PARAMS = params_10km_fibre

SAVE_DATA_POINTS = True
SAVE_FIGURE = True

DATA_POINTS_FILENAME = 'Fig2.csv'
FIGURE_FILENAME = 'paper_fig_2.pdf'

FONTSIZE = 16
TICK_FONTSIZE = 14
DPI = 300

def loss_plot(
        loss_profiles: list[LossProfile],
        max_elevation_range: range = range(30,91),
        ax: typing.Optional[axes.Axes] = None,
        xlim: typing.Optional[tuple[float, float]] = None,
        xlabel: typing.Optional[str] = None,
        fontsize: int = 16,
        tick_fontsize: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Contour plot of channel loss as a function of maximum elevation angle
    '''
    if ax is None:
        fig, ax = plt.subplots()

    if xlabel:
        ax.set_xlabel(xlabel=xlabel, fontsize=16)
    # ax.set_ylabel(ylabel=r'$\phi_\text{max}$ (°)', fontsize=16)

    if xlim is not None:
        ax.set_xlim(xlim[0],xlim[1])

    loss_profiles = [
        loss_profile for loss_profile in loss_profiles
        if int(loss_profile.name.split(' ')[-1].split('°')[0])
        in max_elevation_range
    ]

    X, Y = np.meshgrid(
        loss_profiles[np.argmax([np.array(l.time).shape[0] for l in loss_profiles])].time,
        max_elevation_range
    )
    Z_loss: list[np.ndarray] = []
    max_length = np.max([np.array(l.time).shape[0] for l in loss_profiles])
    for l in loss_profiles:
        buffer_length = (max_length - np.array(l.time).shape[0])//2
        if np.array(l.time).shape[0] % 2 == max_length % 2:
            Z_loss.append(
                np.pad(
                    array=np.array(l.atmospheric) + np.array(l.diffraction),
                    pad_width=(buffer_length, buffer_length),
                    mode='constant',
                    constant_values=np.nan
                )
            )
        else:
            Z_loss.append(
                np.pad(
                    array=np.array(l.atmospheric) + np.array(l.diffraction),
                    pad_width=(buffer_length+1, buffer_length),
                    mode='constant',
                    constant_values=np.nan
                )
            )
    contour_set = ax.contour(
        X,
        Y,
        Z_loss,
        levels=[45,50,55,60,70],
        colors='white'
    )
    filled_contour_set = ax.contourf(
        X,
        Y,
        Z_loss,
        levels=np.linspace(38,85,50),
        cmap='inferno_r'
    )
    cbar = ax.figure.colorbar(mappable=filled_contour_set)
    cbar.set_label(
        label='Loss (dB)',
        fontsize=fontsize,
    )
    cbar.ax.yaxis.set_major_formatter(
        ticker.FormatStrFormatter('%0.1f')
    )
    ax.clabel(
        CS=contour_set,
        colors='white'
    )
    ax.tick_params(labelsize=tick_fontsize)
    ax.set_yticks(ticks=range(30, 91, 10))
    ax.set_xticks(ticks=range(-300, 301, 100))

    return X, Y, np.array(Z_loss)

def qber_plot(
        loss_profiles: list[LossProfile],
        params: list[float],
        max_elevation_range: range = range(30,91),
        ax: typing.Optional[axes.Axes] = None,
        xlim: typing.Optional[tuple[float, float]] = None,
        xlabel: typing.Optional[str] = None,
        fontsize: int = 16,
        tick_fontsize: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Contour plot of instantaneous QBER as a function of maximum elevation angle
    '''
    if ax is None:
        fig, ax = plt.subplots()

    if xlabel:
        ax.set_xlabel(xlabel=xlabel, fontsize=16)
    # ax.set_ylabel(ylabel=r'$\phi_\text{max}$ (°)', fontsize=16)

    if xlim is not None:
        ax.set_xlim(xlim[0],xlim[1])

    loss_profiles = [
        loss_profile for loss_profile in loss_profiles
        if int(loss_profile.name.split(' ')[-1].split('°')[0])
        in max_elevation_range
    ]

    X, Y = np.meshgrid(
        loss_profiles[np.argmax([np.array(l.time).shape[0] for l in loss_profiles])].time,
        max_elevation_range
    )
    Z_qber: list[np.ndarray] = []
    max_length = np.max([np.array(l.time).shape[0] for l in loss_profiles])
    for l in loss_profiles:
        buffer_length = (max_length - np.array(l.time).shape[0])//2
        if np.array(l.time).shape[0] % 2 == max_length % 2:
            Z_qber.append(
                np.pad(
                    array=rates.raw_overpass_instant(
                        params=params,
                        loss_profile=np.array(l.atmospheric)+np.array(l.diffraction)
                    )[0],
                    pad_width=(buffer_length, buffer_length),
                    mode='constant',
                    constant_values=np.nan
                )
            )
        else:
            Z_qber.append(
                np.pad(
                    array=rates.raw_overpass_instant(
                        params=params,
                        loss_profile=np.array(l.atmospheric)+np.array(l.diffraction)
                    )[0],
                    pad_width=(buffer_length+1, buffer_length),
                    mode='constant',
                    constant_values=np.nan
                )
            )
    contour_set = ax.contour(
        X,
        Y,
        Z_qber,
        levels=[0.11],
        colors='white'
    )
    filled_contour_set = ax.contourf(
        X,
        Y,
        Z_qber,
        levels=np.linspace(0,0.5,50),
        cmap='inferno_r'
    )
    cbar = ax.figure.colorbar(mappable=filled_contour_set)
    cbar.set_label(
        label='QBER',
        fontsize=16
    )
    cbar.ax.yaxis.set_major_formatter(
        ticker.FormatStrFormatter('%0.3f')
    )
    ax.clabel(
        CS=contour_set,
        colors='white'
    )
    ax.tick_params(labelsize=tick_fontsize)
    ax.set_yticks(ticks=range(30, 91, 10))
    ax.set_xticks(ticks=range(-300, 301, 100))

    return X, Y, np.array(Z_qber)


if __name__ == '__main__':
    loss_profile_dir = DATA_DIR.joinpath('up_link_passes')
    loss_profiles = get_loss_profiles(directory=loss_profile_dir)

    fig, ax = plt.subplots(
        nrows=2,ncols=1,
        dpi=DPI,
        constrained_layout=True
    )

    fig.supxlabel(
        t='Time (s)',
        fontsize=FONTSIZE
    )
    fig.supylabel(
        t=r'$\phi_\text{max}$ (°)',
        fontsize=FONTSIZE
    )

    top_x, top_y, top_z = loss_plot(
        loss_profiles=loss_profiles,
        ax=ax[0],
        xlim=(-300,300)
    )
    top_points = np.column_stack((top_x.ravel(), top_y.ravel(), top_z.ravel()))

    bottom_x, bottom_y, bottom_z = qber_plot(
        loss_profiles=loss_profiles,
        params=PARAMS,
        ax=ax[1],
        xlim=(-300,300),
    )
    bottom_points = np.column_stack((bottom_x.ravel(), bottom_y.ravel(), bottom_z.ravel()))

    if SAVE_DATA_POINTS:
        with open(file=DATA_POINTS_FILENAME, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Top contour', 'x', 'y', 'z', '', 'Bottom contour', 'x', 'y', 'z'])
            for i in range(len(top_points)):
                writer.writerow(['', *top_points[i], '', '', *bottom_points[i]])

    if SAVE_FIGURE:
        fig.savefig(fname=FIGURE_FILENAME, dpi=DPI)
    plt.show()