import dataclasses
import typing
import pathlib
import csv
import re
from collections import defaultdict

import numpy as np

import csv
import dataclasses
import pathlib
import re
import typing


@dataclasses.dataclass(frozen=True)
class LossProfile:
    max_elevation: float
    time: tuple[float, ...]
    atmospheric: tuple[float, ...]
    diffraction: tuple[float, ...]
    elevation: tuple[float, ...]
    distance: tuple[float, ...]

    @property
    def name(self) -> str:
        return f'Max elevation: {self.max_elevation:g}°'

    @property
    def total_loss(self) -> list[float]:
        return [
            atmospheric + diffraction
            for atmospheric, diffraction in zip(
                self.atmospheric,
                self.diffraction,
                strict=True,
            )
        ]

    def __post_init__(self) -> None:
        lengths = {
            len(self.time),
            len(self.atmospheric),
            len(self.diffraction),
            len(self.elevation),
            len(self.distance),
        }

        if len(lengths) != 1:
            raise ValueError(
                'All loss-profile arrays must have equal lengths'
            )

        if not self.time:
            raise ValueError(
                'A loss profile must contain at least one sample'
            )


def _natural_keys(text: str) -> list[typing.Union[int, str]]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r'(\d+)', text)
    ]


def get_loss_profile(
    max_elevation: float,
    loss_profiles: list[LossProfile],
) -> LossProfile:
    for profile in loss_profiles:
        if profile.max_elevation == max_elevation:
            return profile

    raise ValueError(f'No loss profile for {max_elevation:g}°')


def get_loss_profiles(
    directory: pathlib.Path,
) -> list[LossProfile]:
    loss_profile_files = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.casefold() == '.csv'
        ),
        key=lambda path: _natural_keys(path.name),
    )

    loss_profiles: list[LossProfile] = []
    seen_elevations: set[float] = set()

    for file in loss_profile_files:
        match = re.search(
            pattern=r'_maxelev_(\d+(?:\.\d+)?)degrees$',
            string=file.stem,
            flags=re.IGNORECASE,
        )

        if match is None:
            raise ValueError(
                'Could not determine maximum elevation '
                f'from filename: {file.name}'
            )

        max_elevation = float(match.group(1))

        if max_elevation in seen_elevations:
            raise ValueError(
                f'Duplicate loss profile for '
                f'{max_elevation:g}°: {file.name}'
            )

        seen_elevations.add(max_elevation)

        time: list[float] = []
        atmospheric: list[float] = []
        diffraction: list[float] = []
        elevation: list[float] = []
        distance: list[float] = []

        with file.open(
            mode='r',
            newline='',
            encoding='utf-8',
        ) as csvfile:
            csvreader = csv.DictReader(
                f=csvfile,
                delimiter=',',
            )

            headers = csvreader.fieldnames

            if headers is None:
                raise ValueError(f'No headers found in {file}')

            if len(headers) != 5:
                raise ValueError(
                    f'Expected 5 columns in {file}, '
                    f'got {len(headers)}'
                )

            for row_number, row in enumerate(iterable=csvreader, start=2):
                try:
                    time.append(float(row[headers[0]]))
                    atmospheric.append(float(row[headers[1]]))
                    diffraction.append(float(row[headers[2]]))
                    elevation.append(float(row[headers[3]]))
                    distance.append(float(row[headers[4]]))
                except (TypeError, ValueError, KeyError) as error:
                    raise ValueError(
                        f'Invalid data in {file}, '
                        f'row {row_number}'
                    ) from error

        loss_profiles.append(
            LossProfile(
                max_elevation=max_elevation,
                time=tuple(time),
                atmospheric=tuple(atmospheric),
                diffraction=tuple(diffraction),
                elevation=tuple(elevation),
                distance=tuple(distance),
            )
        )

    return loss_profiles

def optimal_power_curve(angles, pwrs, values):
    '''
    Given scattered (angle, power, value) samples, return arrays:
    (unique_angles_sorted, opt_power_for_each_angle, opt_value_for_each_angle)

    Only uses points where value is finite.
    '''
    by_angle = defaultdict(list)
    for a, p, v in zip(angles, pwrs, values):
        if v is None:
            continue
        if not np.isfinite(v):
            continue
        by_angle[int(a)].append((float(p), float(v)))

    ang_sorted = np.array(sorted(by_angle.keys()), dtype=int)
    opt_p = np.full_like(ang_sorted, np.nan, dtype=float)
    opt_v = np.full_like(ang_sorted, np.nan, dtype=float)

    for i, a in enumerate(ang_sorted):
        pts = by_angle[a]
        # pick power that maximises v
        p_best, v_best = max(pts, key=lambda t: t[1])
        opt_p[i] = p_best
        opt_v[i] = v_best

    return ang_sorted, opt_p, opt_v
