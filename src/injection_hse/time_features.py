"""Periodic time-of-day / week / year features with a validity flag.

A profiled coherent dataset (:class:`injection_hse.data.scenarios.CoherentSpectrumConfig` with a
:class:`~injection_hse.data.scenarios.LoadProfileConfig`) stamps every step with an absolute UTC epoch
time (``samples["time_unix_s"]``). This module turns that epoch time into a small,
wrap-free periodic feature vector the state estimator conditions on, so the model can learn
the diurnal / weekly / seasonal structure of the load profiles.

The vector is, in order::

    [sin(2π·tod), cos(2π·tod),      # time of day
     sin(2π·dow/7), cos(2π·dow/7),  # day of week
     sin(2π·doy), cos(2π·doy),      # day of year (season)
     valid]

with ``tod`` the fraction of the day (UTC), ``dow`` the continuous day-of-week index
(Monday 0 .. Sunday 6, fractional over the day), and ``doy`` the fraction of a 365.25-day
year on the SAME fixed epoch frame :mod:`injection_hse.data.scenarios.profiles` uses (so the seasonal
phase agrees with the generator). ``valid`` is 1 for a real timestamp.

Datasets WITHOUT an absolute timestamp — the randomized snapshot stage, or a non-profiled
coherent sequence (which records only a RELATIVE ``time_s``) — carry no ``time_unix_s``: the
estimator then receives the ALL-ZERO vector (``valid = 0``), the SAME input contract, so ONE
model trains across both timeless snapshots and timestamped profiled sequences. The
consuming model synthesizes that zero block when no timestamp is supplied
(:func:`injection_hse.models.base.broadcast_global_features`).

Plain differentiable torch on real dtypes; runs on CPU/CUDA unchanged and acts on the
trailing (epoch) axis, so any ``[*shape]`` epoch tensor maps to ``[*shape, TIME_FEATURE_DIM]``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

#: number of real time-feature channels (three sin/cos pairs + a validity flag).
TIME_FEATURE_DIM = 7

_SECONDS_PER_DAY = 86400.0
_SECONDS_PER_YEAR = 365.25 * _SECONDS_PER_DAY
# 1970-01-01 (epoch day 0) was a Thursday; Python weekday() has Monday=0, so adding 3 maps
# epoch day 0 to weekday index 3 (Thursday). Matches injection_hse.data.scenarios.profiles so the weekly
# phase is identical to the one the profile generator used.
_EPOCH_WEEKDAY_OFFSET = 3.0

_TWO_PI = 2.0 * math.pi


def time_features(time_unix_s: Tensor) -> Tensor:
    """Periodic time features ``[*shape, TIME_FEATURE_DIM]`` for epoch seconds ``[*shape]``.

    ``time_unix_s`` is absolute UTC epoch seconds (any float dtype; promoted to float64 for
    the modular arithmetic so ~1.7e9 second magnitudes keep their sub-second precision). The
    ``valid`` channel is set to 1. Timeless samples are represented by the all-zero vector
    (see :func:`injection_hse.models.base.broadcast_global_features`), not produced here.
    """
    t = time_unix_s.to(torch.float64)
    tod = torch.remainder(t, _SECONDS_PER_DAY) / _SECONDS_PER_DAY
    dow = torch.remainder(t / _SECONDS_PER_DAY + _EPOCH_WEEKDAY_OFFSET, 7.0) / 7.0
    doy = torch.remainder(t, _SECONDS_PER_YEAR) / _SECONDS_PER_YEAR
    ang_tod = _TWO_PI * tod
    ang_dow = _TWO_PI * dow
    ang_doy = _TWO_PI * doy
    valid = torch.ones_like(t)
    return torch.stack(
        [
            torch.sin(ang_tod),
            torch.cos(ang_tod),
            torch.sin(ang_dow),
            torch.cos(ang_dow),
            torch.sin(ang_doy),
            torch.cos(ang_doy),
            valid,
        ],
        dim=-1,
    )


__all__ = ["TIME_FEATURE_DIM", "time_features"]
