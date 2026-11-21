"""DRiC with a horizon-scaled ridge and correction width (docs/PROTOCOL.md).

The corrector of ridge_corrector.py keeps every constant the same at all horizons: ridge 1 and
8 DCT components for the correction vector and the guard. Here two of them follow the horizon H (r = H / 24):

  ridge   config.ridge * r                       (the same penalty on coefficients normalized by sqrt(H))
  width   8 * r components for the correction vector and the guard; config.local * r for the local ridge output
          (the same band of frequencies per step at every horizon)

The inputs are unchanged: the local ridge reads config.local DCT components of the frozen forecast and of the newest
input sub-blocks, the global ridge reads 8 of the forecast and of every input sub-block, both read the last value
and a constant. With both rules off the class is the original corrector (test_dric_scaled.py checks
this). Ridge, Blend and dct_basis are imported from the unchanged original module.
"""
from dataclasses import dataclass

import numpy as np
from ridge_corrector import Blend, Config, Ridge, dct_basis

REFERENCE_HORIZON = 24
BASE_WIDTH = 8  # ridge_corrector.GUARD_COMPONENTS


@dataclass(frozen=True)
class Scaled:
    config: Config = Config()
    ridge_rule: bool = True      # ridge = config.ridge * H / 24
    width_rule: bool = True      # widths = (8, config.local) * H / 24
    width_factor: float = 1.     # sensitivity: multiplies both widths of the rule


class ScaledCorrector:
    def __init__(self, channels, horizon=24, lookback=96, scaled=Scaled()):
        cfg = scaled.config
        if lookback % horizon or not (cfg.local or cfg.global_):
            raise ValueError('lookback must be a multiple of horizon; at least one ridge is required')
        self.cfg, self.h, self.blocks = cfg, horizon, lookback // horizon
        ratio = horizon / REFERENCE_HORIZON
        grow = ratio * scaled.width_factor if scaled.width_rule else 1.
        self.width = max(1, round(BASE_WIDTH * grow))            # correction vector and guard
        self.kin = cfg.local                                     # local ridge: input components
        self.kout = max(1, round(cfg.local * grow)) if cfg.local else 0
        self.ridge = cfg.ridge * (ratio if scaled.ridge_rule else 1.)
        if not (self.kout <= self.width <= horizon and max(self.kin, BASE_WIDTH) <= horizon):
            raise ValueError((horizon, self.width, self.kin, self.kout))
        self.basis = dct_basis(horizon, max(self.width, BASE_WIDTH))
        self.u = self.basis[:, :self.width]                      # analysis and synthesis of the correction
        self.uf = self.basis[:, :BASE_WIDTH]                     # features
        k = BASE_WIDTH
        self.glob = Ridge(channels, 2 + k + self.blocks * k, self.width, True, self.ridge, cfg.half_life) \
            if cfg.global_ else None
        self.loc = Ridge(channels, 1 + self.kin + (self.kin * cfg.local_blocks + 1 if cfg.local_blocks else 0),
                         self.kout, False, self.ridge, cfg.half_life) if self.kin else None
        self.scale = np.full(channels, np.nan) if cfg.guard else None
        self.decay = 2. ** (-1. / cfg.guard_half_life)
        self.blend = Blend()

    def _features(self, base, window):
        u, c = self.uf, base.shape[1]
        ones, a = np.ones((1, c)), u.T @ base
        sub = window.reshape(self.blocks, self.h, c)[::-1]  # newest first
        last = np.sqrt(self.h) * window[-1:]
        xg = np.concatenate([ones, a] + [u.T @ s for s in sub] + [last]).T if self.glob else None
        kl, nb = self.kin, self.cfg.local_blocks
        xl = np.concatenate([ones, a[:kl]] + ([(u.T @ s)[:kl] for s in sub[:nb]] + [last] if nb else [])).T \
            if self.loc else None
        return xg, xl

    def correct(self, base, window):
        """Correction (horizon x channels) before blending, and the internals needed by `update`."""
        xg, xl = self._features(base, window)
        c, k = base.shape[1], self.width
        zg = self.glob.predict(xg) if self.glob else np.zeros((c, k))
        z = zg.copy()
        if self.loc:
            z[:, :self.kout] += self.loc.predict(xl)
        if self.scale is not None:
            limit = self.cfg.tau * np.sqrt(k * np.nan_to_num(self.scale, nan=np.inf))
            z *= np.minimum(1., limit / np.maximum(np.linalg.norm(z, axis=1), 1e-300))[:, None]
        return self.u @ z.T, (xg, xl, zg)

    def issue(self, base, window):
        correction, pending = self.correct(base, window)
        return base + self.blend.weight() * correction, correction, pending

    def update(self, base, target, correction, pending):
        xg, xl, zg = pending
        residual = target - base
        self.blend.update(correction, residual)
        z = (self.u.T @ residual).T
        keep = np.ones(len(z), bool)
        if self.scale is not None:
            power = np.mean(z ** 2, axis=1)
            fresh = np.isnan(self.scale)
            keep = fresh | (power <= self.cfg.tau ** 2 * self.scale)
            self.scale = np.where(fresh, power, np.where(keep, self.decay * self.scale + (1 - self.decay) * power,
                                                         self.scale))
        if self.glob:
            self.glob.update(xg, z, keep)
        if self.loc:
            self.loc.update(xl, (z - zg)[:, :self.kout], keep)
        return keep

    @property
    def state_bytes(self):
        """Ridges, DCT basis and component ids, guard scale and blend."""
        state = self.basis.nbytes + 8 * self.width + self.blend.state_bytes
        state += sum(r.state_bytes for r in (self.glob, self.loc) if r)
        return state + (self.scale.nbytes if self.scale is not None else 0)


def replay(base, target, windows, scaled=Scaled(), stop=None):
    """Chronological replay over blocks [n, horizon, channels]: issue each block, then learn from its target.
    With `stop`, the updates end before block `stop` (later blocks are issued from the state of that time).
    Returns the blended and unblended forecasts, the number of skipped channel updates and the state size."""
    n, h, c = base.shape
    corrector = ScaledCorrector(c, h, windows.shape[1], scaled)
    issued, raw, skipped = np.empty_like(base), np.empty_like(base), 0
    for t in range(n):
        issued[t], correction, pending = corrector.issue(base[t], windows[t])
        raw[t] = base[t] + correction
        if stop is None or t < stop:
            skipped += int((~corrector.update(base[t], target[t], correction, pending)).sum())
    return issued, raw, skipped, corrector.state_bytes


def static_replay(val_base, val_target, val_windows, base, windows, scaled=Scaled()):
    """Fit the corrector on the chronological validation blocks, then issue every evaluation block unchanged."""
    n, h, c = base.shape
    corrector = ScaledCorrector(c, h, windows.shape[1], scaled)
    for t in range(len(val_base)):
        _, correction, pending = corrector.issue(val_base[t], val_windows[t])
        corrector.update(val_base[t], val_target[t], correction, pending)
    issued, raw = np.empty_like(base), np.empty_like(base)
    for t in range(n):
        issued[t], correction, _ = corrector.issue(base[t], windows[t])
        raw[t] = base[t] + correction
    return issued, raw, corrector.state_bytes
