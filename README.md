# DRiC: DCT Ridge Corrector

Reference implementation and numerical records for *Local and Global Recursive Ridge Regression for Online Residual Correction of Frozen Multivariate Forecasters under Label Delay* (T. Fujimoto and H. Nishi, 2026).

DRiC (DCT Ridge Corrector) corrects a frozen multi-horizon forecaster online from its own residuals. Each block of `H` steps, the corrector predicts the leading DCT coefficients of the next residual block from the leading DCT coefficients of the frozen forecast and of the latest input block. The prediction uses an exponentially weighted ridge regression whose sufficient statistics are updated only after the block's target is complete (label delay equal to the horizon). The ridge penalty and the number of corrected coefficients are set from the horizon alone: with `r = H / 24`, the penalty is `r`, the local regression outputs `K r` coefficients, and the correction vector and the guard use `8 r` (`dric_scaled.py`). At `H = 24` these are the constants of the original corrector (`ridge_corrector.py`).

- **DRiC-L (local):** one regression per channel; DRiC-L4 and DRiC-L8 read $K=4$ and $K=8$ coefficients (`local4`, `local8` in the records).
- **DRiC-G (global, `global` in the records):** one regression shared by all channels; its regression statistics do not depend on the channel count (only the guard keeps one scale, 8 bytes, per channel).
- **Guard:** keeps a running scale of each channel's residual. Blocks far outside that scale are left out of the update, and the issued correction is clipped by the same scale.

## Install and test

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest test_ridge_corrector test_dric_scaled
.venv/bin/python reproduce.py
```

`reproduce.py` recomputes the paper's summary tables for the horizons 24, 48, and 96 (mean, panel-weighted mean and median MSE reductions, mean MAE reduction, retained state, paired wins against ELF) from the condition-level records and checks them against `records/summary.csv`.

## Correct your own forecasts

Save chronological arrays in an NPZ file:

- `base` and `target`, each `[blocks, horizon, channels]`;
- `windows`, `[blocks, lookback, channels]`, the input window of each forecast. `lookback` must be a multiple of `horizon`.

Then run:

```bash
.venv/bin/python replay.py forecasts.npz --corrector local --components 8
.venv/bin/python replay.py forecasts.npz --corrector global
```

The defaults are the paper's settings: reference penalty 1 (the penalty is `H / 24`), a forgetting half-life of 128 blocks, guard threshold `tau = 10`, and a blending weight shared by all channels and fitted on the last 32 blocks. `--fixed-settings` keeps the constants of `H = 24` at every horizon. In Python:

```python
from dric_scaled import Scaled, ScaledCorrector
from ridge_corrector import Config
corrector = ScaledCorrector(channels, horizon=48, lookback=96, scaled=Scaled(Config(local=8)))
issued, correction, pending = corrector.issue(base_block, window)   # before the target is known
corrector.update(base_block, target_block, correction, pending)     # after the block is complete
```

## Records

Every file has a `horizon` column (24, 48, or 96).

| File | Content |
|---|---|
| `records/conditions_main.csv` | MSE, MAE, their reductions relative to the frozen forecast, and retained state for every condition and every method of the main comparison (ELF with the shared blend, `elf`, and with its own weighting, `elf_own`). Covers 7 datasets × 3 backbones × 3 seeds × legacy/refit, and the standard Weather file, at each horizon |
| `records/conditions_correctors_all_settings.csv` | The correctors' component, sensitivity, and frozen-corrector variants, with the number of skipped channel updates |
| `records/conditions_comparator_settings.csv` | ORCA with its published replay buffer (`orca_pub`), the one-at-a-time settings of ELF with the shared blend (`elf_*`) and with ELF's own weighting (`elf_*_own`), and ELF trained on every window (`elf_stride1`, `elf_stride1_own`) |
| `records/panel_means.csv` | Per horizon, dataset, and backbone means plotted in the paper's accuracy–state figures |
| `records/summary.csv` | Values of the paper's comparison tables |
| `records/frozen_forecasts.csv` | Selected number of training updates and the MSE and MAE of every frozen forecast |

In the `table` column, `filled` marks the Weather file with its −9999 missing-value codes forward-filled (Weather* in the paper). `standard` marks every other dataset, including the original Weather file. The evaluation protocol is in [docs/PROTOCOL.md](docs/PROTOCOL.md).

Names in `conditions_correctors_all_settings.csv`: `local8`, `local4`, and `global` are the paper's correctors. Suffixes mark one change: `_fixed` (the constants of `H = 24` at every horizon), `_penalty_only` and `_widths_only` (one of the two horizon settings), `_lam…` (reference penalty), `_widths…` (both widths multiplied), `_tau…`, `_hl…` (half-life in blocks), `_noguard`, `_noinput`, `_plus_global`, `_fixed_allblocks` (all input sub-blocks, with the fixed constants), `_half_frozen` and `_half_online` (second half of the evaluation, with updates stopped at mid-stream or continued), and `_static` (fitted on the validation range, then frozen; legacy bases only).

## Related repository

EPOC, the compressed-residual corrector compared in the paper, is at <https://github.com/keiotakmin/endpoint-preserving-residual-correction>.

Code is provided under the [MIT License](LICENSE). Dataset licenses remain with their providers.
