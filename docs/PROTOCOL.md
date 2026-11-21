# Evaluation protocol

## Data
- ETTh1, ETTh2, ETTm1, ETTm2 (7 channels each), and the Weather (21), Electricity (321), and Traffic (862) CSV files distributed with Autoformer / Time-Series-Library. All numerical columns are forecast.
- **Split:** with series length `T`, `n = floor(T/2)` and `a = floor(0.8 n)`. Legacy bases are trained on `[0, a)`, `[a, n)` is used for selection, and refit bases are retrained on `[0, n)`.
- **Normalization:** each channel is standardized with the mean and standard deviation of `[0, n)`. Errors are reported on this scale.
- **Forecasts:** lookback 96; horizons `H` = 24, 48, and 96; origins every `H` samples from `n`. Only complete target blocks are scored.
- **Weather:** the standard Weather file contains 81 cells with the code −9999. The main view uses a copy in which these cells are forward-filled, with every base retrained on it. The standard file is reported separately.
- **Exchange:** omitted, because the compact PatchTST base without instance normalization fails on its evaluation period.

## Frozen forecasters
- **Backbones:** DLinear, a compact PatchTST, and iTransformer of the same size.
- **Training:** seeds 0–2, Adam at 1e-3, batch size 32. The update count is chosen from {200, 500, 1000, 2000, 4000, 8000, 20000} by validation MSE; when the best count is the last one, training continues to 40,000 and then to 80,000 updates. `records/frozen_forecasts.csv` lists the selected counts.
- **Conditions:** at each horizon, 7 datasets × 3 backbones × 3 seeds × legacy/refit = 126 conditions, plus 18 on the standard Weather file.

## Methods
- **Order of information:** every method issues a block before its target is available and updates only after the target block is complete. Every method starts the evaluation from an empty state.
- **Blending:** every correction is scaled by one weight in [0, 1], shared by all channels and fitted by least squares on the last 32 completed blocks.
- **Correctors:** half-life 128 blocks and guard threshold 10 at every horizon. Two settings follow the horizon, with `r = H / 24`: the ridge penalty is `r` (the same penalty on coefficients normalized by `sqrt(H)`), and the widths are `8 r` DCT components for the correction vector and the guard and `K r` for the output of the local regression (the same frequency band per sample at every horizon). The inputs do not change: the local corrector reads `K = 4` or `8` components of the forecast and of the latest input sub-block, the global corrector 8 components of the forecast and of all input sub-blocks. At `H = 24` the settings are penalty 1 and widths 8 and `K`.
- **Comparators:** persistence and a daily seasonal naive forecast; TAFAS, PETSA, and COSA with their official modules and FAC reconstructed from its equations, one Adam step per complete block, with the learning rate (and, for TAFAS and PETSA, the gate initialization) selected per horizon, dataset, backbone, and seed on the legacy validation range; ORCA with its official network and a replay buffer of 32 blocks (`orca`; its published buffer of 3,000 windows is `orca_pub`); ELF's linear forecaster with 86 input features and the leading `floor(0.9 H / 2)` output frequencies, updated after every complete block from one window per block, combined with the frozen forecast by the shared blending weight (`elf`) or by ELF's own weighting module (`elf_own`); `conditions_comparator_settings.csv` adds ELF's one-at-a-time settings with both combinations and ELF trained on every window (`elf_stride1`, `elf_stride1_own`); EPOC with its default configuration.
- **Controls with the corrector frozen:** `_half_frozen` stops the updates after the first half of the evaluation blocks and is scored, like `_half_online`, on the second half; `_static` is fitted on the validation range of the legacy base in time order and then issues every evaluation block unchanged.
- **Development:** the correctors' structure, features, and constants were developed on the 126 conditions at `H = 24`. The two horizon settings were set after the correctors with these constants had been evaluated at `H = 48` and `96` on the same conditions (`*_fixed` in the records). The results are empirical results on the conditions used for development, not an independent validation.

## Retained state
Retained state counts the arrays kept between forecasts, at the dtype they are implemented with:
- **Not counted:** the frozen forecaster, the shared input window, and issued forecasts awaiting their targets.
- **Counted for the correctors:** the ridge statistics (upper-triangular Gram matrix and right-hand side), the DCT basis (`H` × `8 r` values) and component indices, one guard scale per channel, and the blending weight's history (528 bytes).
