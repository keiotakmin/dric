"""Correct saved forecasts of a frozen forecaster block by block.

The input NPZ holds chronological arrays `base` and `target` of shape [blocks, horizon, channels] and `windows` of
shape [blocks, lookback, channels] (the input window of each forecast); lookback must be a multiple of horizon.
Each block is issued before its target is used, as in deployment with label delay equal to the horizon.

The ridge penalty and the number of corrected DCT components follow the horizon (penalty H / 24, widths scaled by
H / 24; see docs/PROTOCOL.md). --fixed-settings keeps the constants of H = 24 at every horizon.

  python replay.py forecasts.npz --corrector local --components 8 --output corrected.npz
"""
import argparse

import numpy as np

from dric_scaled import Scaled, replay
from ridge_corrector import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('forecasts')
    parser.add_argument('--corrector', choices=('local', 'global'), default='local')
    parser.add_argument('--components', type=int, default=8, help='input DCT components of the local corrector (4 or 8)')
    parser.add_argument('--tau', type=float, default=10.)
    parser.add_argument('--ridge', type=float, default=1., help='reference penalty (the penalty at H = 24)')
    parser.add_argument('--half-life', type=float, default=128.)
    parser.add_argument('--no-guard', action='store_true')
    parser.add_argument('--fixed-settings', action='store_true',
                        help='do not scale the penalty and the widths with the horizon')
    parser.add_argument('--output', default='corrected.npz')
    args = parser.parse_args()
    with np.load(args.forecasts) as z:
        base, target, windows = (np.asarray(z[k], float) for k in ('base', 'target', 'windows'))
    config = Config(local=args.components if args.corrector == 'local' else 0, global_=args.corrector == 'global',
                    guard=not args.no_guard, tau=args.tau, ridge=args.ridge, half_life=args.half_life)
    scaled = Scaled(config, ridge_rule=not args.fixed_settings, width_rule=not args.fixed_settings)
    issued, raw, skipped, state = replay(base, target, windows, scaled)
    static, mse = np.mean((target - base) ** 2), np.mean((target - issued) ** 2)
    np.savez(args.output, issued=issued, unblended=raw)
    print(f'MSE frozen {static:.6g}  corrected {mse:.6g}  reduction {100 * (1 - mse / static):.2f}%  '
          f'skipped channel updates {skipped}  retained state {state} B')


if __name__ == '__main__':
    main()
