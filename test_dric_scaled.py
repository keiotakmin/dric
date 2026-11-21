import unittest

import numpy as np

import ridge_corrector as rc
from dric_scaled import Scaled, ScaledCorrector, replay
from ridge_corrector import Config

L = 96


def stream(h, n=60, c=4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n * h + L)
    series = np.sin(2 * np.pi * t / 24)[:, None] * rng.uniform(.5, 2, c) + .1 * rng.standard_normal((len(t), c))
    windows = np.stack([series[i * h:i * h + L] for i in range(n)])
    target = np.stack([series[i * h + L:i * h + L + h] for i in range(n)])
    base = .8 * target + .05 * rng.standard_normal(target.shape)  # a biased frozen forecast
    return base, target, windows


CONFIGS = [Config(local=4), Config(local=8), Config(local=0, global_=True), Config(local=8, global_=True),
           Config(local=8, guard=False)]


class TestScaled(unittest.TestCase):
    def test_rules_off_is_the_original_corrector(self):
        for h in (24, 48, 96):
            base, target, windows = stream(h)
            for config in CONFIGS:
                ours = replay(base, target, windows, Scaled(config, ridge_rule=False, width_rule=False))
                ref = rc.replay(base, target, windows, config)
                for a, b in zip(ours[:2], ref[:2]):
                    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
                self.assertEqual(ours[2:], ref[2:])

    def test_reference_horizon(self):
        """At H = 24 the horizon settings are the original constants."""
        base, target, windows = stream(24)
        for config in CONFIGS:
            ours = replay(base, target, windows, Scaled(config))
            ref = rc.replay(base, target, windows, config)
            np.testing.assert_allclose(ours[0], ref[0], rtol=0, atol=1e-12)
            self.assertEqual(ours[3], ref[3])

    def test_settings_follow_the_horizon(self):
        for h, r in ((24, 1), (48, 2), (96, 4)):
            for k in (4, 8):
                corrector = ScaledCorrector(3, h, L, Scaled(Config(local=k)))
                self.assertEqual((corrector.ridge, corrector.width, corrector.kin, corrector.kout), (r, 8 * r, k, k * r))
                self.assertEqual(corrector.loc.rhs.shape, (3, 2 * k + 2, k * r))
            corrector = ScaledCorrector(3, h, L, Scaled(Config(local=0, global_=True)))
            self.assertEqual(corrector.glob.rhs.shape, (1, 10 + 8 * (L // h), 8 * r))

    def test_causal(self):
        """Changing the target of block k leaves every forecast issued up to block k unchanged."""
        base, target, windows = stream(48)
        k = 30
        changed = target.copy()
        changed[k] += 5.
        for config in CONFIGS:
            a = replay(base, target, windows, Scaled(config))[0]
            b = replay(base, changed, windows, Scaled(config))[0]
            np.testing.assert_array_equal(a[:k + 1], b[:k + 1])
            self.assertFalse(np.array_equal(a[k + 1:], b[k + 1:]))

    def test_updates_can_stop(self):
        base, target, windows = stream(48)
        online = replay(base, target, windows, Scaled(Config(local=8)))[0]
        frozen = replay(base, target, windows, Scaled(Config(local=8)), stop=30)[0]
        np.testing.assert_array_equal(online[:31], frozen[:31])
        self.assertFalse(np.array_equal(online[31:], frozen[31:]))


if __name__ == '__main__':
    unittest.main()
