"""Scientific invariants for the E27 selection-only intervention."""
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nar import e27_direction_selection as e27
from nar import activation_experiments as act


class E27Tests(unittest.TestCase):
    def test_selection_global_percent_and_bos_ties(self):
        scores = np.zeros((4, 100), dtype=np.float32)
        scores[:, 0] = 100
        scores[0, 1:8] = np.arange(90, 83, -1)
        scores[1:, 1] = 4
        selected = e27.selection_indices(scores)
        np.testing.assert_array_equal(selected['B_top1'], [0, 100, 200, 300])
        np.testing.assert_array_equal(selected['B_top1_no_bos'], [1, 101, 201, 301])
        np.testing.assert_array_equal(selected['C_top1pct'], [0, 100, 200, 300])
        np.testing.assert_array_equal(selected['C_top1pct_no_bos'], [1, 2, 3, 4])

    def test_streaming_pool_matches_all_rows(self):
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(e27, NCAL=4, LENGTH=100, STRIDE=5):
            model = SimpleNamespace(config=SimpleNamespace(hidden_size=256, intermediate_size=256),
                                    model=SimpleNamespace(layers=[None]))
            collector = e27.Capture(model, Path(tmp))
            gen = torch.Generator().manual_seed(11)
            values = torch.randn((4, 100, 256), generator=gen).bfloat16()
            values[:, 0] *= 10
            for begin in (0, 2):
                collector.start = begin
                for site in act.SITES:
                    collector.consume(site, 0, values[begin:begin + 2])
            collector.close(); collector.finish('synthetic')
            for site in act.SITES:
                payload = torch.load(Path(tmp) / f'{site}_layer_00.selection.pt', weights_only=True)
                expected = e27.selection_indices(values.abs().amax(-1).float().numpy())
                for method, ids in expected.items():
                    np.testing.assert_array_equal(payload['variants'][method]['indices'], ids)
                    torch.testing.assert_close(payload['variants'][method]['values'], values.reshape(-1, 256)[ids], rtol=0, atol=0)

    def test_degenerate_bos_completion_is_deterministic(self):
        gen = torch.Generator().manual_seed(9)
        direction = torch.randn((1, 256), generator=gen)
        x = direction.repeat(12, 1)
        v, info = e27.estimate_basis(x, 2, 71, exact=True)
        w, _ = e27.estimate_basis(x, 2, 71, exact=True)
        self.assertEqual(info['selected_numerical_rank'], 1)
        self.assertEqual(info['null_completion_directions'], 1)
        torch.testing.assert_close(v, w, rtol=0, atol=0)
        torch.testing.assert_close(v.T @ v, torch.eye(2), rtol=1e-5, atol=1e-6)
        self.assertLess(float((x - (x @ v) @ v.T).norm() / x.norm()), 1e-5)

    def test_householder_capture_and_transpose_fold(self):
        gen = torch.Generator().manual_seed(13)
        v = torch.linalg.qr(torch.randn((256, 2), generator=gen)).Q
        x = torch.randn((24, 256), generator=gen)
        factor = act.factor_from_vectors(v, x, 128)
        recovered = e27.recover_basis(factor)
        torch.testing.assert_close(recovered @ recovered.T, v @ v.T, rtol=1e-4, atol=1e-6)
        signs = e27.seed_signs(256, 0, 'qkv', 0, device='cpu')
        rotated = factor.apply(x, signs)
        actual = (rotated.reshape(-1, 2, 128).mean(-1).square().sum() * 128) / rotated.square().sum()
        expected = (x @ v).square().sum() / x.square().sum()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
        weights = torch.randn((9, 256), generator=gen)
        torch.testing.assert_close(rotated @ factor.apply(weights, signs).T, x @ weights.T, rtol=1e-4, atol=2e-5)

    def test_eight_angles(self):
        a = torch.eye(16)[:, :8]
        b = a.clone(); b[:, -1] = torch.eye(16)[:, 8]
        angles = e27.angles_degrees(a, b)
        self.assertEqual(len(angles), 8)
        np.testing.assert_allclose(angles[:-1], 0, atol=1e-5)
        self.assertAlmostEqual(angles[-1], 90, places=5)

    def test_paired_seed_ci_and_D_trigger(self):
        rows = []
        for site in e27.CONDITIONS:
            for i, delta in enumerate([-.1, .0, .2]):
                for method, ppl in [('A_full', 8 + i), ('B_top1', 8 + i + delta)]:
                    rows.extend({'model': 'synthetic', 'site': site, 'method': method,
                                 'seed': e27.SEED + i, 'sequence': j, 'nll': math.log(ppl)} for j in range(64))
        summary = e27.summarize(rows)
        row = next(r for r in summary if r['site'] == 'both' and r['method'] == 'B_top1')
        mean = np.mean([-.1, 0, .2]); half = act.TCRIT_DF2_90 * np.std([-.1, 0, .2], ddof=1) / math.sqrt(3)
        self.assertAlmostEqual(row['paired_ppl_delta_vs_A'], mean)
        self.assertAlmostEqual(row['paired_90ci_low_vs_A'], mean - half)
        self.assertAlmostEqual(row['paired_90ci_high_vs_A'], mean + half)
        self.assertFalse(e27.trigger_d(summary))
        row['paired_ppl_delta_vs_A'] = -.001
        self.assertTrue(e27.trigger_d(summary))
        with self.assertRaises(RuntimeError):
            e27.summarize(rows + rows[:1])

    def test_nonfinite_selection_rejected(self):
        with self.assertRaises(RuntimeError):
            e27.rank_indices([float('nan')], [0], 1)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
