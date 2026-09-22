"""Small archived-data regression checks; no point-cloud download required."""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'

import numpy as np
import pandas as pd
from src.analysis import run_lofo
from src.features import parse_filename
from src.io_utils import canonical_order, read_features, validate_features, output_dir
from src.settings import METHODS


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features = read_features(ROOT / 'data/reference/midpoint_60m_fixture.csv')

    def test_primary_lofo_matches_archived_predictions(self):
        name = 'MidpointCompact_LR'
        predictions, _, summary = run_lofo(self.features, 'five_class', {name: METHODS[name]}, 1)
        expected = pd.read_csv(ROOT / 'data/reference/primary_lofo_fixture.csv', dtype={'HeldField':str})
        keys = ['HeldField', 'TreeKey']
        got = predictions.sort_values(keys).reset_index(drop=True)
        expected = expected.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(got[expected.columns], expected, check_dtype=False)
        reference = pd.read_csv(ROOT / 'data/reference/04_table4A_main_model_midpoint.csv')
        row = reference[reference['Method'] == name].iloc[0]
        np.testing.assert_allclose(summary.iloc[0]['Accuracy'], row['Accuracy'], atol=1e-12)

    def test_parallel_completion_order_does_not_change_training_order(self):
        shuffled = self.features.sample(frac=1, random_state=19)
        restored = canonical_order(shuffled)
        self.assertEqual(restored['TreeKey'].tolist(), self.features['TreeKey'].tolist())

    def test_duplicate_observations_fail(self):
        duplicate = pd.concat([self.features, self.features.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_features(duplicate)

    def test_cross_altitude_label_conflict_fails(self):
        extra = self.features.iloc[:1].copy()
        extra['Altitude'] = 20
        extra['Class'] = (extra['Class'] + 1) % 5
        with self.assertRaisesRegex(ValueError, 'conflicting labels'):
            validate_features(pd.concat([self.features, extra], ignore_index=True))

    def test_original_project_cannot_be_an_output(self):
        with self.assertRaisesRegex(ValueError, 'subdirectory'):
            output_dir(ROOT.parent)

    def test_parser_preserves_leading_zeroes_and_type(self):
        meta = parse_filename(Path('3_0012_2_valencia_late_060_1.las'))
        self.assertEqual(meta, (3, '0012', '2', 'valencia_late', 60, '1'))


if __name__ == '__main__':
    unittest.main()
