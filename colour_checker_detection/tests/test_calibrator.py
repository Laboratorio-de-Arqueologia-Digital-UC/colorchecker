from __future__ import annotations

import unittest

import numpy as np

from colour_checker_detection.calibrator import calculate_ccm


class TestCalibrator(unittest.TestCase):
    def test_calculate_ccm_shape(self):
        """Test that calculate_ccm returns a 3x3 matrix."""
        # Create dummy data: 24 patches, 3 channels
        measured = np.random.rand(24, 3)
        reference = np.random.rand(24, 3)

        M = calculate_ccm(measured, reference)

        self.assertEqual(M.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(M)))

    def test_calculate_ccm_identity(self):
        """Test that if measured ~= reference, M should be close to identity."""
        reference = np.random.rand(24, 3)
        # Measured is reference (perfect match)
        measured = reference.copy()

        M = calculate_ccm(measured, reference)

        # Should be Identity matrix
        np.testing.assert_array_almost_equal(M, np.eye(3), decimal=5)


if __name__ == "__main__":
    unittest.main()
