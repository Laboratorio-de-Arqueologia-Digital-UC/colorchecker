from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import rawpy
from colour_checker_detection.io import load_raw_linear, load_raw_visual


class TestIO(unittest.TestCase):
    @patch("colour_checker_detection.io.rawpy.imread")
    def test_load_raw_linear(self, mock_imread):
        """Test load_raw_linear handling of 16-bit normalization and WB extraction."""
        # Setup mock
        mock_raw = MagicMock()
        mock_context = MagicMock()
        mock_imread.return_value = mock_context
        mock_context.__enter__.return_value = mock_raw

        # Mock raw.postprocess return (dummy image 10x10x3)
        # return random int 16bit
        dummy_img = np.random.randint(0, 65535, (10, 10, 3), dtype=np.uint16)
        mock_raw.postprocess.return_value = dummy_img

        # Mock WB
        mock_raw.camera_whitebalance = [2.0, 1.0, 1.5, 1.0]

        # Call
        path = "dummy.CR2"
        # We need to mock Path.exists to return True
        with patch("pathlib.Path.exists", return_value=True):
            img, wb = load_raw_linear(path)

        # Assertions
        self.assertEqual(wb, [2.0, 1.0, 1.5, 1.0])
        self.assertEqual(img.dtype, np.float64) # Should be float after div
        self.assertTrue(np.all(img >= 0) and np.all(img <= 1.0))

        # Check rawpy call args
        mock_raw.postprocess.assert_called_with(
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.raw,
            output_bps=16
        )

    @patch("colour_checker_detection.io.rawpy.imread")
    def test_load_raw_visual(self, mock_imread):
        """Test load_raw_visual handling of 8-bit normalization."""
        # Setup mock
        mock_raw = MagicMock()
        mock_context = MagicMock()
        mock_imread.return_value = mock_context
        mock_context.__enter__.return_value = mock_raw

        # Mock raw.postprocess return (dummy image 8bit)
        dummy_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        mock_raw.postprocess.return_value = dummy_img

        # Call
        path = "dummy.CR2"
        with patch("pathlib.Path.exists", return_value=True):
            img = load_raw_visual(path, brightness=2.0)

        # Assertions
        self.assertEqual(img.dtype, np.float64)
        self.assertTrue(np.all(img >= 0) and np.all(img <= 1.0))

        mock_raw.postprocess.assert_called_with(
            use_camera_wb=True,
            bright=2.0,
            no_auto_bright=True
        )

if __name__ == "__main__":
    unittest.main()
