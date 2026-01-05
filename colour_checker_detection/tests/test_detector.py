from __future__ import annotations

import unittest
from unittest.mock import patch

from colour_checker_detection.detector import detect_chart


class TestDetector(unittest.TestCase):
    @patch("colour_checker_detection.detector.detect_colour_checkers_templated")
    def test_detect_chart_passthrough(self, mock_detect):
        """Test that detect_chart correctly calls the underlying implementation."""
        dummy_image = "image_data"
        dummy_kwargs = {"param": 1}

        detect_chart(dummy_image, **dummy_kwargs)

        mock_detect.assert_called_once_with(dummy_image, **dummy_kwargs)


if __name__ == "__main__":
    unittest.main()
