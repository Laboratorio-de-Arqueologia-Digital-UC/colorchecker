
from __future__ import annotations

from typing import Any

from colour_checker_detection.detection import detect_colour_checkers_templated


def detect_chart(image: Any, **kwargs: Any) -> Any:
    """
    Wrapper around detection algorithms to find the ColorChecker.
    Currently defaults to templated matching.
    """
    return detect_colour_checkers_templated(image, **kwargs)
