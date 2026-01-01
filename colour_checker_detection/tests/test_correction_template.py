# tests/test_correction/test_correction_template.py
"""
Tests comprehensive validation for correction_template.py
Matches user requirements:
- Imports & Dependencies
- Detection (Basic & Edge cases)
- Output Generation (JSON structure, PNG)
- Error Handling
"""

import json
from pathlib import Path

import numpy as np
import pytest


def test_imports_and_dependencies():
    """✅ Imports y dependencias"""
    try:
        import colour
        import cv2
        import rawpy

        import colour_checker_detection.correction_template
        from colour_checker_detection.detection import detect_colour_checkers_templated
    except ImportError as e:
        pytest.fail(f"Missing dependency: {e}")


def test_detection_logic_basic(tmp_path):
    """✅ Detección de ColorChecker y ✅ Generación de salidas"""
    from colour_checker_detection.correction_template import main

    # Setup paths
    base_dir = Path("G:/colour-checker-detection/colour_checker_detection")
    images_dir = base_dir / "local_test"
    output_dir = tmp_path / "output_basic"
    output_dir.mkdir()

    if not images_dir.exists() or not any(images_dir.glob("*.ARW")):
        pytest.skip("No real images found for testing detection logic.")

    # Mock plt.show to avoid UI, allow savefig
    import matplotlib.pyplot as plt

    with pytest.MonkeyPatch.context() as m:
        m.setattr(plt, "show", lambda: None)

        main(images_dir=images_dir, output_dir=output_dir)

    # Validar salidas
    assert output_dir.exists()
    files = list(output_dir.iterdir())
    assert len(files) > 0, "No files generated in output directory"

    # Check for JSON and PNG
    json_files = list(output_dir.glob("*.json"))
    png_files = list(output_dir.glob("*.png"))

    assert len(json_files) > 0, "No JSON files generated"
    assert len(png_files) > 0, "No PNG files generated"

    # ✅ Output tiene estructura esperada (JSON)
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        assert "image" in data
        assert "method" in data
        assert "swatches" in data
        assert isinstance(data["swatches"], list)
        assert len(data["swatches"]) == 24

        # ✅ Valores numéricos en rangos correctos
        first_swatch = data["swatches"][0]
        assert "index" in first_swatch
        assert "coordinates_px" in first_swatch
        assert "color_detected_linear" in first_swatch
        assert "delta_e_2000" in first_swatch

        rgb = first_swatch["color_detected_linear"]
        assert len(rgb) == 3
        # Linear RGB should be 0-1 (usually) or raw values.
        # The script divides by 65535 or 255. `read_raw_high_res` divides by 65535 or 255.
        # So range should be 0-1.
        assert all(0.0 <= x <= 1.5 for x in rgb), (
            f"RGB values out of expected range (0-1.5): {rgb}"
        )


def test_edge_cases_empty_image():
    """✅ Manejo de errores: Imagen vacía / invalida"""
    from colour_checker_detection.detection import detect_colour_checkers_templated

    empty_img = np.zeros((100, 100, 3), dtype=np.float32)
    # Should return empty list, not crash
    res = detect_colour_checkers_templated(empty_img, additional_data=True)
    assert isinstance(res, (list, tuple))
    assert len(res) == 0


def test_edge_cases_none_input():
    """✅ Validar inputs incorrectos (None)"""
    from colour_checker_detection.detection import detect_colour_checkers_templated

    # Expect error for None
    with pytest.raises((AttributeError, TypeError, Exception)):
        detect_colour_checkers_templated(None)  # pyright: ignore
