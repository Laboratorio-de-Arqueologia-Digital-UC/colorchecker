# tests/test_test_script.py
"""Tests básicos para test.py"""

from pathlib import Path

import pytest


def test_test_script_exists():
    """Verifica que el módulo test.py existe y se puede importar."""
    try:
        from colour_checker_detection import test
    except ImportError:
        pytest.fail("No se pudo importar colour_checker_detection.test")
    assert test is not None


def test_functions_exist():
    """Verifica que las funciones principales existen."""
    from colour_checker_detection.test import adapter_yolo_inferencer, run_benchmark

    assert callable(run_benchmark)
    assert callable(adapter_yolo_inferencer)


def test_full_benchmark(tmp_path):
    """Test del benchmark completo usando imágenes locales."""
    from colour_checker_detection.test import run_benchmark

    # Path hardcoded conocido del proyecto (según instrucción del usuario)
    base_dir = Path("G:/colour-checker-detection/colour_checker_detection")
    images_dir = base_dir / "local_test"
    model_path = base_dir / "models/colour-checker-detection-l-seg.pt"

    if not images_dir.exists() or not any(images_dir.iterdir()):
        pytest.skip(f"Directorio de imágenes no encontrado o vacío: {images_dir}")

    if not model_path.exists():
        pytest.skip(f"Modelo no encontrado: {model_path}")

    # Ejecutar benchmark
    # No genera archivos de salida, solo logs y visualizaciones (que pueden bloquear en CI pero ok local)
    # Mockeamos plt.show para no bloquear
    import matplotlib.pyplot as plt

    with pytest.MonkeyPatch.context() as m:
        m.setattr(plt, "show", lambda: None)
        run_benchmark(model_path, images_dir)
