# tests/test_correction_swatches_benchmark.py
"""
Tests comprehensive validation for correction_swatches_benchmark.py
Matches user requirements:
- Comparison of methods (Templated vs Segmentation)
- Statistics (Avg Delta E, Paired T-Test p-value, Drift)
- Output tables
"""

from pathlib import Path

import pytest


def test_imports_and_dependencies():
    """✅ Imports y dependencias"""
    try:
        from colour_checker_detection.correction_swatches_benchmark import (
            main,
            run_benchmark_analysis,
        )
    except ImportError as e:
        pytest.fail(f"Missing dependency: {e}")


def test_analysis_logic(tmp_path):
    """✅ Comparación de métodos y ✅ Estadísticas"""
    from colour_checker_detection.correction_swatches_benchmark import (
        run_benchmark_analysis,
    )

    base_dir = Path("G:/colour-checker-detection/colour_checker_detection")
    images_dir = base_dir / "local_test"

    if not images_dir.exists() or not any(images_dir.glob("*.ARW")):
        pytest.skip("No real images found for testing analysis logic.")

    results = run_benchmark_analysis(images_dir)
    assert results is not None

    # ✅ Que genere métricas: Delta E, tiempo de ejecución
    data_rgb = results["data_rgb"]
    data_time = results["data_time"]
    data_comp = results["data_comp"]
    stats = results["stats"]

    assert len(data_rgb) > 0
    assert len(data_time) > 0
    # Comparison might be empty if only one method detects things, but we expect both to work on standard images.
    assert len(data_comp) > 0

    # Verify structure of one row
    row_rgb = data_rgb[0]
    assert "dE 2000" in row_rgb
    assert "RGB" in row_rgb
    assert isinstance(row_rgb["dE 2000"], float)

    # ✅ Que genere t-test con p-value válido
    # If we have comparisons, we should have stats
    if len(data_comp) > 0:
        assert "t_stat" in stats
        assert "p_val" in stats
        assert 0 <= stats["p_val"] <= 1.0  # p-value range

    # ✅ Reporte drift geométrico en píxeles
    row_comp = data_comp[0]
    assert "Drift (px)" in row_comp
    assert row_comp["Drift (px)"] >= 0


def test_main_execution(capsys):
    """✅ Output de tablas (stdout)"""
    from colour_checker_detection.correction_swatches_benchmark import main

    base_dir = Path("G:/colour-checker-detection/colour_checker_detection")
    images_dir = base_dir / "local_test"

    if not images_dir.exists() or not any(images_dir.glob("*.ARW")):
        pytest.skip("No real images found for testing main execution.")

    main(images_dir=images_dir)

    captured = capsys.readouterr()
    # ✅ Validar columnas esperadas
    assert "|Image" in captured.out
    assert "|Method" in captured.out
    assert "|dE 2000" in captured.out
    assert "|Time (s)" in captured.out

    # Verify Significance Table
    assert "Significancia Estadística" in captured.out
    assert "|T-Statistic" in captured.out
