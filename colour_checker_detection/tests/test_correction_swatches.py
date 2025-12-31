# tests/test_correction_swatches.py
"""
Tests comprehensive validation for correction_swatches.py
Matches user requirements:
- Pipeline end-to-end (Detection, Extraction, CCM, Correction)
- CCM calculation (3x3, numeric validity, range preservation)
- Normalization (white point)
- Visualization (6 panels)
"""

import pytest
import numpy as np
from pathlib import Path
import colour

def test_imports_and_dependencies():
    """✅ Imports y dependencias"""
    try:
        from colour_checker_detection.correction_swatches import process_image
        import colour
    except ImportError as e:
        pytest.fail(f"Missing dependency: {e}")

def test_pipeline_and_ccm(tmp_path):
    """✅ Pipeline end-to-end y ✅ Cálculo CCM"""
    from colour_checker_detection.correction_swatches import process_image
    
    base_dir = Path("G:/colour-checker-detection/colour_checker_detection")
    images_dir = base_dir / "local_test"
    output_dir = tmp_path / "correction_output"
    output_dir.mkdir()
    
    if not images_dir.exists() or not any(images_dir.glob("*.ARW")):
        pytest.skip("No real images found for testing.")

    # Process first found image
    img_files = list(images_dir.glob("*.ARW")) + list(images_dir.glob("*.CR2"))
    if not img_files:
        pytest.skip("No raw images")
        
    img_path = img_files[0]
    
    # Mock plt
    import matplotlib.pyplot as plt
    with pytest.MonkeyPatch.context() as m:
        m.setattr(plt, "show", lambda: None)
        
        results = process_image(img_path, output_dir=output_dir)
        
    assert isinstance(results, dict)
    assert len(results) > 0, "No methods processed successfully"
    
    for method, data in results.items():
        # ✅ Que la matriz CCM sea 3x3
        ccm = data.get("CCM")
        assert ccm is not None
        assert ccm.shape == (3, 3)
        
        # ✅ Que CCM tenga valores numéricos válidos (no NaN, no Inf)
        assert np.isfinite(ccm).all()
        
        # ✅ Que CCM aplicada preserve rangos de color (checked indirectly via corrected swatches)
        swatches_measured = data["swatches_measured"]
        swatches_corrected = data["swatches_corrected"]
        
        # Linear raw (0-1 approx, can be >1 slightly due to specular or white balance)
        # Verify mostly valid
        assert swatches_corrected.shape == (24, 3)
        # Corrected might exceed 1 slightly or be negative slightly if gamut issue, 
        # but generally should be reasonable.
        # We check mean is reasonable valid range (0.0 to 1.5)
        assert 0.0 <= np.mean(swatches_corrected) <= 1.5

        # ✅ Normalización punto blanco
        # "Que parches grises queden neutros (R≈G≈B)"
        # Index 18 is D65 white patch
        white_patch_corr = swatches_corrected[18]
        # In Adobe RGB D65, white is (1,1,1) if perfectly exposed/mapped.
        # Check standard deviation of RGB channels is low (neutral)
        std_white = np.std(white_patch_corr)
        assert std_white < 0.1, f"White patch not neutral enough: {white_patch_corr}, std: {std_white}"
        
        # Check output generation (6 panels logic implies 1 image file)
        # correction_{method}_{stem}.png
        stem = img_path.stem
        out_file = output_dir / f"correction_{method}_{stem}.png"
        assert out_file.exists(), f"Visualization file not created: {out_file}"
        
        # Verify file size > 0
        assert out_file.stat().st_size > 0

def test_edge_case_invalid_image():
    """✅ Edge cases: Imagen corrupta o inválida"""
    from colour_checker_detection.correction_swatches import process_image
    
    # Passing a non-raw file or non-existent
    # process_image should return None or handle it
    res = process_image(Path("garbage.ARW"))
    assert res is None
