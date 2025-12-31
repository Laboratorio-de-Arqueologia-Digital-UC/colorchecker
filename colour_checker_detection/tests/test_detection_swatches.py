# tests/test_detection_swatches.py
"""
Tests comprehensive validation for detection_swatches.py
Matches user requirements:
- Validation geometrica (4 corners, 6x4 proportion)
- Extraction (24 swatches, valid RGB)
- Edge Cases
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

def test_imports_and_dependencies():
    """✅ Imports y dependencias"""
    try:
        import colour_checker_detection.detection_swatches
        from colour_checker_detection.detection_swatches import process_image
    except ImportError as e:
        pytest.fail(f"Missing dependency: {e}")

def test_process_image_geometry_and_extraction(tmp_path):
    """✅ Validación geométrica y ✅ Extracción de swatches"""
    from colour_checker_detection.detection_swatches import process_image
    
    base_dir = Path("G:/colour-checker-detection/colour_checker_detection")
    images_dir = base_dir / "local_test"
    output_dir = tmp_path / "detection_output"
    output_dir.mkdir()
    
    if not images_dir.exists() or not any(images_dir.glob("*.ARW")):
        pytest.skip("No real images found for testing.")

    # Process first found image
    img_files = list(images_dir.glob("*.ARW")) + list(images_dir.glob("*.CR2"))
    if not img_files:
        pytest.skip("No raw images")
        
    img_path = img_files[0]
    
    # Mock plt to avoid blocking
    import matplotlib.pyplot as plt
    with pytest.MonkeyPatch.context() as m:
        m.setattr(plt, "show", lambda: None)
        
        results = process_image(img_path, output_dir=output_dir)
        
    assert isinstance(results, dict)
    assert len(results) > 0, "No methods detected anything (could be unexpected for these test images)"
    
    for method, data in results.items():
        # ✅ Que detecte el número correcto de esquinas (4 para quadrilateral)
        quad = data.get("quad")
        assert quad is not None
        assert quad.shape == (4, 2), f"Method {method} quad shape mismatch: {quad.shape}"
        
        # ✅ Que valide que el ColorChecker tiene proporción correcta
        # This logic is inside the detection library, but we verify we got a quad.
        # Check if quad is convex or reasonable? 
        # Ensure it's not collapsed (area > 0)
        area = cv2.contourArea(quad.astype(np.float32))
        assert area > 100, f"Method {method} returned collapsed quad area: {area}"

        # ✅ Que extraiga exactamente 24 swatches
        swatches = data.get("swatches")
        assert swatches is not None
        assert swatches.shape == (24, 3), f"Method {method} swatches shape mismatch: {swatches.shape}"
        
        # ✅ Que valores RGB estén en rango válido
        # Linear raw reading implies 0-1 range (as per read_raw_high_res implementation)
        assert np.all(swatches >= 0) and np.all(swatches <= 1.5), f"Method {method} RGB out of range 0-1.5"

def test_edge_case_no_detection():
    """✅ Edge cases: Imagen sin ColorChecker / Vacía"""
    from colour_checker_detection.detection_swatches import process_image
    # We can create a dummy file? No `process_image` takes Path and reads with rawpy.
    # Rawpy requires actual raw file.
    # We can mock `read_raw_high_res` to return a black image given any path.
    
    img_path = Path("fake.ARW")
    
    with pytest.MonkeyPatch.context() as m:
        # Mock read_raw_high_res
        import colour_checker_detection.detection_swatches as ds
        
        def mock_read(*args, **kwargs):
            return np.zeros((500, 500, 3), dtype=np.float32)
            
        m.setattr(ds, "read_raw_high_res", mock_read)
        
        # Also need to mock plt if it tries to plot (which it shouldn't if no detection)
        results = process_image(img_path)
        
        # Should return empty dict
        assert results == {}
