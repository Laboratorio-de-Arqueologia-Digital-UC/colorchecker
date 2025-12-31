"""
Benchmark: Segmentation vs Templated Detection
==============================================

Compares the performance of different detection methods focusing on:
1. Success Rate
2. Geometric Drift (Centroid distance)
3. Color Divergence (Impact on final extracted values)
"""

from __future__ import annotations

import logging
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import time
import warnings

# Suppress Colour warnings
import colour.utilities
from colour.utilities import suppress_warnings

from colour_checker_detection.detection import (
    detect_colour_checkers_segmentation,
    detect_colour_checkers_templated,
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC
)
from colour_checker_detection.detection.common import (
    sample_colour_checker
)
import colour
import rawpy
from colour.utilities import as_float_array

# Metadata
__author__ = "Laboratorio de Arqueología Digital UC"
__copyright__ = "Copyright 2018 Laboratorio de Arqueología Digital UC"
__license__ = "Apache-2.0 - https://opensource.org/licenses/Apache-2.0"
__maintainer__ = "Laboratorio de Arqueología Digital UC"
__email__ = "victor.mendez@uc.cl"
__status__ = "Development"

# Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
LOGGER = logging.getLogger(__name__)

# Suppress specific Colour usage warnings related to signature changes
warnings.filterwarnings("ignore", category=colour.utilities.ColourUsageWarning)

def read_raw_high_res(path: Path, brightness: float = 1.5, linear: bool = False):
    """Lectura de RAW: Visual (sRGB) o Lineal (Camera Space)"""
    if not path.exists():
        raise FileNotFoundError(f"{path} no existe")

    with rawpy.imread(str(path)) as raw:
        if linear:
            # Linear 16-bit
            img_rgb = raw.postprocess(
                gamma=(1, 1), 
                no_auto_bright=True,
                use_camera_wb=True, 
                output_color=rawpy.ColorSpace.raw,
                output_bps=16
            )
            return as_float_array(img_rgb) / 65535.0
        else:
            # Visual sRGB
            img_rgb = raw.postprocess(
                use_camera_wb=True,
                bright=brightness, 
                no_auto_bright=True
            )
            return as_float_array(img_rgb) / 255.0

def calculate_centroid(quad):
    """Calcula el centroide de un cuadrilarero (4, 2)"""
    return np.mean(quad, axis=0)

def main():
    base_dir = Path("G:/colour-checker-detection")
    images_dir = base_dir / "colour_checker_detection" / "local_test"
    
    img_files = (
        list(images_dir.glob("*.CR2")) + 
        list(images_dir.glob("*.ARW")) + 
        list(images_dir.glob("*.RAF"))
    )
    
    if not img_files:
        LOGGER.error("No images found in %s", images_dir)
        return


    # Global accumulators for T-Test
    seg_rgb_all = []
    temp_rgb_all = []
    
    # Global Data Collection
    data_rgb = []   # {image, method, idx, rgb, de_ref}
    data_time = []  # {image, method, time}
    data_comp = []  # {image, diff_de, diff_drift}
    
    seg_rgb_all = []
    temp_rgb_all = []

    # Use a context manager to suppress warnings during the whole execution
    with suppress_warnings(python_warnings=True):
        
        for img_path in img_files:
            try:
                # 1. Load Images
                img_srgb = read_raw_high_res(img_path, linear=False)
                img_linear = read_raw_high_res(img_path, linear=True)
                
                # Rotate if vertical
                h, w, _ = img_srgb.shape
                if h > w:
                    img_srgb = cv2.rotate(img_srgb, cv2.ROTATE_90_CLOCKWISE)
                    img_linear = cv2.rotate(img_linear, cv2.ROTATE_90_CLOCKWISE)
                    h, w = w, h # swap
                    
                settings = SETTINGS_DETECTION_COLORCHECKER_CLASSIC.copy()
                settings["working_width"] = w
                settings["working_height"] = h
                
                rect_canon = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

                # Reference Prep for Accuracy (dE vs Ref)
                # We need to know 'true' values to calculate dE for each method individually.
                # Since we don't have ground truth, 'dE 2000' in the first table likely refers to
                # the error AFTER correction using that method's detection.
                # This aligns with correction_swatches.py logic.
                
                # ... skipped full reference setup for brevity, relying on internal `correction` call if needed,
                # BUT `sample_colour_checker` only gives extracted. 
                # To get dE vs Ref, we must run the correction flow or just compare extracted vs Ref (which is HUGE in linear).
                # The user likely wants the "dE after correction".
                # Let's perform the CHECK inside the loop.

                # 2. Run Segmentation
                t0 = time.time()
                res_seg = detect_colour_checkers_segmentation(img_srgb, additional_data=True, **settings)
                time_seg = time.time() - t0
                
                # 3. Run Templated
                t0 = time.time()
                res_temp = detect_colour_checkers_templated(img_srgb, additional_data=True, **settings)
                time_temp = time.time() - t0
                
                has_seg = len(res_seg) > 0
                has_temp = len(res_temp) > 0
                
                if has_seg:
                    data_time.append({"Image": img_path.name, "Method": "Segmentation", "Time": time_seg})
                if has_temp:
                    data_time.append({"Image": img_path.name, "Method": "Templated", "Time": time_temp})

                # Process Colors
                quad_seg = res_seg[0].quadrilateral if has_seg else None
                quad_temp = res_temp[0].quadrilateral if has_temp else None
                
                # Settings for extraction (disable auto-ref)
                settings_extract = settings.copy()
                settings_extract["reference_values"] = None
                
                # Common Ref for dE calc (AdobeRGB D65)
                # (Re-using logic from correction_swatches.py would be ideal, but for now we simplify)
                # We will just capture the RGB values. If dE vs Ref is needed, we need the Ref.
                # To keep it simple and safe: We will report the Extracted RGB.
                # The User's "dE 2000" column in RGB table -> Let's calculate it vs the OTHER method if both exist?
                # OR vs Reference?
                # Given the user put it in a single method row, it implies vs Reference (Accuracy).
                # Determining Reference for RAW is hard without calibration.
                # However, previous output showed "dE_MEAN | 14.88". That was Seg vs Temp.
                # IF the user wants that, we can put it there.
                # But they asked for "Valores RGB".
                
                # Let's collect extraction
                rgb_seg = None
                rgb_temp = None
                
                if has_seg:
                    data_s = sample_colour_checker(img_linear, quad_seg, rect_canon, **settings_extract)
                    if data_s:
                        rgb_seg = data_s.swatch_colours
                        # We don't have individual dE vs Ref unless we run correction.
                        # We will leave dE column as N/A or calculate if we implement the full pipeline.
                        # For benchmark speed, maybe skip full correction?
                        # User asked for "dE 2000" in the table. 
                        # I will populate it with "N/A" for now unless I bring in the full Ref logic.
                        # actually, let's bring the Ref logic back, it's safer.
                        pass

                if has_temp:
                    data_t = sample_colour_checker(img_linear, quad_temp, rect_canon, **settings_extract)
                    if data_t:
                        rgb_temp = data_t.swatch_colours
                        
                # Comparison Data (Seg vs Temp)
                diff_de = None
                if rgb_seg is not None and rgb_temp is not None:
                     # Calculate dE between Seg and Temp
                     RGB_COLOURSPACE = colour.models.RGB_COLOURSPACES['Adobe RGB (1998)']
                     XYZ_s = colour.RGB_to_XYZ(rgb_seg, RGB_COLOURSPACE, RGB_COLOURSPACE.whitepoint, chromatic_adaptation_transform=None)
                     Lab_s = colour.XYZ_to_Lab(XYZ_s, RGB_COLOURSPACE.whitepoint)
                     XYZ_t = colour.RGB_to_XYZ(rgb_temp, RGB_COLOURSPACE, RGB_COLOURSPACE.whitepoint, chromatic_adaptation_transform=None)
                     Lab_t = colour.XYZ_to_Lab(XYZ_t, RGB_COLOURSPACE.whitepoint)
                     
                     de_arr = colour.delta_E(Lab_s, Lab_t, method="CIE 2000")
                     diff_de = np.mean(de_arr)
                     
                     # Comparison Stats
                     c_seg = calculate_centroid(quad_seg)
                     c_temp = calculate_centroid(quad_temp)
                     drift = np.linalg.norm(c_seg - c_temp)
                     
                     data_comp.append({
                         "Image": img_path.name,
                         "Mean dE (Seg vs Temp)": diff_de,
                         "Drift (px)": drift
                     })
                     
                     seg_rgb_all.extend(rgb_seg)
                     temp_rgb_all.extend(rgb_temp)
                     
                     # Fill Main RGB Table
                     # We will use the dE (Seg vs Temp) as the value in the column?
                     # No, that's confusing.
                     # Let's calculate dE vs Reference (D65 AdobeRGB) AFTER correction (Cheung 2004).
                     # This gives "Accuracy".
                     
                     # --- Prepare References (Once) ---
                     if 'swatches_ref_adobe' not in locals():
                         cc_ref_data = colour.characterisation.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
                         xyY_ref = np.array(list(cc_ref_data.data.values()))
                         XYZ_ref_d50 = colour.xyY_to_XYZ(xyY_ref)
                         adobe_rgb = colour.models.RGB_COLOURSPACES['Adobe RGB (1998)']
                         w_d65 = adobe_rgb.whitepoint
                         w_d50 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']
                         w_d50_XYZ = colour.xy_to_XYZ(w_d50)
                         w_d65_XYZ = colour.xy_to_XYZ(w_d65)
                         XYZ_ref_d65 = colour.chromatic_adaptation(XYZ_ref_d50, w_d50_XYZ, w_d65_XYZ)
                         # Simple normalization
                         XYZ_ref_d65_norm = XYZ_ref_d65 * (w_d65_XYZ / XYZ_ref_d65[18])
                         swatches_ref_adobe = colour.XYZ_to_RGB(XYZ_ref_d65_norm, adobe_rgb, w_d65, chromatic_adaptation_transform=None)
                         Lab_ref = colour.XYZ_to_Lab(colour.RGB_to_XYZ(swatches_ref_adobe, adobe_rgb, w_d65, chromatic_adaptation_transform=None), w_d65)
                     
                     # Calculate Accuracy for Seg
                     sw_c_seg = colour.colour_correction(rgb_seg, rgb_seg, swatches_ref_adobe, method='Cheung 2004')
                     XYZ_c_s = colour.RGB_to_XYZ(sw_c_seg, adobe_rgb, w_d65, chromatic_adaptation_transform=None)
                     Lab_c_s = colour.XYZ_to_Lab(XYZ_c_s, w_d65)
                     de_seg_ref = colour.delta_E(Lab_c_s, Lab_ref, method="CIE 2000")
                     
                     # Calculate Accuracy for Temp
                     sw_c_temp = colour.colour_correction(rgb_temp, rgb_temp, swatches_ref_adobe, method='Cheung 2004')
                     XYZ_c_t = colour.RGB_to_XYZ(sw_c_temp, adobe_rgb, w_d65, chromatic_adaptation_transform=None)
                     Lab_c_t = colour.XYZ_to_Lab(XYZ_c_t, w_d65)
                     de_temp_ref = colour.delta_E(Lab_c_t, Lab_ref, method="CIE 2000")
                     
                     for i in range(24):
                         data_rgb.append({
                             "Image": img_path.name, "Method": "Segmentation", "Idx": i, 
                             "RGB": rgb_seg[i], "dE 2000": de_seg_ref[i]
                         })
                         data_rgb.append({
                             "Image": img_path.name, "Method": "Templated", "Idx": i, 
                             "RGB": rgb_temp[i], "dE 2000": de_temp_ref[i]
                         })

            except Exception as e:
                LOGGER.error(f"Error extracting {img_path.name}: {e}")

    # --- PRINT TABLES ---
    
    print("\nValores RGB:")
    print(f"|{'Image':<20} |{'Method':<14} |{'Idx':<4} |{'R, G, B':<30} |{'dE 2000':<8} |")
    print(f"|{'-'*20}-|{'-'*14}-|{'-'*4}-|{'-'*30}-|{'-'*8}-|")
    for row in data_rgb:
        r, g, b = row['RGB']
        rgb_str = f"{r:.4f}, {g:.4f}, {b:.4f}"
        print(f"|{row['Image']:<20} | {row['Method']:<14} |{row['Idx']:<4} | {rgb_str:<30} | {row['dE 2000']:.4f}|")

    print("\nTiempo:")
    print(f"|{'Image':<20} |{'Method':<14} |{'Time (s)':<10} |")
    print(f"|{'-'*20}-|{'-'*14}-|{'-'*10}-|")
    for row in data_time:
        print(f"|{row['Image']:<20} | {row['Method']:<14} |{row['Time']:<10.4f} |")

    print("\nComparación por Imagen (Seg vs Temp):")
    print(f"|{'Image':<20} |{'Mean dE (Diff)':<16} |{'Drift (px)':<12} |")
    print(f"|{'-'*20}-|{'-'*16}-|{'-'*12}-|")
    for row in data_comp:
         print(f"|{row['Image']:<20} | {row['Mean dE (Seg vs Temp)']:<16.4f} | {row['Drift (px)']:<12.2f} |")

    # Statistical Significance
    if seg_rgb_all and temp_rgb_all:
         from scipy import stats as st
         seg_flat = np.array(seg_rgb_all).flatten()
         temp_flat = np.array(temp_rgb_all).flatten()
         t_stat, p_val = st.ttest_rel(seg_flat, temp_flat)
         
         print("\nSignificancia Estadística (Paired T-Test):")
         print(f"|{'Metric':<20} |{'Value':<20} |")
         print(f"|{'-'*20}-|{'-'*20}-|")
         print(f"|{'T-Statistic':<20} | {t_stat:<20.4f} |")
         print(f"|{'P-Value':<20} | {p_val:<20.4e} |")
         sig = "YES" if p_val < 0.05 else "NO"
         print(f"|{'Significant Diff?':<20} | {sig:<20} |")

if __name__ == "__main__":
    main()
