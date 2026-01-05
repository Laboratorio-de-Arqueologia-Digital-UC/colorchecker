"""
Colour - Checker Detection - Correction Swatches
================================================

Defines the scripts for colour checker detection and correction.

Features:
- Detects ColorChecker Classic using template matching (Post-2014 reference).
- Calculates Color Correction Matrix (CCM) from Linear values to AdobeRGB (D65).
- Exports results in multiple formats:
    - JSON: Comprehensive report with coordinates, RGB values and Delta E metrics.
    - CCM (.txt): 3x3 Correction Matrix in plain text.
    - 3D LUT (.cube): 33x33x33 Lookup Table for external editing.
    - DCP (.dcp): Digital Camera Profile (XML + Binary via dcpTool).
- Visualization: Generates a 6-panel debug image (Detection, Measured, Corrected, Reference, Error, Full Image).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import colour
import cv2
import matplotlib.pyplot as plt
import numpy as np
from colour.characterisation import CCS_COLOURCHECKERS
from colour.difference import delta_E
from colour.models import RGB_COLOURSPACES

# New modules
from colour_checker_detection.calibrator import calculate_ccm
from colour_checker_detection.detection import SETTINGS_DETECTION_COLORCHECKER_CLASSIC
from colour_checker_detection.detection.common import sample_colour_checker
from colour_checker_detection.detector import detect_chart
from colour_checker_detection.io import load_raw_linear, load_raw_visual

__author__ = "Laboratorio de Arqueología Digital UC"
__copyright__ = "Copyright 2018 Laboratorio de Arqueología Digital UC"
__license__ = "Apache-2.0 - https://opensource.org/licenses/Apache-2.0"
__maintainer__ = "Laboratorio de Arqueología Digital UC"
__email__ = "victor.mendez@uc.cl"
__status__ = "Development"

__all__ = ["run_batch_process"]

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def get_dynamic_swatch_centers(
    working_width: int, working_height: int, is_vertical=False
) -> np.ndarray:
    """
    Calcula los centros ideales de los parches adaptándose a la orientación detectada.
    """
    settings = SETTINGS_DETECTION_COLORCHECKER_CLASSIC
    # Por defecto 'swatches_horizontal': 6, 'swatches_vertical': 4
    if is_vertical:
        sw_h = settings.get("swatches_vertical", 4)
        sw_v = settings.get("swatches_horizontal", 6)
    else:
        sw_h = settings.get("swatches_horizontal", 6)
        sw_v = settings.get("swatches_vertical", 4)

    w = working_width
    h = working_height

    # Grid simple de centros
    xs = np.linspace(w / (sw_h * 2), w - w / (sw_h * 2), sw_h)
    ys = np.linspace(h / (sw_v * 2), h - h / (sw_v * 2), sw_v)

    centers = []
    for y in ys:
        for x in xs:
            centers.append([x, y])
    return np.array(centers, dtype=np.float32)  # (24, 2)


def get_transf_matrix_and_centers(w, h, quad):
    """Calcula la matriz de transformación y los centros reproyectados"""
    rect_std = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # Calcular centros teoricos en el espacio canonico
    steps_h = 6
    steps_v = 4

    step_x = w / steps_h
    step_y = h / steps_v

    centers = []
    for row in range(steps_v):
        y = (row + 0.5) * step_y
        for col in range(steps_h):
            x = (col + 0.5) * step_x
            centers.append([x, y])

    rect_centers = np.array(centers, dtype=np.float32).reshape(-1, 1, 2)

    # Homography
    H = cv2.getPerspectiveTransform(rect_std, quad.astype(np.float32))

    # Project back to image
    proj_centers = cv2.perspectiveTransform(rect_centers, H).reshape(-1, 2)
    return proj_centers


def run_batch_process(
    images_dir: Path | str,
    output_dir: Path | str | None = None,
    dcp_tool_path: Path | str | None = None
) -> list[dict]:

    if images_dir is None:
        raise ValueError("images_dir argument is mandatory.")

    images_dir = Path(images_dir)
    if not images_dir.exists():
         raise FileNotFoundError(f"Images directory {images_dir} not found.")

    # BUSCAR IMAGENES (.CR2, .ARW, .RAF)
    img_files = (
        list(images_dir.glob("*.CR2"))
        + list(images_dir.glob("*.ARW"))
        + list(images_dir.glob("*.RAF"))
    )
    if not img_files:
        LOGGER.error("No se encontraron imagenes RAW en %s", images_dir)
        return []

    # Output Dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = images_dir / "results" / timestamp

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # dcpTool lookup
    if dcp_tool_path:
        dcp_tool_path = Path(dcp_tool_path)
    else:
        found = shutil.which("dcpTool")
        if found:
            dcp_tool_path = Path(found)
            LOGGER.info(f"dcpTool encontrado en PATH: {dcp_tool_path}")
        else:
            LOGGER.warning("dcpTool no encontrado en PATH. La generación de DCP binarios se omitirá.")

    batch_results = []

    for img_path in img_files:
        LOGGER.info("=== PROCESANDO IMAGEN: %s ===", img_path.name)

        # 2. Lectura Visual (sRGB) para Detección
        LOGGER.info("Cargando sRGB...")
        try:
            img_srgb = load_raw_visual(img_path, brightness=1.5)
        except Exception as e:
            LOGGER.error(f"Error cargando imagen {img_path}: {e}")
            continue

        # Check rotación vertical
        h, w, _ = img_srgb.shape
        is_vertical = h > w
        if is_vertical:
            LOGGER.info("Detectado Vertical -> Rotando -90")
            img_srgb = cv2.rotate(img_srgb, cv2.ROTATE_90_CLOCKWISE)
            h, w = w, h  # Swap dims

        # Settings
        settings = SETTINGS_DETECTION_COLORCHECKER_CLASSIC.copy()
        settings["working_width"] = w
        settings["working_height"] = h

        # 3. Detección Wrapper
        method_name = "Templated"
        LOGGER.info(f"Ejecutando Detección por {method_name.upper()}...")

        det_found = None
        try:
            # Using new detector module
            det_res = detect_chart(
                img_srgb,
                additional_data=True,
                segmenter_kwargs=settings,
                extractor_kwargs=settings,
                **settings,
            )
            if det_res:
                det_found = det_res[0]
                LOGGER.info(f"   {method_name}: Detectado con éxito.")
            else:
                LOGGER.warning(f"   {method_name}: No se encontró nada.")
                continue  # Skip if not found
        except Exception as e:
            LOGGER.warning(f"   {method_name}: Error -> {e}")
            continue

        # Rectangulo canonico para sampleo
        rect_canon = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        # 4. PREPARACIÓN DE REFERENCIAS (AdobeRGB / D65)
        # -----------------------------------------------
        LOGGER.info("Preparando Referencias AdobeRGB (D65) - Post-2014...")
        # Usamos la versión post-2014 para mayor fidelidad con Passports modernos
        cc_ref_data = CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
        xyY_ref = np.array(list(cc_ref_data.data.values()))
        XYZ_ref_d50 = colour.xyY_to_XYZ(xyY_ref)

        # Adaptación cromática de D50 (ref) a D65 (target)
        adobe_rgb_space = RGB_COLOURSPACES["Adobe RGB (1998)"]
        w_d50_xy = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
        w_d50_XYZ = colour.xy_to_XYZ(w_d50_xy)
        w_d65_xy = adobe_rgb_space.whitepoint
        w_d65_XYZ = colour.xy_to_XYZ(w_d65_xy)
        XYZ_ref_d65 = colour.chromatic_adaptation(XYZ_ref_d50, w_d50_XYZ, w_d65_XYZ)

        # Normalización de Punto Blanco
        white_patch_XYZ = XYZ_ref_d65[18]
        normalization_factor = w_d65_XYZ / white_patch_XYZ
        XYZ_ref_d65_norm = XYZ_ref_d65 * normalization_factor

        # Referencias en AdobeRGB Lineal
        swatches_ref_adobe = colour.XYZ_to_RGB(
            XYZ_ref_d65_norm,
            adobe_rgb_space,
            w_d65_xy,
            chromatic_adaptation_transform=None,
        )

        # 5. Procesamiento
        LOGGER.info(f"Procesando Extracción para: {method_name}")

        # Iniciando Extracción Lineal
        # Using new io module
        try:
            img_linear, as_shot_wb = load_raw_linear(img_path)
            LOGGER.info(f"    WB Cámara detectado: {as_shot_wb}")
        except Exception as e:
            LOGGER.error(f"Error cargando RAW Linear {img_path}: {e}")
            continue

        if is_vertical:
            img_linear = cv2.rotate(img_linear, cv2.ROTATE_90_CLOCKWISE)

        # a) Optimizar Orientación en sRGB
        visual_data = sample_colour_checker(
            img_srgb, det_found.quadrilateral, rect_canon, samples=32, **settings
        )

        quad_optimized = visual_data.quadrilateral

        # b) Extraer Linear usa el Quad FIX
        linear_settings = settings.copy()
        linear_settings["reference_values"] = None

        linear_data = sample_colour_checker(
            img_linear, quad_optimized, rect_canon, samples=32, **linear_settings
        )

        # 6. Corrección y Reporte
        if linear_data:
            swatches_measured = linear_data.swatch_colours

            # --- CÁLCULO DE CCM (Matriz) EXPLICITA ---
            # Using new calibrator module
            M = calculate_ccm(swatches_measured, swatches_ref_adobe)

            # Aplicar la corrección usando la matriz calculada
            swatches_corrected = np.einsum("ij,...j->...i", M, swatches_measured)

            # --- EVALUACIÓN DELTA E ---
            # Clipping para evaluación visual (0-1)
            swatches_corrected_clipped = np.clip(swatches_corrected, 0, 1)

            XYZ_corr = colour.RGB_to_XYZ(
                swatches_corrected_clipped,
                adobe_rgb_space,
                w_d65_xy,
                chromatic_adaptation_transform=None,
            )
            Lab_corr = colour.XYZ_to_Lab(XYZ_corr, adobe_rgb_space.whitepoint)

            XYZ_ref = colour.RGB_to_XYZ(
                swatches_ref_adobe,
                adobe_rgb_space,
                w_d65_xy,
                chromatic_adaptation_transform=None,
            )
            Lab_ref = colour.XYZ_to_Lab(XYZ_ref, adobe_rgb_space.whitepoint)

            de00 = delta_E(Lab_corr, Lab_ref, method="CIE 2000")
            avg_de = np.mean(de00)
            max_de = np.max(de00)
            LOGGER.info(f"    Delta E 2000: Promedio={avg_de:.2f}, Max={max_de:.2f}")

            # --- EXPORTACIONES ---
            base_name = f"correction_{method_name}_{img_path.stem}"

            # 1. JSON
            LOGGER.info("    Generando reporte JSON...")
            try:
                proj_centers = get_transf_matrix_and_centers(w, h, quad_optimized)
                json_data = {
                    "image": img_path.name,
                    "method": method_name,
                    "wb_as_shot": as_shot_wb, # Report WB as requested
                    "reference": "ColorChecker24 - After November 2014 (D65)",
                    "ccm": M.tolist(),
                    "swatches": [],
                }

                for i in range(24):
                    swatch_info = {
                        "index": i,
                        "coordinates_px": proj_centers[i].tolist(),
                        "color_detected_linear": swatches_measured[i].tolist(),
                        "color_corrected_adobe": swatches_corrected[i].tolist(),
                        "color_reference_adobe": swatches_ref_adobe[i].tolist(),
                        "delta_e_2000": float(de00[i]),
                    }
                    json_data["swatches"].append(swatch_info)

                json_path = output_dir / f"{base_name}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=4)
                LOGGER.info("    -> JSON guardado.")

                # Append to batch results
                batch_results.append(json_data)

            except Exception as e:
                LOGGER.error(f"Error generando JSON: {e}")

            # 2. CCM / MAT (Texto / Numpy)
            LOGGER.info("    Generando archivo CCM (.txt)...")
            try:
                ccm_path = output_dir / f"{base_name}_ccm.txt"
                np.savetxt(ccm_path, M, fmt="%.8f")
                LOGGER.info("    -> CCM TXT guardado.")
            except Exception as e:
                LOGGER.error(f"Error generando CCM TXT: {e}")

            # 3. 3D LUT (.cube)
            LOGGER.info("    Generando 3D LUT (.cube)...")
            try:
                size = 33
                LUT = colour.LUT3D(
                    name=f"CCM for {img_path.name}",
                    size=size,
                )
                LUT.table = np.einsum("ij,...j->...i", M, LUT.table)
                LUT.table = np.clip(LUT.table, 0, 1)

                lut_path = output_dir / f"{base_name}.cube"
                colour.write_LUT(LUT, str(lut_path))
                LOGGER.info("    -> .cube guardado.")
            except Exception as e:
                LOGGER.error(f"Error generando LUT: {e}")

            # 4. ICC Profile (Skipped)

            # 5. DCP (XML + Binary via dcpTool)
            LOGGER.info("    Generando DCP (Camera Profile)...")
            try:
                xml_content = f"\"<dcpData>\n    <ProfileName>{base_name}</ProfileName>\n    <ColorMatrix1 Rows=\"3\" Cols=\"3\">\n        {M[0, 0]:.6f} {M[0, 1]:.6f} {M[0, 2]:.6f}\n        {M[1, 0]:.6f} {M[1, 1]:.6f} {M[1, 2]:.6f}\n        {M[2, 0]:.6f} {M[2, 1]:.6f} {M[2, 2]:.6f}\n    </ColorMatrix1>\n    <CalibrationIlluminant1>21</CalibrationIlluminant1>\n</dcpData>\""

                xml_path = output_dir / f"{base_name}.xml"
                dcp_path_out = output_dir / f"{base_name}.dcp"

                # Re-using f-string properly
                xml_content = f'''<dcpData>
    <ProfileName>{base_name}</ProfileName>
    <ColorMatrix1 Rows="3" Cols="3">
        {M[0, 0]:.6f} {M[0, 1]:.6f} {M[0, 2]:.6f}
        {M[1, 0]:.6f} {M[1, 1]:.6f} {M[1, 2]:.6f}
        {M[2, 0]:.6f} {M[2, 1]:.6f} {M[2, 2]:.6f}
    </ColorMatrix1>
    <CalibrationIlluminant1>21</CalibrationIlluminant1>
</dcpData>'''

                with open(xml_path, "w", encoding="utf-8") as f:
                    f.write(xml_content)
                LOGGER.info("    -> XML DCP guardado.")

                if dcp_tool_path and dcp_tool_path.exists():
                    cmd = [str(dcp_tool_path), "-c", str(xml_path), str(dcp_path_out)]
                    LOGGER.info(f"    Ejecutando dcpTool: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True, capture_output=True)
                    LOGGER.info("    -> .dcp Generado Exitosamente.")
                else:
                    LOGGER.info("    dcpTool no disponible o no encontrado. DCP binario omitido.")

            except Exception as e:
                LOGGER.error(f"Error generando DCP: {e}")

            # 7. Visualización
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(2, 3)

            # A: Imagen Original
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(np.clip(img_srgb, 0, 1))
            ax0.set_title(f"A: Detección ({method_name})")

            # Dibujar BBox / Quad
            poly = quad_optimized.astype(int)
            poly = np.vstack([poly, poly[0]])
            ax0.plot(poly[:, 0], poly[:, 1], "r-", linewidth=2)

            # Dibujar centros calculados anteriormente
            ax0.scatter(
                proj_centers[:, 0], proj_centers[:, 1], c="yellow", s=20, marker="x"
            )
            for idx, (px, py) in enumerate(proj_centers):
                ax0.text(
                    px,
                    py,
                    str(idx),
                    color="cyan",
                    fontsize=9,
                    fontweight="bold",
                    ha="right",
                    va="bottom",
                )
            ax0.axis("off")

            # B: Parches Medidos
            ax1 = fig.add_subplot(gs[0, 1])
            grid_meas = np.reshape(swatches_measured, (4, 6, 3))
            vis_meas = grid_meas / (np.max(grid_meas) if np.max(grid_meas) > 0 else 1)
            ax1.imshow(np.power(np.clip(vis_meas, 0, 1), 1 / 2.2))
            ax1.set_title("B: Medido (Índices)")
            for i in range(24):
                r, c = divmod(i, 6)
                vis_lum = np.mean(np.power(vis_meas[r, c], 1 / 2.2))
                txt_col = "white" if vis_lum < 0.5 else "black"
                ax1.text(
                    c,
                    r,
                    str(i),
                    ha="center",
                    va="center",
                    color=txt_col,
                    fontweight="bold",
                    fontsize=10,
                )
            ax1.axis("off")

            # C: Parches Corregidos
            ax2 = fig.add_subplot(gs[0, 2])
            grid_corr = np.reshape(swatches_corrected, (4, 6, 3))
            ax2.imshow(np.power(np.clip(grid_corr, 0, 1), 1 / 2.2))
            ax2.set_title("C: Corregido (AdobeRGB)")
            ax2.axis("off")

            # D: Referencias
            ax3 = fig.add_subplot(gs[1, 1])
            grid_ref = np.reshape(swatches_ref_adobe, (4, 6, 3))
            ax3.imshow(np.power(np.clip(grid_ref, 0, 1), 1 / 2.2))
            ax3.set_title("D: Referencia (Neutral)")
            ax3.axis("off")

            # E: Gráfico de Error
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.bar(range(24), de00, color="teal")
            ax4.axhline(
                float(avg_de), color="red", linestyle="--", label=f"Prom: {avg_de:.2f}"
            )
            ax4.set_title("E: Error Delta E 2000")
            ax4.legend()

            # F: Imagen Aplicada
            ax5 = fig.add_subplot(gs[1, 2])
            LOGGER.info("    Generando vista previa de imagen corregida...")
            # Aplicar matriz a la imagen lineal completa
            h_im, w_im, c_im = img_linear.shape
            img_lin_flat = img_linear.reshape(-1, 3)
            img_corr_flat = np.einsum("ij,...j->...i", M, img_lin_flat)
            img_corrected_full = img_corr_flat.reshape(h_im, w_im, c_im)

            ax5.imshow(np.power(np.clip(img_corrected_full, 0, 1), 1 / 2.2))
            ax5.set_title("F: Imagen Corregida (M)")
            ax5.axis("off")

            plt.tight_layout()

            # Guardar Imagen
            out_path_img = output_dir / f"{base_name}.png"
            plt.savefig(out_path_img, bbox_inches="tight")
            LOGGER.info("Resultado de corrección guardado en: %s", out_path_img)
            plt.close(fig)

    return batch_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colour Checker Detection & Correction Batch Process")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing RAW images")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--dcp_tool_path", type=str, default=None, help="Path to dcpTool.exe")

    args = parser.parse_args()

    try:
        run_batch_process(args.images_dir, args.output_dir, args.dcp_tool_path)
    except Exception as e:
        LOGGER.error(f"Execution Error: {e}")
