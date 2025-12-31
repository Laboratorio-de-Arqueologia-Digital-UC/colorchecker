"""
Colorchecker Pipeline - Correction Swatches
================================================

Defines the scripts for colour checker detection and correction.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import colour
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from colour.characterisation import CCS_COLOURCHECKERS
from colour.difference import delta_E
from colour.models import RGB_COLOURSPACES

# Importaciones de la librería interna
from colour_checker_detection.detection import (
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC,
    detect_colour_checkers_segmentation,
    detect_colour_checkers_templated,
)
from colour_checker_detection.detection.common import (
    sample_colour_checker,
)

__author__ = "Laboratorio de Arqueología Digital UC"
__copyright__ = "Copyright 2018 Laboratorio de Arqueología Digital UC"
__license__ = "Apache-2.0 - https://opensource.org/licenses/Apache-2.0"
__maintainer__ = "Laboratorio de Arqueología Digital UC"
__email__ = "victor.mendez@uc.cl"
__status__ = "Development"

__all__ = ["main"]

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


def read_raw_high_res(path: Path, brightness: float = 1.5, linear: bool = False):
    """Lectura de RAW: Visual (sRGB) o Lineal (Camera Space)"""
    from colour.utilities import as_float_array

    if not path.exists():
        raise FileNotFoundError(f"{path} no existe")

    with rawpy.imread(str(path)) as raw:
        if linear:
            # Modo Lineal: 16-bit para preservar detalle en sombras
            img_rgb = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.raw,
                output_bps=16,
            )
            return as_float_array(img_rgb) / 65535.0
        # Modo Visual
        img_rgb = raw.postprocess(
            use_camera_wb=True, bright=brightness, no_auto_bright=True
        )
        return as_float_array(img_rgb) / 255.0


def process_image(img_path: Path, output_dir: Path | None = None):
    """Procesa una imagen individual y retorna resultados para testing."""
    LOGGER.info("=== PROCESANDO IMAGEN: %s ===", img_path.name)

    # 2. Lectura Visual (sRGB)
    LOGGER.info("Cargando sRGB...")
    try:
        img_srgb = read_raw_high_res(img_path, brightness=1.5, linear=False)
    except Exception as e:
        LOGGER.error(f"Error leyendo imagen {img_path}: {e}")
        return None

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

    # 3. Detección Multi-Método
    methods_to_try = {
        "Plantillas": detect_colour_checkers_templated,
        "Segmentación": detect_colour_checkers_segmentation,
    }

    all_detections = {}
    for name, detection_func in methods_to_try.items():
        LOGGER.info(f"Ejecutando Detección por {name.upper()}...")
        try:
            det_res = detection_func(
                img_srgb,
                additional_data=True,
                segmenter_kwargs=settings,
                extractor_kwargs=settings,
                **settings,
            )
            if det_res:
                all_detections[name] = det_res[0]
                LOGGER.info(f"   {name}: Detectado con éxito.")
            else:
                LOGGER.warning(f"   {name}: No se encontró nada.")
        except Exception as e:
            LOGGER.warning(f"   {name}: Error -> {e}")

    if not all_detections:
        LOGGER.warning(
            "No se detectó nada por ningún método en %s. Saltando.", img_path.name
        )
        return {}

    # Rectangulo canonico para sampleo
    rect_canon = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # 4. PREPARACIÓN DE REFERENCIAS (AdobeRGB / D65)
    LOGGER.info("Preparando Referencias AdobeRGB (D65) - Post-2014...")
    cc_ref_data = CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    xyY_ref = np.array(list(cc_ref_data.data.values()))
    XYZ_ref_d50 = colour.xyY_to_XYZ(xyY_ref)

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

    swatches_ref_adobe = colour.XYZ_to_RGB(
        XYZ_ref_d65_norm, adobe_rgb_space, w_d65_xy, chromatic_adaptation_transform=None
    )

    results = {}

    # 5. Procesamiento por cada método detectado
    for method_name, det in all_detections.items():
        LOGGER.info(f"Procesando Extracción para: {method_name}")

        # Iniciando Extracción Lineal
        img_linear = read_raw_high_res(img_path, linear=True)
        if is_vertical:
            img_linear = cv2.rotate(img_linear, cv2.ROTATE_90_CLOCKWISE)

        LOGGER.info(f"  [{method_name}] Optimizando Orientación...")
        visual_data = sample_colour_checker(
            img_srgb, det.quadrilateral, rect_canon, samples=32, **settings
        )

        quad_optimized = visual_data.quadrilateral

        linear_settings = settings.copy()
        linear_settings["reference_values"] = None

        linear_data = sample_colour_checker(
            img_linear, quad_optimized, rect_canon, samples=32, **linear_settings
        )

        # 5. Corrección de Color y Evaluación
        if linear_data:
            swatches_measured = linear_data.swatch_colours

            # --- CÁLCULO DE CCM --- (Required for testing)
            LOGGER.info(
                f"    [{method_name}] Calculando Corrección de Color (Cheung 2004)..."
            )
            # colour.colour_correction usually returns corrected swatches.
            # To get the CCM itself using colour library is tricky as `colour_correction` wraps it.
            # But the user asked to verify "Que la matriz CCM sea 3x3".
            # The function `colour.colour_correction` doesn't return matrix.
            # However, `colour.characterisation.matrix_colour_correction_Cheung2004` DOES return it if we call it directly,
            # OR we can assume if the output is correct, the internal matrix was 3x3.
            # BUT the requirement is explicit: "Que la matriz CCM sea 3x3".
            # I should verify if I can extract it.
            # `colour.colour_correction` implementation calls `colour.matrix_colour_correction_Cheung2004` (swatches -> swatches)
            # or uses `CCM` if we used fitting.
            # Let's perform fitting explicitly to get the CCM for the test report.

            CCM = colour.characterisation.matrix_colour_correction_Cheung2004(
                swatches_measured, swatches_ref_adobe
            )

            # Now apply it
            swatches_corrected = colour.algebra.vector_dot(CCM, swatches_measured)

            # Store in results
            results[method_name] = {
                "CCM": CCM,
                "swatches_measured": swatches_measured,
                "swatches_corrected": swatches_corrected,
                "swatches_ref": swatches_ref_adobe,
            }

            # --- EVALUACIÓN DELTA E ---
            XYZ_corr = colour.RGB_to_XYZ(
                swatches_corrected,
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
            LOGGER.info(
                f"    [{method_name}] Delta E 2000: Promedio={avg_de:.2f}, Max={max_de:.2f}"
            )

            if output_dir:
                # 6. Visualización Expandida
                fig = plt.figure(figsize=(18, 10))
                gs = fig.add_gridspec(2, 3)

                # A: Imagen Original con Detección e Índices
                ax0 = fig.add_subplot(gs[0, 0])
                ax0.imshow(np.clip(img_srgb, 0, 1))
                ax0.set_title(f"A: Detección ({method_name})")

                # Dibujar BBox / Quad
                poly = quad_optimized.astype(int)
                poly = np.vstack([poly, poly[0]])
                ax0.plot(poly[:, 0], poly[:, 1], "r-", linewidth=2)

                # Restaurar Índices y Centros
                try:
                    rect_std = np.array(
                        [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
                    )
                    from colour_checker_detection.detection.common import swatch_masks

                    masks = swatch_masks(w, h, 6, 4, samples=32)
                    rect_centers = []
                    for mask in masks:
                        cy = (mask[0] + mask[1]) / 2.0
                        cx = (mask[2] + mask[3]) / 2.0
                        rect_centers.append([cx, cy])
                    rect_centers = np.array(rect_centers, dtype=np.float32).reshape(
                        -1, 1, 2
                    )
                    H_vis = cv2.getPerspectiveTransform(
                        rect_std, quad_optimized.astype(np.float32)
                    )
                    proj_centers = cv2.perspectiveTransform(
                        rect_centers, H_vis
                    ).reshape(-1, 2)

                    ax0.scatter(
                        proj_centers[:, 0],
                        proj_centers[:, 1],
                        c="yellow",
                        s=20,
                        marker="x",
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
                except Exception as e:
                    LOGGER.warning("Visual projection failed: %s", e)
                ax0.axis("off")

                # B: Parches Medidos
                ax1 = fig.add_subplot(gs[0, 1])
                grid_meas = np.reshape(swatches_measured, (4, 6, 3))
                vis_meas = grid_meas / (
                    np.max(grid_meas) if np.max(grid_meas) > 0 else 1
                )
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
                    avg_de, color="red", linestyle="--", label=f"Prom: {avg_de:.2f}"
                )
                ax4.set_title("E: Error Delta E 2000")
                ax4.set_xlabel("Índice")
                ax4.set_ylabel("Delta E")
                ax4.legend()

                # F: Imagen Aplicada
                ax5 = fig.add_subplot(gs[1, 2])
                LOGGER.info(
                    f"    [{method_name}] Generando vista previa de imagen corregida..."
                )
                # Apply CCM explicitly
                # img_corrected_full = colour.colour_correction(..., method='Cheung 2004') calls fitting internally?
                # Using vecmul is safer as we calculated CCM
                img_corrected_full = colour.algebra.vecmul(CCM, img_linear)

                ax5.imshow(np.power(np.clip(img_corrected_full, 0, 1), 1 / 2.2))
                ax5.set_title("F: Imagen Corregida")
                ax5.axis("off")

                plt.tight_layout()

                # Guardar
                out_path = output_dir / f"correction_{method_name}_{img_path.stem}.png"
                plt.savefig(out_path, bbox_inches="tight")
                LOGGER.info("Resultado de corrección guardado en: %s", out_path)

                plt.close(fig)

    return results


def main(images_dir: Path | None = None, output_dir: Path | None = None):
    # 1. Configuración
    if images_dir is None:
        base_dir = Path("G:/colour-checker-detection")  # Asumiendo path del user
        images_dir = base_dir / "colour_checker_detection" / "local_test"

    # BUSCAR IMAGENES (.CR2, .ARW, .RAF)
    img_files = (
        list(images_dir.glob("*.CR2"))
        + list(images_dir.glob("*.ARW"))
        + list(images_dir.glob("*.RAF"))
    )
    if not img_files:
        LOGGER.error("No se encontraron imagenes RAW en %s", images_dir)
        return

    # Output Dir
    if output_dir is None:
        base_dir = Path("G:/colour-checker-detection")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_dir / "colour_checker_detection" / "test_results" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in img_files:
        process_image(img_path, output_dir)


if __name__ == "__main__":
    main()
