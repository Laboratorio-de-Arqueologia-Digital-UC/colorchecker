import logging
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Importaciones de la librería
from colour_checker_detection.detection import (
    detect_colour_checkers_segmentation,
    detect_colour_checkers_templated,
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC,
)
from colour_checker_detection.detection.common import (
    sample_colour_checker,
)
import rawpy

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


__author__ = "Laboratorio de Arqueología Digital UC"
__copyright__ = "Copyright 2018 Laboratorio de Arqueología Digital UC"
__license__ = "Apache-2.0 - https://opensource.org/licenses/Apache-2.0"
__maintainer__ = "Laboratorio de Arqueología Digital UC"
__email__ = "victor.mendez@uc.cl"
__status__ = "Development"


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
        # User Request: Save to test_results
        output_dir = base_dir / "colour_checker_detection" / "test_results" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)


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
        LOGGER.warning("No se detectó nada por ningún método en %s.", img_path.name)
        return {}

    # Rectangulo canonico para sampleo
    rect_canon = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    results = {}

    # 4. Procesamiento por cada método detectado
    for method_name, det in all_detections.items():
        LOGGER.info(f"Procesando Extracción para: {method_name}")

        # Iniciando Extracción Lineal
        img_linear = read_raw_high_res(img_path, linear=True)
        if is_vertical:
            img_linear = cv2.rotate(img_linear, cv2.ROTATE_90_CLOCKWISE)

        # a) Optimizar Orientación en sRGB
        LOGGER.info(f"  [{method_name}] Optimizando Orientación...")
        visual_data = sample_colour_checker(
            img_srgb, det.quadrilateral, rect_canon, samples=32, **settings
        )

        quad_optimized = visual_data.quadrilateral

        # b) Extraer Linear usando ese Quad FIX
        linear_settings = settings.copy()
        linear_settings["reference_values"] = None

        linear_data = sample_colour_checker(
            img_linear, quad_optimized, rect_canon, samples=32, **linear_settings
        )

        if linear_data:
            vals = linear_data.swatch_colours
            results[method_name] = {"quad": quad_optimized, "swatches": vals}

            if output_dir:
                # 5. Visualización Rápida
                try:
                    # Validar Orientación por Brillo
                    means = np.mean(vals, axis=1)
                    brightest_idx = np.argmax(means)
                    LOGGER.info(
                        f"    [{method_name}] B-Idx: {brightest_idx}, White: {np.round(vals[18], 4)}"
                    )

                    # Plot
                    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
                    ax[0].imshow(np.clip(img_srgb, 0, 1))
                    ax[0].set_title(f"A: {img_path.name} ({method_name})")

                    # Dibujar Quad
                    poly = quad_optimized.astype(int)
                    poly = np.vstack([poly, poly[0]])  # Cerrar
                    ax[0].plot(poly[:, 0], poly[:, 1], "r-", linewidth=2)

                    # Swatches Preview
                    swatch_grid = np.zeros((4, 6, 3))
                    for i in range(24):
                        r, c = divmod(i, 6)
                        color_vis = np.power(np.clip(vals[i], 0, 1), 1 / 2.2)
                        color_vis /= np.max(color_vis) if np.max(color_vis) > 0 else 1
                        swatch_grid[r, c] = color_vis

                        vis_lum = np.mean(color_vis)
                        txt_col = "black" if vis_lum > 0.5 else "white"
                        ax[1].text(
                            c,
                            r,
                            str(i),
                            ha="center",
                            va="center",
                            fontsize=12,
                            color=txt_col,
                            fontweight="bold",
                        )

                    ax[1].imshow(swatch_grid)
                    ax[1].set_title(f"B: Extracción Lineal ({method_name})")

                    # Guardar con prefijo de método
                    out_path = output_dir / f"debug_{method_name}_{img_path.stem}.png"
                    plt.savefig(out_path, bbox_inches="tight")
                    LOGGER.info("Resultado guardado en: %s", out_path)

                    plt.close(fig)
                except Exception as e:
                    LOGGER.warning(f"Error en visualización: {e}")

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


if __name__ == "__main__":
    main()
