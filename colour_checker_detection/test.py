"""
Local Traceability Test & Benchmark
===================================

Script para pruebas locales de trazabilidad de ColorCheckers en imágenes RAW
de alta resolución, comparando múltiples métodos de detección:
1. Inferencia (YOLOv8)
2. Segmentación (Visión Clásica)
3. Plantillas (Homografía)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from matplotlib import patches

try:
    from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]
except ImportError:
    YOLO = None

# Importaciones de la librería interna
from colour.hints import NDArrayFloat
from colour.utilities import as_float_array

__author__ = "Laboratorio de Arqueología Digital UC"
__copyright__ = "Copyright 2018 Laboratorio de Arqueología Digital UC"
__license__ = "Apache-2.0 - https://opensource.org/licenses/Apache-2.0"
__maintainer__ = "Laboratorio de Arqueología Digital UC"
__email__ = "victor.mendez@uc.cl"
__status__ = "Development"

from colour_checker_detection import (
    detect_colour_checkers_inference,
    detect_colour_checkers_segmentation,
    detect_colour_checkers_templated,
)
from colour_checker_detection.detection.common import (
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC,
)

# Force registration of requirements (Fix for KeyError: 'scikit-learn')
try:
    import colour_checker_detection.utilities.requirements  # noqa: F401
except ImportError:
    pass

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# Constantes por defecto
DEFAULT_MODEL_REL_PATH = (
    "colour_checker_detection/models/colour-checker-detection-l-seg.pt"
)


def read_raw_high_res(path: Path, brightness: float = 1.5) -> NDArrayFloat:
    """Lee un archivo RAW y lo procesa a resolución completa."""
    if not path.exists():
        msg = f"El archivo {path} no existe."
        raise FileNotFoundError(msg)

    with rawpy.imread(str(path)) as raw:
        img_rgb = raw.postprocess(
            use_camera_wb=True, bright=brightness, no_auto_bright=True
        )
    return as_float_array(img_rgb) / 255.0


def adapter_yolo_inferencer(
    image: NDArrayFloat, model: Any, bbox_cache: list | None = None
) -> list[Any]:
    """
    Adaptador para convertir la salida de YOLOv8 al formato esperado por
    `detect_colour_checkers_inference` de la librería.
    """
    # YOLO espera uint8 o float correcto. Convertimos para asegurar.
    img_input = (image * 255).astype(np.uint8)

    # Inferencia
    results = model.predict(img_input, imgsz=1280, conf=0.4, verbose=False)

    inference_data = []

    for r in results:
        if bbox_cache is not None and r.boxes:
            for box in r.boxes:
                bbox_cache.append(box.xyxy[0].cpu().numpy())

        if not r.masks:
            continue

        masks = r.masks.data.cpu().numpy()  # (N, H, W)
        boxes = r.boxes

        for i, mask in enumerate(masks):
            # Redimensionar máscara al tamaño original de la imagen si difiere
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())

            inference_data.append((conf, cls_id, mask))

    return inference_data


def get_dynamic_swatch_centers(
    working_width: int, working_height: int, is_vertical=False
) -> np.ndarray:
    """
    Calcula los centros ideales de los parches adaptándose a la orientación detectada.
    """
    settings = SETTINGS_DETECTION_COLORCHECKER_CLASSIC
    # Por defecto 'swatches_horizontal': 6, 'swatches_vertical': 4
    if is_vertical:
        sw_h = settings["swatches_vertical"]  # 4
        sw_v = settings["swatches_horizontal"]  # 6
    else:
        sw_h = settings["swatches_horizontal"]  # 6
        sw_v = settings["swatches_vertical"]  # 4

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


def visualize_comparison(
    image: NDArrayFloat,
    results_dict: dict[str, Any],
    bbox_cache_inference: list,
    filename: str,
    resolution_settings: dict[str, int],
) -> None:
    """
    Visualiza Resultados + Swatches Batches (CON DETECCIÓN DINÁMICA DE TOPOLOGÍA)
    """

    methods = list(results_dict.keys())
    n_methods = len(methods)

    if n_methods == 0:
        return

    img_display = np.clip(np.power(image, 1 / 2.2), 0, 1)

    # Config base
    w_work = resolution_settings["working_width"]
    h_work = resolution_settings["working_height"]

    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6), squeeze=False)
    fig.suptitle(
        f"Detección de Batches de Color: {filename}", fontsize=16, weight="bold"
    )

    for idx, method in enumerate(methods):
        ax = axes[0, idx]
        ax.imshow(img_display)
        ax.set_title(method, fontsize=12)
        ax.axis("off")

        # 1. BBox YOLO
        if method == "Inferencia" and bbox_cache_inference:
            if len(bbox_cache_inference) > 0:
                best_bbox = bbox_cache_inference[0]
                x1, y1, x2, y2 = best_bbox
                w_box, h_box = x2 - x1, y2 - y1
                rect = patches.Rectangle(
                    (x1, y1),
                    w_box,
                    h_box,
                    linewidth=2,
                    edgecolor="#39FF14",
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    x1,
                    y1 - 10,
                    "BBox (YOLO)",
                    color="#39FF14",
                    fontsize=8,
                    weight="bold",
                )

        detections = results_dict[method]

        if not detections:
            ax.text(
                0.5,
                0.5,
                "No detectado",
                color="red",
                ha="center",
                transform=ax.transAxes,
                fontsize=12,
            )
        else:
            detection = detections[0]

            if (
                hasattr(detection, "quadrilateral")
                and detection.quadrilateral is not None
            ):
                quad = detection.quadrilateral
                x_coords = quad[:, 0]
                y_coords = quad[:, 1]
                min_x, max_x = np.min(x_coords), np.max(x_coords)
                min_y, max_y = np.min(y_coords), np.max(y_coords)

                width_bbox = max_x - min_x
                height_bbox = max_y - min_y

                # 2. BBox Calculation
                rect_calc = patches.Rectangle(
                    (min_x, min_y),
                    width_bbox,
                    height_bbox,
                    linewidth=3,
                    edgecolor="#FF00FF",
                    facecolor="none",
                    linestyle="solid",
                )
                ax.add_patch(rect_calc)

                # 3. VISUALIZE SWATCHES (DYNAMC TOPOLOGY CHECK)
                try:
                    # Determinar si el quad detectado es "Vertical" (Alto > Ancho en espacio de proyección)
                    # Medimos longitudes de lados: Lado 0-1 (Top?), Lado 1-2 (Right?)
                    # Orden usual de puntos en quad: TL, TR, BR, BL (o similar)
                    p0, p1, p2, _p3 = quad

                    # Distancias
                    d01 = np.linalg.norm(
                        p1 - p0
                    )  # Primer lado (asumimos Height/Right?)
                    d12 = np.linalg.norm(
                        p2 - p1
                    )  # Segundo lado (asumimos Width/Bottom?)

                    # CORRECCIÓN: La lógica anterior d12 > d01 detectaba Horizontal como Vertical.
                    # Si asumimos orden TR, BR, BL, TL: d01=Height, d12=Width.
                    # Vertical implica Height > Width => d01 > d12.
                    # Horizontal implica Width > Height => d12 > d01.
                    is_vertical_quad = d01 > d12

                    # Generar centros ideales según topología detectada
                    ideal_centers = get_dynamic_swatch_centers(
                        w_work, h_work, is_vertical=bool(is_vertical_quad)
                    )

                    # Definir rectángulo canónico correspondiente
                    # Si es vertical: (0,0) -> (w, h) PERO la grilla es 4x6
                    rectangle = np.array(
                        [[w_work, 0], [w_work, h_work], [0, h_work], [0, 0]],
                        dtype=np.float32,
                    )

                    H = cv2.getPerspectiveTransform(rectangle, quad.astype(np.float32))

                    centers_reshaped = ideal_centers.reshape(-1, 1, 2)
                    projected_centers = cv2.perspectiveTransform(centers_reshaped, H)
                    projected_centers = projected_centers.reshape(-1, 2)

                    ax.scatter(
                        projected_centers[:, 0],
                        projected_centers[:, 1],
                        c="yellow",
                        s=10,
                        marker="x",
                        label="Swatches",
                    )

                except Exception as e:
                    LOGGER.warning("Error proyectando swatches: %s", e)

    plt.tight_layout()
    plt.show()


def run_benchmark(
    model_path: Path,
    images_dir: Path,
    extensions: tuple[str, ...] = (".arw", ".raf", ".cr2"),
) -> None:
    # Check imports debug
    try:
        import sklearn  # noqa: F401
    except ImportError:
        LOGGER.warning(
            "Scikit-learn no está instalado. El método de Plantillas fallará."
        )

    if not model_path.exists():
        LOGGER.error("Modelo no encontrado en: %s", model_path)
        return

    model = None
    if YOLO is not None:
        LOGGER.info("Cargando modelo YOLO: %s", model_path)
        model = YOLO(str(model_path))
    else:
        LOGGER.warning("Ultralytics not installed. Inference method will be skipped.")

    image_files = []
    for ext in extensions:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
        image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
    image_files = sorted(list(set(image_files)))

    if not image_files:
        LOGGER.warning("No se encontraron imágenes en %s", images_dir)
        return

    for img_path in image_files:
        LOGGER.info("=" * 60)
        LOGGER.info("Evaluando: %s", img_path.name)

        try:
            img_rgb_native = read_raw_high_res(img_path)
            h_native, w_native = img_rgb_native.shape[:2]

            LOGGER.info("Dimensiones Nativas: %dx%d px", w_native, h_native)

            # --- CORRECCIÓN DE ROTACIÓN ---
            is_rotated = False
            img_processing = img_rgb_native

            if h_native > w_native:
                LOGGER.info(
                    "Imagen Vertical detectada. Rotando -90 grados para procesamiento..."
                )
                img_processing = cv2.rotate(img_rgb_native, cv2.ROTATE_90_CLOCKWISE)
                is_rotated = True

            h_proc, w_proc = img_processing.shape[:2]
            LOGGER.info("Dimensiones Procesamiento: %dx%d px", w_proc, h_proc)

            # Settings
            resolution_settings = {
                "working_width": w_proc,
                "working_height": h_proc,
            }

            results = {}
            inference_bboxes = []

            # --- MÉTODO 1: SEGMENTACIÓN ---
            LOGGER.info(">> Ejecutando SEGMENTACIÓN...")
            try:
                # Solo Detección (Sin debug manual)
                res_seg = detect_colour_checkers_segmentation(
                    img_processing,
                    segmenter_kwargs=resolution_settings,
                    extractor_kwargs=resolution_settings,
                    additional_data=True,
                    **resolution_settings,
                )
                LOGGER.info("   Encontrados: %d", len(res_seg))
                results["Segmentación"] = res_seg
            except Exception as e:
                LOGGER.warning("   Fallo en segmentación: %s", e)
                results["Segmentación"] = []

            # --- MÉTODO 2: INFERENCIA (YOLO Wrappeado) ---
            if model is not None:
                LOGGER.info(">> Ejecutando INFERENCIA (YOLO Wrappeado)...")
                try:
                    inference_bboxes.clear()

                    def custom_inferencer(img, **kwargs):
                        return adapter_yolo_inferencer(img, model, inference_bboxes)

                    res_inf = detect_colour_checkers_inference(
                        img_processing,
                        inferencer=custom_inferencer,
                        inferred_confidence=0.4,
                        extractor_kwargs=resolution_settings,
                        additional_data=True,
                        **resolution_settings,
                    )
                    LOGGER.info("   Encontrados: %d", len(res_inf))

                    # FILTRADO DE RESULTADOS: Top-1
                    if len(res_inf) > 1:
                        LOGGER.info(
                            "   Filtrando resultados de inferencia (quedando con el mejor)..."
                        )
                        res_inf = (res_inf[0],)
                        if len(inference_bboxes) > 1:
                            inference_bboxes = [inference_bboxes[0]]  # Keep top 1

                    results["Inferencia"] = res_inf
                except Exception as e:
                    LOGGER.warning("   Fallo en inferencia: %s", e)
                    results["Inferencia"] = []
            else:
                LOGGER.info(
                    ">> INFERENCIA (YOLO Wrappeado) - Skipped (Module not found)"
                )

            # --- MÉTODO 3: PLANTILLAS ---
            LOGGER.info(">> Ejecutando PLANTILLAS (Templated)...")
            try:
                res_tpl = detect_colour_checkers_templated(
                    img_processing,
                    segmenter_kwargs=resolution_settings,
                    extractor_kwargs=resolution_settings,
                    additional_data=True,
                    **resolution_settings,
                )
                LOGGER.info("   Encontrados: %d", len(res_tpl))
                results["Plantillas"] = res_tpl
            except Exception as e:
                LOGGER.warning(
                    "   Fallo en plantillas: %s (Tipo: %s)", e, type(e).__name__
                )
                results["Plantillas"] = []

            # Visualizar comparación con Swatches
            visualize_comparison(
                img_processing,  # type: ignore
                results,
                inference_bboxes,
                img_path.name,
                resolution_settings,
            )

        except Exception:
            LOGGER.exception("Error procesando imagen %s", img_path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark de Detección de ColorCheckers"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent / "models" / "colour-checker-detection-l-seg.pt",
        help="Ruta al modelo YOLO .pt",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).parent / "local_test",
        help="Directorio de imágenes de prueba",
    )

    args = parser.parse_args()
    run_benchmark(args.model, args.dir)
