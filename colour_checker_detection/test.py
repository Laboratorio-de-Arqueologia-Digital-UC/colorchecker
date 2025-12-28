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
import os
from pathlib import Path
from typing import Any

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from ultralytics import YOLO

# Importaciones de la librería interna
from colour.hints import NDArrayFloat, cast
from colour.utilities import as_float_array

from colour_checker_detection.detection.common import as_int32_array
from colour_checker_detection import (
    detect_colour_checkers_inference,
    detect_colour_checkers_segmentation,
    detect_colour_checkers_templated,
)
from colour_checker_detection.detection.segmentation import (
    segmenter_default,
)
from colour_checker_detection.detection.templated import (
    segmenter_templated,
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
DEFAULT_MODEL_REL_PATH = "colour_checker_detection/models/colour-checker-detection-l-seg.pt"


def read_raw_high_res(path: Path, brightness: float = 1.5) -> NDArrayFloat:
    """Lee un archivo RAW y lo procesa a resolución completa."""
    if not path.exists():
        msg = f"El archivo {path} no existe."
        raise FileNotFoundError(msg)

    with rawpy.imread(str(path)) as raw:
        img_rgb = raw.postprocess(
            use_camera_wb=True,
            bright=brightness,
            no_auto_bright=True
        )
    return as_float_array(img_rgb) / 255.0


def adapter_yolo_inferencer(image: NDArrayFloat, model: YOLO, bbox_cache: list = None) -> list[Any]:
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
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            
            inference_data.append((conf, cls_id, mask))
            
    return inference_data


def get_ideal_swatch_centers(working_width: int, working_height: int) -> np.ndarray:
    """Calcula los centros ideales de los parches."""
    settings = SETTINGS_DETECTION_COLORCHECKER_CLASSIC
    sw_h = settings["swatches_horizontal"] # 6
    sw_v = settings["swatches_vertical"]   # 4
    
    w = working_width
    h = working_height
    
    xs = np.linspace(w / (sw_h * 2), w - w / (sw_h * 2), sw_h)
    ys = np.linspace(h / (sw_v * 2), h - h / (sw_v * 2), sw_v)
    
    centers = []
    for y in ys:
        for x in xs:
            centers.append([x, y])
    return np.array(centers, dtype=np.float32).reshape(-1, 1, 2)


def visualize_comparison(
    image: NDArrayFloat,
    results_dict: dict[str, Any],
    bbox_cache_inference: list,
    debug_info_dict: dict[str, Any], # Nuevo: Diccionario explícito de datos de debug
    filename: str
) -> None:
    """
    Visualiza los resultados de múltiples métodos, incluyendo DEBUG de pasos intermedios.
    """
    
    methods = list(results_dict.keys())
    n_methods = len(methods)
    
    if n_methods == 0:
        return

    img_display = np.clip(np.power(image, 1 / 2.2), 0, 1)
    
    fig, axes = plt.subplots(2, n_methods, figsize=(6 * n_methods, 12), squeeze=False)
    fig.suptitle(f"Comparación de Detección & DEBUG: {filename}", fontsize=16, weight='bold')
   
    for idx, method in enumerate(methods):
        # === FILA 1: RESULTADO FINAL ===
        ax = axes[0, idx]
        ax.imshow(img_display)
        ax.set_title(f"{method} - Resultado", fontsize=12)
        ax.axis('off')
        
        # Inferencia BBox (Directo de YOLO)
        if method == "Inferencia" and bbox_cache_inference:
            # Solo mostrar el de mayor confianza (ya filtrado previamente o mostramos el mejor)
            # Como bbox_cache_inference es una lista global, asumimos que contiene los relevantes.
            # Para mayor limpieza, si hay muchos, solo dibujamos el 1ro.
            if len(bbox_cache_inference) > 0:
                # Ordenar por no podemos (no tenemos conf aqui en la lista simple). 
                # Asumimos que YOLO devuelve ordenados por conf.
                best_bbox = bbox_cache_inference[0] 
                x1, y1, x2, y2 = best_bbox
                w_box, h_box = x2 - x1, y2 - y1
                rect = patches.Rectangle(
                    (x1, y1), w_box, h_box,
                    linewidth=2, edgecolor='#39FF14', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 10, "BBox (YOLO)", color='#39FF14', fontsize=8, weight='bold')

        detections = results_dict[method]
        
        if not detections:
            ax.text(0.5, 0.5, "No detectado", color='red', 
                    ha='center', transform=ax.transAxes, fontsize=12)
        else:
            # Dibujar BBox MAGENTA (Calculado) - Solo para el mejor candidato
            detection = detections[0] # Asumimos el mejor es el primero
            
            if hasattr(detection, 'quadrilateral') and detection.quadrilateral is not None:
                quad = detection.quadrilateral
                x_coords = quad[:, 0]
                y_coords = quad[:, 1]
                min_x, max_x = np.min(x_coords), np.max(x_coords)
                min_y, max_y = np.min(y_coords), np.max(y_coords)
                
                width_bbox = max_x - min_x
                height_bbox = max_y - min_y
                
                rect_calc = patches.Rectangle(
                    (min_x, min_y), width_bbox, height_bbox,
                    linewidth=3, edgecolor='#FF00FF', facecolor='none', linestyle='solid'
                )
                ax.add_patch(rect_calc)

        # === FILA 2: DEBUG PROCESS (Image K + Raw Clusters) ===
        ax_debug = axes[1, idx]
        ax_debug.set_title(f"{method} - Procesamiento (Intermediate)", fontsize=10)
        ax_debug.axis('off')
        
        # Recuperar datos de debug manuales
        debug_data = debug_info_dict.get(method)
        
        if debug_data and hasattr(debug_data, 'image'): # 'image' es image_k en la dataclass de segmentación
            img_debug_k = debug_data.image
            ax_debug.imshow(img_debug_k, cmap='gray')
            
            # Dibujar Clusters
            if hasattr(debug_data, 'clusters'):
                for cluster in debug_data.clusters:
                     if cluster is not None and len(cluster) > 0:
                        poly_points = cluster.reshape(-1, 2)
                        poly = patches.Polygon(
                            poly_points,
                            closed=True,
                            linewidth=1,
                            edgecolor='cyan',
                            facecolor='none',
                            alpha=0.6
                        )
                        ax_debug.add_patch(poly)
            
            n_clusters = len(debug_data.clusters) if debug_data.clusters is not None else 0
            ax_debug.text(10, 10, f"Clusters: {n_clusters}", 
                          color='white', backgroundcolor='black', fontsize=8)
        else:
             ax_debug.text(0.5, 0.5, "Sin datos intermedios", 
                      ha='center', transform=ax_debug.transAxes, color='gray')


    plt.tight_layout()
    plt.show()


def run_benchmark(
    model_path: Path,
    images_dir: Path,
    extensions: tuple[str, ...] = ('.arw', '.raf', '.cr2')
) -> None:
    
    # Check imports debug
    try:
        import sklearn  # noqa: F401
    except ImportError:
        LOGGER.warning("Scikit-learn no está instalado. El método de Plantillas fallará.")

    if not model_path.exists():
        LOGGER.error("Modelo no encontrado en: %s", model_path)
        return

    LOGGER.info("Cargando modelo YOLO: %s", model_path)
    model = YOLO(str(model_path))

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
                LOGGER.info("Imagen Vertical detectada. Rotando -90 grados para procesamiento...")
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
            debug_info = {} # Guardar datos de segmentación crudos
            inference_bboxes = [] 

            # --- MÉTODO 1: SEGMENTACIÓN ---
            LOGGER.info(">> Ejecutando SEGMENTACIÓN...")
            try:
                # A) Detección Completa
                res_seg = detect_colour_checkers_segmentation(
                    img_processing, 
                    segmenter_kwargs=resolution_settings,
                    extractor_kwargs=resolution_settings,
                    additional_data=True,
                    **resolution_settings
                )
                LOGGER.info("   Encontrados: %d", len(res_seg))
                results["Segmentación"] = res_seg

                # B) Debug: Llamada manual al Segmenter para obtener image_k
                # Necesitamos llamar con los MISMOS parámetros
                # Nota: segmenter_default requiere kwargs con settings mezclados
                seg_settings = SETTINGS_DETECTION_COLORCHECKER_CLASSIC.copy()
                seg_settings.update(resolution_settings) # Override working_width
                
                debug_seg_data = segmenter_default(
                    img_processing,
                    additional_data=True,
                    **seg_settings
                )
                debug_info["Segmentación"] = debug_seg_data

            except Exception as e:
                LOGGER.warning("   Fallo en segmentación: %s", e)
                results["Segmentación"] = []

            # --- MÉTODO 2: INFERENCIA (YOLO Wrappeado) ---
            LOGGER.info(">> Ejecutando INFERENCIA (YOLO Wrappeado)...")
            try:
                # Limpiar cache antes de ejecutar
                inference_bboxes.clear() 
                
                custom_inferencer = lambda img, **kwargs: adapter_yolo_inferencer(img, model, inference_bboxes)
                
                res_inf = detect_colour_checkers_inference(
                    img_processing, 
                    inferencer=custom_inferencer,
                    inferred_confidence=0.4,
                    extractor_kwargs=resolution_settings,
                    additional_data=True,
                    **resolution_settings
                )
                LOGGER.info("   Encontrados: %d", len(res_inf))
                
                # FILTRADO DE RESULTADOS: Quedarse solo con el mejor
                if len(res_inf) > 1:
                     LOGGER.info("   Filtrando resultados de inferencia (quedando con el mejor)...")
                     # Asumimos que el primero es el mejor o podríamos ordenar por área/confianza si la tuviéramos
                     # En adapter_yolo_inferencer, YOLO devuelve ordenado por confianza usualmente.
                     res_inf = (res_inf[0],)
                     # También filtrar bbox cache
                     if len(inference_bboxes) > 1:
                          inference_bboxes = [inference_bboxes[0]] # Keep top 1
                
                results["Inferencia"] = res_inf
                # Inferencia no tiene "image_k" en el mismo sentido, debug_info vacío o custom máscaras
            except Exception as e:
                LOGGER.warning("   Fallo en inferencia: %s", e)
                results["Inferencia"] = []

            # --- MÉTODO 3: PLANTILLAS ---
            LOGGER.info(">> Ejecutando PLANTILLAS (Templated)...")
            try:
                res_tpl = detect_colour_checkers_templated(
                    img_processing, 
                    segmenter_kwargs=resolution_settings, 
                    extractor_kwargs=resolution_settings,
                    additional_data=True,
                    **resolution_settings
                )
                LOGGER.info("   Encontrados: %d", len(res_tpl))
                results["Plantillas"] = res_tpl
                
                # B) Debug: Llamada manual a segmenter_templated
                tpl_settings = SETTINGS_DETECTION_COLORCHECKER_CLASSIC.copy()
                tpl_settings.update(resolution_settings)
                
                debug_tpl_data = segmenter_templated(
                    img_processing,
                    additional_data=True,
                    **tpl_settings
                )
                debug_info["Plantillas"] = debug_tpl_data

            except Exception as e:
                LOGGER.warning("   Fallo en plantillas: %s (Tipo: %s)", e, type(e).__name__)
                results["Plantillas"] = []

            # Visualizar comparación
            visualize_comparison(img_processing, results, inference_bboxes, debug_info, img_path.name)

        except Exception:
            LOGGER.exception("Error procesando imagen %s", img_path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark de Detección de ColorCheckers")
    parser.add_argument(
        "--model", 
        type=Path, 
        default=Path(__file__).parent / "models" / "colour-checker-detection-l-seg.pt",
        help="Ruta al modelo YOLO .pt"
    )
    parser.add_argument(
        "--dir", 
        type=Path, 
        default=Path(__file__).parent / "local_test",
        help="Directorio de imágenes de prueba"
    )
    
    args = parser.parse_args()
    run_benchmark(args.model, args.dir)