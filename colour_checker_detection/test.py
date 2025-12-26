import rawpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from ultralytics import YOLO

# --- RUTAS ABSOLUTAS ---
MODEL_PATH = r"G:\colour-checker-detection\colour_checker_detection\models\colour-checker-detection-l-seg.pt"
RESOURCES_PATH = Path(r"G:\colour-checker-detection\colour_checker_detection\local_test")

def read_raw_high_res(path):
    """
    Lee el archivo RAW y lo revela a resolución completa.
    """
    with rawpy.imread(str(path)) as raw:
        # bright=1.5 para compensar sombras en arqueología
        # use_camera_wb=True para mantener fidelidad de color
        return raw.postprocess(use_camera_wb=True, bright=1.5)

def implementar_deteccion_local():
    # 1. Cargar el modelo YOLOv8-seg localmente
    if not Path(MODEL_PATH).exists():
        print(f"[!] ERROR: No se encuentra el modelo en {MODEL_PATH}")
        return
    
    print(f"[*] Cargando modelo local: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 2. Filtrar imágenes en la carpeta local_test
    formatos = ['*.arw', '*.ARW', '*.raf', '*.RAF', '*.cr2', '*.CR2']
    lista = []
    for ext in formatos:
        lista.extend(list(RESOURCES_PATH.glob(ext)))
    
    lista_imagenes = sorted(list(set(lista)))
    
    if not lista_imagenes:
        print(f"[!] No se encontraron archivos RAW en {RESOURCES_PATH}")
        return

    for imagen_path in lista_imagenes:
        print(f"\n{'-'*60}")
        print(f"[+] LOCALIZANDO CARTA EN: {imagen_path.name}")
        
        try:
            # 3. Revelado a resolución completa
            img_rgb = read_raw_high_res(imagen_path)
            h_orig, w_orig = img_rgb.shape[:2]
            print(f"[*] Resolución efectiva: {w_orig}x{h_orig} px")

            # 4. Inferencia YOLOv8
            #imgsz=1280 es el estándar de entrenamiento
            results = model.predict(img_rgb, imgsz=1280, conf=0.4, verbose=False)

            # Preparar visualización (Normalizamos a 0.0-1.0 para matplotlib)
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(np.power(img_rgb / 255.0, 1/2.2))

            encontrado = False
            for r in results:
                if r.boxes:
                    for box in r.boxes:
                        encontrado = True
                        # Coordenadas [x1, y1, x2, y2] escaladas a resolución original
                        coords = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        x1, y1, x2, y2 = coords
                        ancho, alto = x2 - x1, y2 - y1

                        print(f"    - [IA] Confianza: {conf:.2f}")
                        print(f"    - [BBOX REAL] X:{int(x1)} Y:{int(y1)} | W:{int(ancho)} H:{int(alto)}")

                        # Grosor de línea dinámico (aprox 0.5% del ancho de imagen)
                        grosor = max(4, int(w_orig / 200))
                        
                        # Dibujar el Bounding Box (Verde Neón)
                        rect = patches.Rectangle(
                            (x1, y1), ancho, alto,
                            linewidth=grosor, edgecolor='#39FF14', facecolor='none'
                        )
                        ax.add_patch(rect)
                        
                        # Texto proporcional a la resolución
                        ax.text(x1, y1 - (h_orig/80), f"ColorChecker {conf:.2f}", 
                                color='white', weight='bold', fontsize=max(10, int(w_orig/130)),
                                bbox=dict(facecolor='#39FF14', alpha=0.5))
                else:
                    print("    - [!] La IA no encontró la carta en esta toma.")

            ax.set_title(f"Detección de Ubicación: {imagen_path.name}")
            ax.axis('off')
            plt.show()

        except Exception as e:
            print(f"    - [ERROR] {e}")

if __name__ == "__main__":
    implementar_deteccion_local()