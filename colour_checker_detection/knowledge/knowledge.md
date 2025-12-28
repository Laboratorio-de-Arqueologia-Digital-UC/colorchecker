# Knowledge Base & Learnings: Colour Checker Detection

Este documento recopila los hallazgos técnicos, problemas encontrados y soluciones aplicadas durante la implementación y depuración del script de benchmark para `colour-checker-detection`.

## 1. Resolución y Espacio de Coordenadas

### Problema
La librería `colour-checker-detection` está diseñada por defecto para redimensionar las imágenes de entrada a una resolución de trabajo "estándar" (típicamente `working_width=1440` px).
- Las funciones `detect_colour_checkers_*` llaman internamente a `reformat_image`.
- Esta función redimensiona la imagen.
- **Consecuencia Crítica:** Las coordenadas detectadas (bounding boxes, centroides de parches) se devuelven en este espacio de coordenadas reducido (1440px). Si se intenta visualizar o extraer colores sobre la imagen RAW original (ej. 6000px), los resultados aparecen diminutos y desalineados.

### Solución
Para garantizar **Trazabilidad** y precisión en imágenes de alta resolución, se debe **forzar la resolución nativa**.

**Implementación Correcta:**
Se deben pasar `working_width` y `working_height` con las dimensiones de la imagen original en los `kwargs` del **nivel superior** de la llamada a la función.

```python
# CORRECTO
resolution_settings = {"working_width": w, "working_height": h}
detect_colour_checkers_segmentation(
    image, 
    ..., 
    **resolution_settings  # <--- CRUCUAL: Pasar aquí para evitar redimensionado
)
```

**Nota:** Pasarlo solo dentro de `segmenter_kwargs` **NO es suficiente**, ya que la función envoltorio (`detect_...`) procesa la imagen *antes* de llamar al segmentador, usando los `kwargs` globales.

---

## 2. Rotación Implícita de Imágenes Verticales

### Problema
La función interna `reformat_image` (en `detection/common.py`) contiene lógica que **rota automáticamente** cualquier imagen cuya altura sea mayor que su anchura (`h > w`).
- Esto ocurre de manera transparente al usuario.
- **Consecuencia:** Si se pasa una imagen vertical (Portrait), la detección corre sobre la versión rotada (Landscape). Las coordenadas devueltas corresponden a la imagen rotada. Al intentar dibujar estas coordenadas sobre la imagen vertical original, el Bounding Box aparece "fuera" de la carta o en una posición incorrecta.

### Solución
Es imperativo detectar la orientación de la imagen en el script cliente (`test.py`) y manipularla explícitamente **antes** de llamar a la librería.

1.  Detectar si `h > w`.
2.  Si es cierto, rotar la imagen `-90º` (o `90º Clockwise`) para convertirla en Landscape.
3.  Pasar esta imagen rotada a la detección.
4.  Usar esta misma imagen rotada para la visualización.

```python
if h_native > w_native:
    img_processing = cv2.rotate(img_rgb_native, cv2.ROTATE_90_CLOCKWISE)
    # Actualizar dimensiones W/H para los settings de resolución
else:
    img_processing = img_rgb_native
```

---

## 3. Dependencias Ocultas y Registro

### Problema
El método `detect_colour_checkers_templated` ("Plantillas") depende de `scikit-learn` para clustering y ajustes. Aunque la librería esté instalada, puede fallar con `KeyError: 'scikit-learn'` si no está registrada en el sistema de plugins de `colour-science`.

### Solución
Se debe forzar la importación del módulo de requerimientos de la librería antes de ejecutar la detección.

```python
try:
    import colour_checker_detection.utilities.requirements  # noqa: F401
except ImportError:
    pass
```

---

## 4. Visualización Consistente (Bounding Boxes)

### Diferencias entre Métodos
- **Inferencia (YOLO):** Devuelve Bounding Boxes alineados a los ejes (x1, y1, x2, y2).
- **Segmentación / Plantillas:** Devuelven `quadrilaterals` (polígonos de 4 puntos), que pueden estar rotados/perspectivados.

### Estrategia de Unificación
Para permitir una comparación visual directa ("Manzanas con Manzanas"), se debe calcular el **Axis-Aligned Bounding Box (AABB)** envolvente para los métodos que devuelven cuadriláteros.

```python
min_x, max_x = np.min(quad[:, 0]), np.max(quad[:, 0])
min_y, max_y = np.min(quad[:, 1]), np.max(quad[:, 1])
# Dibujar rect(min_x, min_y, w, h)
```

Esto permite verificar si la región de interés general detectada coincide con la de YOLO.

---

## 5. Matplotlib y Rangos de Color

### Problema
`matplotlib.pyplot.imshow` falla con `ValueError` si recibe imágenes de tipo `float` con valores fuera del rango `[0, 1]`. Las imágenes RAW procesadas pueden tener valores > 1.0 (superblancos).

### Solución
Siempre aplicar clipping antes de visualizar.
```python
img_display = np.clip(image, 0, 1)
```

---

## 6. Depuración Avanzada e Imágenes Intermedias (X-Ray)

### Problema
Las funciones de alto nivel (`detect_colour_checkers_*`) encapsulan el proceso y devuelven solo la estructura final `DataDetectionColourChecker`.
- **Datos Perdidos:** La imagen binaria procesada (`image_k`) y los clusters de contornos crudos (usados internamente por el segmentador) NO se exponen en el resultado final, incluso con `additional_data=True`.
- Esto dificulta el debug visual cuando la detección falla (no sabemos si falló el umbralizado o el filtrado de contornos).

### Solución
Para visualizar los pasos intermedios, es necesario realizar una **doble llamada**:
1.  Llamar a la función de alto nivel para obtener el resultado final.
2.  Llamar manualmente a la función de bajo nivel `segmenter_default` (o `segmenter_templated`) con los mismos parámetros.

```python
# Debug Call: Recuperar image_k y clusters
debug_seg_data = segmenter_default(
    image,
    additional_data=True,
    **settings
)
# Ahora debug_seg_data.image contiene la imagen binaria interna
```

---

## 7. Filtrado de Inferencia (YOLO)

### Problema
En imágenes de alta resolución o complejas, el modelo de inferencia puede detectar múltiples candidatos (falsos positivos o múltiples crops de la misma carta).
- Esto ensucia la visualización y las métricas.

### Solución
Implementar un filtro post-inferencia para conservar únicamente la detección de mayor confianza (`conf`).

```python
if len(results) > 1:
    results.sort(key=lambda x: x.conf, reverse=True)
    results = [results[0]]  # Keep Top-1 Only
```
