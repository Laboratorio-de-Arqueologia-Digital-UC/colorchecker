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
Siempre aplicar clipping antes de visualizar. Para imágenes RAW lineales, es vital aplicar también una **Corrección Gamma** (típicamente 1/2.2) para que la imagen no se vea excesivamente oscura.

```python
# Corrección Gamma + Clipping
img_display = np.clip(np.power(image, 1 / 2.2), 0, 1)
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
    results = [results[0]]  # Keep Top-1 Only
```

---

## 8. Visualización de Swatches y Topología Dinámica

### Problema
Para visualizar correctamente la posición de los parches de color (swatches) sobre un *quadrilateral* detectado que puede estar rotado o en perspectiva, no basta con una grilla estática. Además, la interpretación de "vertical vs horizontal" basada en las longitudes de los lados del cuadrilátero puede ser engañosa si no se asume un orden de vértices consistente.

**Error Detectado:** Comparar `lado1 > lado2` ciegamente puede llevar a clasificar una carta horizontal como vertical si el orden de los puntos empieza en una esquina diferente.

### Solución
1.  **Cálculo de Homografía:** Usar `cv2.getPerspectiveTransform` para mapear los centros ideales de una carta "plana" (canónica) hacia el cuadrilátero detectado.
2.  **Topología Dinámica:** Verificar la orientación geométrica real (e.g., distancia Top-Left a Bottom-Left vs Top-Left a Top-Right) para decidir si generar una grilla de centros Vertical (4x6) u Horizontal (6x4).

```python
# Determinar orientación basado en geometría proyectada
d_height = np.linalg.norm(p1 - p0)
d_width = np.linalg.norm(p2 - p1)
is_vertical = d_height > d_width

# Generar centros y proyectar
ideal_centers = get_dynamic_swatch_centers(w, h, is_vertical)
H = cv2.getPerspectiveTransform(rect_canonico, quad_detectado)
centers_projected = cv2.perspectiveTransform(ideal_centers, H)
```

---

## 9. Adaptador Custom para Modelos Externos (YOLO)

### Contexto
La librería soporta inferencia custom mediante un callback `inferencer`. Sin embargo, YOLOv8 devuelve objetos complejos (`Results` object con `boxes`, `masks`), mientras que la librería espera una lista simple de tuplas `(confidence, class_id, mask)`.

### Solución
Implementar un **Adaptador** que traduzca la salida. Es crítico manejar el redimensionamiento de las máscaras, ya que YOLO puede devolver máscaras a menor resolución que la imagen de entrada original.

```python
def adapter_yolo_inferencer(image, model, ...):
    results = model.predict(image, ...)
    inference_data = []
    
    for r in results:
        # Extraer máscaras y redimensionar si es necesario
        mask = cv2.resize(r.masks.data, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        inference_data.append((conf, cls_id, mask))
        
    return inference_data
```

---

## 10. Análisis Estructural del Repositorio

### Arbol de Archivos (Simplificado)
```text
colour-checker-detection/
├── colour_checker_detection/
│   ├── detection/          # Lógica Core
│   │   ├── common.py       # Utilidades y Constantes (SETTINGS_, reformat_image)
│   │   ├── inference.py    # Wrapper para YOLO (Llama a scripts/inference.py)
│   │   ├── segmentation.py # Detección clásica (OpenCV)
│   │   ├── templated.py    # Matching por plantillas (sklearn)
│   │   └── plotting.py     # Funciones de visualización
│   ├── scripts/
│   │   └── inference.py    # CLI Script externo para aislar dependencia YOLO AGPL
│   ├── models/             # Pesos .pt de YOLO
│   ├── utilities/          # Helpers (requirements.py)
│   └── test.py             # [USER] Script de desarrollo local y benchmark
├── docs/                   # Documentación Sphinx
├── tasks.py                # Definición de tareas 'invoke' (CI/CD)
└── pyproject.toml          # Configuración de build y dependencias
```

### Arquitectura de Inferencia Desacoplada
Una decisión crítica de diseño en la librería es el **desacoplamiento de la inferencia YOLO**:
- **Diseño Oficial (`inference.py`)**: La función `inferencer_default` no importa `ultralytics` directamente. En su lugar, crea un proceso secundario (`subprocess.call`) que ejecuta `scripts/inference.py`.
    - *Razón*: Licencia. `colour-checker-detection` es BSD-3-Clause, mientras que YOLOv8 es AGPL. El desacoplamiento vía CLI aísla la contaminación de la licencia.
    - *Costo*: Alto overhead por I/O de disco (escribir imagen temporal -> procesar -> leer resultado).
    - *Impacto en `test.py`*: Nuestro script `test.py` rompe este aislamiento por eficiencia ("Adapter Pattern"), importando `ultralytics` directamente. Esto es aceptable para uso local/privado, pero cambiaría la licencia si se distribuye.

### Flujo de Datos y Coordenadas
1.  **Entrada**: Imagen RAW o Alta Resolución.
2.  **Pre-proceso (`common.reformat_image`)**: La librería *siempre* intenta redimensionar a `working_width` (1440px) y rotar si `h > w`.
3.  **Detección**: Ocurre en el espacio de 1440px.
4.  **Salida**: Las coordenadas retornadas están en 1440px.
5.  **Desafío**: Mapear estas coordenadas de vuelta a la imagen original de 6000px+ requiere mantener un registro preciso del factor de escala y la rotación aplicada, algo que la librería hace opacamente. Por eso en `test.py` calculamos nuestras propias transformaciones o forzamos `working_width` nativo.

---

## 11. Identidad y Construcción del Repositorio

### ¿Qué es?
`colour-checker-detection` es un paquete de software científico desarrollado en **Python** como parte del ecosistema **Colour Science**. Es una librería especializada y modular diseñada para ser integrada en pipelines de procesamiento de imágenes más grandes.

### ¿Para qué sirve?
Su objetivo único es la **detección y localización automatizada de cartas de color** (ColorCheckers) en imágenes digitales.
Sus casos de uso principales son:
1.  **Calibración de Color**: Automatizar la extracción de valores RGB de los parches de referencia para generar perfiles de color (ICC/DCP) o transformaciones de corrección de color.
2.  **Visión Artificial & Fotogrametría**: Servir como paso de pre-procesamiento para normalizar imágenes antes de realizar reconstrucciones 3D o análisis científicos.
3.  **Benchmarking**: Comparar la fidelidad de color de diferentes cámaras o iluminaciones.

### ¿Cómo está construido?
La arquitectura del proyecto sigue estándares modernos de ingeniería de software en Python (PEP 517), priorizando la calidad del código y la separación de preocupaciones.

#### 1. Stack Tecnológico
-   **Lenguaje**: Python 3.10+.
-   **Dependencias Core**:
    -   `numpy`: Cálculo numérico y manejo de arrays.
    -   `opencv-python`: Procesamiento de imágenes (visión clásica).
    -   `colour-science`: Fundamentos de colorimetría.
-   **Dependencias Opcionales**:
    -   `ultralytics`: Para inferencia basada en Deep Learning (YOLO).
    -   `scikit-learn`: Para algoritmos de clustering en detección por plantillas.

#### 2. Ingeniería y Build System
-   **Backend de Construcción**: Usa **Hatchling** (definido en `[build-system]`), lo que facilita la creación de paquetes reproducibles (`wheels`, `sdist`).
-   **Orquestación de Tareas**: Utiliza **Invoke** (`tasks.py`) para definir y ejecutar comandos complejos de desarrollo como:
    -   `invoke methods`: Ejecutar tests, formateo, linting y construcción de documentación.
    -   `invoke quality`: Asegurar estándares estrictos con `pyright` (tipado estático) y `ruff`.
-   **Estrategia de Licenciamiento**: Mantiene una licencia permisiva **BSD-3-Clause** para el núcleo, mientras aísla componentes con licencias virales (AGPL de YOLO) en scripts externos (`scripts/inference.py`), demostrando una arquitectura consciente de lo legal.


---

## 12. Extracción Lineal y Orientación (Linear Extraction)

### Problema: Auto-Rotación Fallida en Datos Lineales
Cuando procesamos imágenes RAW en modo "Lineal" (`gamma=1.0`, `no_auto_bright=True`), los valores RGB resultantes son físicamente radiométricos pero visualmente muy oscuros (ej. valores típicos ~0.005 en vez de ~0.5).

La función `sample_colour_checker` incluye una lógica interna de **Auto-Rotación**:
1. Extrae los colores.
2. Los compara (MSE) con valores de referencia (`settings.reference_values`), los cuales están típicamente en espacio sRGB/corrección gamma.
3. Si rotar la carta 90º/180º reduce el error, la función asume que la carta detectada estaba rotada y reordena los parches.

**El Fallo:** Al comparar datos Lineales Oscuros (0.005) vs Referencia sRGB Brillante (0.5), el error MSE es enorme en cualquier orientación. La lógica de minimización se vuelve ruido y a menudo selecciona una orientación incorrecta (ej. 180º).

### Solución
Al realizar la **Segunda Pasada (Extracción Científica/Lineal)**, se debe desactivar explícitamente esta auto-rotación. Confiamos ciegamente en que la orientación del `quadrilateral` detectado en la primera pasada (visual) es la correcta.

```python
# Desactivar auto-rotación pasando reference_values=None
linear_settings = resolution_settings.copy()
linear_settings["reference_values"] = None 

linear_chart_data = sample_colour_checker(
     img_linear, 
     best_detection.quadrilateral, # Usar orientación de la detección visual
     ...,
     **linear_settings # <--- CRÍTICO
)
```

---

## 13. Entorno de Proyecto y Flujo de Trabajo (UV)

### Herramienta Principal: `uv`
Este proyecto utiliza **`uv`** como gestor de paquetes y entorno virtual. Esto es mandatorio para garantizar la reproducibilidad y evitar conflictos de dependencias (especialmente con librerías pesadas como `ultralytics`, `opencv` y `colour-science`).

### Reglas de Oro
1.  **NO usar pip global**: Nunca instalar paquetes con `pip install` directa en el entorno global del usuario.
2.  **Ejecución vía `uv run`**: Todos los scripts, tests y tareas deben ejecutarse prefijados por `uv run`. Esto asegura que se usen exactamente las versiones definidas en el `pyproject.toml` y el archivo de bloqueo (`uv.lock` si existe).
3.  **Revisión de Lógica Existente**: Antes de implementar soluciónes "from scratch" o funciones utilitarias nuevas, es OBLIGATORIO revisar la implementación interna del repositorio (especialmente `detection/common.py`). Muchas veces la funcionalidad requerida (ej. auto-rotación, slices, métricas) ya existe y está probada. Reinventar la rueda introduce bugs y deuda técnica innecesaria.

### Comandos Frecuentes
- **Ejecutar Script de Prueba**:
  ```bash
  uv run python colour_checker_detection/test.py
  ```
- **Instalar/Sincronizar Dependencias**:
  ```bash
  uv sync --all-extras --dev
  ```
- **Ejecutar Tareas de Mantenimiento**:
  ```bash
  uv run invoke preflight
  ```


El uso de `uv` proporciona un entorno aislado, rápido y determinista, crucial para un proyecto científico donde las versiones menores de librerías numéricas pueden alterar los resultados de precisión.


---

## 14. Refinamiento de Orientación y Geometría

### Problema: Espejado (Mirroring) e Inversión 180°
Se observaron dos problemas persistentes en la extracción de swatches:
1.  **Espejado (Mirroring)**: El método de Segmentación a menudo devuelve contornos con orden de vértices inconsistente, causando que la imagen extraída esté espejada (izquierda/derecha invertida).
2.  **Inversión 180°**: Incluso sin espejado, la carta aparecía rotada 180° (Negro en Top-Left en lugar de Dark Skin).

### Solución 1: Normalización de Vértices (`order_points`)
Para prevenir el espejado, se implementó una función `order_points` que fuerza un orden geométrico estricto de los vértices del cuadrilátero detectado:
-   Índice 0: Top-Left (TL)
-   Índice 1: Top-Right (TR)
-   Índice 2: Bottom-Right (BR)
-   Índice 3: Bottom-Left (BL)

Esto asegura que la transformación de perspectiva (`warpPerspective`) siempre mapee consistentemente la geometría detectada al rectángulo canónico.

### Solución 2: Rectángulo Canónico Coherente
Se detectó un desajuste crítico entre el orden de `order_points` y la definición del `rectangle` objetivo.
-   **Incorrecto (Previo)**: `[TL, BL, BR, TR]` (Orden tipo ciclo).
-   **Correcto**: `[TL, TR, BR, BL]` (Orden de lectura Z/Raster).
Corregir el rectángulo objetivo es tan vital como ordenar los puntos de entrada para evitar torsiones en la imagen.

### Solución 3: Auto-Orientación en Lineal (Linear Reference Values)
Aunque inicialmente se pensó desactivar la auto-rotación en el paso lineal, esto causó que métodos sensibles a rotación (como Plantillas/Templated) fallaran, mostrando resultados rotados -90º o 180º.
Para habilitar la auto-orientación correcta en el espacio lineal (oscuro):
1.  Tomar los valores de referencia sRGB estándar (0..1).
2.  Aplicar Gamma Inversa aproximada (`pow(val, 2.2)`) para llevarlos al dominio lineal.
3.  **IMPORTANTE**: Aplicar `np.clip(val, 0, 1)` antes de la potencia, ya que los valores de referencia pueden contener micro-negativos (ruido numérico o fuera de gamut) que generan `NaN` al elevar a potencia fraccionaria. `NaN` en referencias rompe silenciosamente la auto-rotación.

Con estos ajustes, `sample_colour_checker` puede minimizar el error MSE incluso con datos oscuros y orientar correctamente la carta (Dark Skin @ Index 0).

---

## 15. Script de Desarrollo Rápido (`detection_swatches.py`)

### Propósito
Para iteraciones de debug más ágiles, se creó `colour_checker_detection/detection_swatches.py`. Este script:
-   **Procesa UNA sola imagen** (la primera `.CR2`/`.ARW` en `local_test/`).
-   **Usa SOLO el método Templated** (el más robusto para orientación).
-   **Genera un diagnóstico visual** en `test_results/[TIMESTAMP]/debug_detection.png`.

### Flujo de Trabajo
1.  **Lectura Dual**:
    -   `sRGB` (gamma corregida, brillo ajustado) para detección visual.
    -   `Linear` (gamma=1.0, 16-bit, Camera Raw Space) para extracción radiométrica.
2.  **Detección en sRGB**: Usa `detect_colour_checkers_templated`.
3.  **Orientación Robusta**:
    -   Se llama a `sample_colour_checker` con la imagen sRGB y los `reference_values` para obtener el `quadrilateral` optimizado (MSE minimizado).
4.  **Extracción en Linear**:
    -   Se llama a `sample_colour_checker` con la imagen Linear y `reference_values=None` (desactivando re-rotación interna), usando el quad ya optimizado del paso anterior.
5.  **Visualización**:
    -   **Panel Izquierdo**: Imagen sRGB con quad dibujado y **índices de swatch (0-23)** proyectados.
    -   **Panel Derecho**: Cuadrícula 4x6 reconstruida con valores lineales (gamma-corregidos para visualización) y etiquetas de índice.

### Debug Visual
Las etiquetas numéricas (0-23) permiten verificar inmediatamente:
-   **Índice 0** debe coincidir con el parche **Piel Oscura** (marrón, esquina superior-izquierda de la carta física).
-   **Índice 18** debe ser el más brillante (**Blanco**).
-   **Índice 23** debe ser el más oscuro (**Negro**).

Si los índices no coinciden con la posición física esperada, indica un problema de orientación en la detección o en la definición del rectángulo canónico.

### Extracción 16-Bit
La lectura lineal usa `output_bps=16` y normalización `/65535.0` para maximizar la precisión en valores muy oscuros (las sombras en modo lineal sin gamma son cercanas a cero). Esto evita pérdida de información por cuantización de 8-bit.

---

## 16. Reglas de Índices de Swatches (ColorChecker Classic)

| Índice | Nombre Técnico    | Color Esperado       |
|--------|-------------------|----------------------|
| 0      | Dark Skin         | Marrón Oscuro        |
| 1      | Light Skin        | Beige                |
| ...    | ...               | ...                  |
| 18     | White             | Blanco (más brillante) |
| 19     | Neutral 8         | Gris Claro           |
| 20     | Neutral 6.5       | Gris Medio           |
| 21     | Neutral 5         | Gris Medio-Oscuro    |
| 22     | Neutral 3.5       | Gris Oscuro          |
| 23     | Black             | Negro (más oscuro)   |

El orden sigue el patrón de lectura occidental (izquierda-derecha, arriba-abajo) cuando la carta está en orientación horizontal estándar (Logo X-Rite arriba).
