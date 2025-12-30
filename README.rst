ColorChecker Pipeline
=====================

**colorchecker-pipeline** es una implementación avanzada para la integración de corrección de color en pipelines de fotogrametría, derivado del proyecto ``colour-checker-detection``. Implementa detección automática, extracción y corrección colorimétrica de cartas **ColorChecker** (optimizado para **ColorChecker Passport post-2014**) a partir de imágenes **RAW**.

Mantenido por el **Laboratorio de Arqueología Digital UC**.

.. image:: https://raw.githubusercontent.com/colour-science/colour-checker-detection/master/docs/_static/ColourCheckerDetection_001.png
    :alt: Colour Checker Detection
    :align: center

Descripción del Proyecto
------------------------

El objetivo principal es proporcionar un flujo de trabajo (pipeline) científico que garantice la trazabilidad del color desde el sensor de la cámara hasta un espacio de color estándar (**AdobeRGB Linear**) con iluminante **D65**. El sistema automatiza la localización de la carta utilizando tanto técnicas clásicas como **Deep Learning (YOLOv8)**, optimiza la orientación de los parches y calcula la **Matriz de Corrección de Color (CCM)**.

Características Principales
---------------------------

*   **Detección Híbrida**: 
    *   **Segmentación Clásica**: Algoritmos de OpenCV y matching por plantillas.
    *   **Deep Learning (Nuevo)**: Inferencia robusta utilizando modelos **YOLOv8** (vía `ultralytics`) para condiciones difíciles.
*   **Extracción de 16-bits (Camera Space)**: Lectura directa de datos radiométricos lineales usando ``rawpy``, evitando procesamientos gamma intermedios.
*   **Corrección CCM (Cheung 2004)**: Cálculo de matrices de transformación colorimétrica precisas utilizando la librería ``colour-science``.
*   **Normalización de Punto Blanco**: Algoritmo propio para asegurar que las referencias y el resultado final sean perfectamente neutros (R=G=B) en parches grises, eliminando tintes verdosos.
*   **Visualización Técnica de 6 Paneles**: Generación automática de reportes visuales detallados.

Dependencias e Instalación
--------------------------

El proyecto utiliza ``uv`` como gestor de entorno moderno de Python para garantizar velocidad y reproducibilidad.

Sincronización del entorno::

    uv sync

Esto instalará automáticamente todas las dependencias definidas en ``pyproject.toml``, incluyendo:
*   ``colour-science``
*   ``rawpy``
*   ``opencv-python``
*   ``ultralytics`` (para inferencia YOLO)
*   ``matplotlib``

Scripts Principales
-------------------

El repositorio incluye herramientas para diferentes etapas del procesado, desde pruebas iniciales hasta el pipeline completo de corrección:

1.  **test.py**
    *   **Función**: Punto de entrada inicial para testing. Realiza pruebas de detección con diversos métodos (Classic, Templated, Inference) para "lograr la detección".
    *   **Uso**:
        .. code-block:: bash

            uv run python colour_checker_detection/test.py

2.  **colour_checker_detection/detection_swatches.py**
    *   **Función**: Se encarga de la detección *correcta* y precisa de los parches (swatches) individuales, validando la geometría del ColorChecker antes de proceder.
    *   **Uso**:
        .. code-block:: bash

            uv run python colour_checker_detection/detection_swatches.py

3.  **colour_checker_detection/correction_swatches.py**
    *   **Función**: Orquestador final. Realiza la corrección de color completa utilizando la información de los pasos anteriores (Detección -> Extracción -> CCM -> Reporte).
    *   **Uso**:
        .. code-block:: bash

            uv run python colour_checker_detection/correction_swatches.py

4.  **colour_checker_detection/scripts/inference.py**
    *   **Función**: Herramienta de soporte para inferencia pura con YOLOv8.
    *   **Uso**:
        .. code-block:: bash

            uv run python colour_checker_detection/scripts/inference.py --input imagen.jpg --show

Herramientas de Mantenimiento
-----------------------------

El archivo ``tasks.py`` facilita tareas comunes administrativas mediante ``invoke``:

*   **Limpiar temporales y caché**:
    .. code-block:: bash
    
        uv run inv clean

*   **Exportar requirements.txt**:
    .. code-block:: bash
    
        uv run inv requirements

Salidas Esperadas (Outputs)
---------------------------

Los resultados se generan en ``test_results/[TIMESTAMP]/``. El script principal ``correction_swatches.py`` produce una imagen compuesta de 6 paneles para auditoría técnica:

*   **Panel A (Detección)**: Imagen original sRGB con la carta detectada (BBox) y los puntos de muestreo reproyectados.
*   **Panel B (Medido)**: Visualización de los parches extraídos directamente del RAW lineal (con gamma aplicada solo para visualización), indexados del 0 al 23.
*   **Panel C (Corregido)**: Los mismos parches después de aplicar la Matriz de Corrección de Color (CCM) calculada, transformados al espacio destino (AdobeRGB).
*   **Panel D (Referencia)**: Valores teóricos ideales de la carta ColorChecker (Post-2014) en AdobeRGB, ajustados al iluminante D65.
*   **Panel E (Error)**: Gráfico de barras mostrando el error Delta E 2000 para cada parche individual, junto con el promedio (Avg) y máximo (Max).
*   **Panel F (Resultado Final)**: Previsualización de la imagen completa corregida colorimétricamente.

Licencia
--------

Licencia
--------

Este proyecto opera bajo un modelo de licenciamiento mixto para respetar el trabajo original y gestionar las nuevas contribuciones:

1.  **Core del Proyecto (Upstream)**:
    *   **Licencia**: BSD-3-Clause.
    *   **Archivos**: Todo el código base original (`colour_checker_detection/*`), excepto lo indicado abajo.
    *   **Detalles**: Ver archivo ``LICENSE``.

2.  **Scripts de Inferencia (YOLOv8)**:
    *   **Licencia**: GNU AGPLv3.
    *   **Archivos**: Módulos que importan ``ultralytics`` (`scripts/inference.py`).
    *   **Detalles**: Requerido por la licencia de la librería Ultralytics.

3.  **Scripts de Corrección Personalizados**:
    *   **Licencia**: Apache 2.0.
    *   **Archivos**: ``colour_checker_detection/correction_swatches.py`` y sus modificaciones asociadas.
    *   **Detalles**: Ver archivo ``LICENSE_APACHE``.
