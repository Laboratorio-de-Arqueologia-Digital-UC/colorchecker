Pipeline de Calibración de Color Automática
==========================================

Este repositorio contiene una implementación avanzada para la detección, extracción y corrección colorimétrica de cartas **ColorChecker** (específicamente optimizado para **ColorChecker Passport post-2014**) a partir de imágenes en formato **RAW**.

.. image:: https://raw.githubusercontent.com/colour-science/colour-checker-detection/master/docs/_static/ColourCheckerDetection_001.png
    :alt: Colour Checker Detection
    :align: center

Descripción del Proyecto
------------------------

El objetivo principal es proporcionar un flujo de trabajo (pipeline) científico que garantice la trazabilidad del color desde el sensor de la cámara hasta un espacio de color estándar (**AdobeRGB Linear**) con iluminante **D65**. El sistema automatiza la localización de la carta, la orientación de los parches y el cálculo de la **Matriz de Corrección de Color (CCM)**.

Características Principales
---------------------------

*   **Detección Híbrida**: Combina algoritmos de segmentación clásica (OpenCV) y matching por plantillas para máxima robustez en diferentes condiciones de iluminación.
*   **Extracción de 16-bits (Camera Space)**: Lectura directa de datos radiométricos lineales usando ``rawpy``, evitando procesamientos gamma intermedios.
*   **Corrección CCM (Cheung 2004)**: Cálculo de matrices de transformación colorimétrica precisas utilizando la librería ``colour-science``.
*   **Normalización de Punto Blanco**: Algoritmo propio para asegurar que las referencias y el resultado final sean perfectamente neutros (R=G=B) en parches grises, eliminando tintes verdosos.
*   **Visualización Técnica de 6 Paneles**: Generación automática de reportes visuales para inspección de calidad.

Dependencias
------------

El proyecto utiliza ``uv`` como gestor de entorno para garantizar reproducibilidad:

*   **Fundamentales**: ``python >= 3.10``, ``colour-science >= 0.4.3``, ``rawpy``, ``opencv-python``.
*   **Análisis**: ``numpy``, ``scikit-learn`` (para detección por plantillas), ``matplotlib``.

Instalación y Uso
-----------------

Sincronización del entorno::

    uv sync

Ejecución del pipeline de calibración::

    uv run python colour_checker_detection/correction_swatches.py

Scripts Principales
-------------------

1.  **correction_swatches.py**: El orquestador principal. Realiza detección, extracción lineal, cálculo de CCM, normalización y visualización.
2.  **detection_swatches.py**: Versión centrada en la validación geométrica y extracción batch de múltiples formatos RAW (.CR2, .ARW, .RAF).
3.  **test.py**: Herramienta de benchmark para evaluar la precisión de los modelos de detección.

Funciones Críticas
------------------

*   ``detect_colour_checkers_segmentation`` / ``templated``: Funciones de detección que operan en resolución nativa para mantener precisión 1:1.
*   ``sample_colour_checker``: Extrae los valores RGB y optimiza la orientación (rotación 0, 90, 180, 270) basándose en el error MSE contra la referencia.
*   ``colour.colour_correction``: Implementa la transformación lineal de mínimos cuadrados para generar la imagen corregida.
*   ``White Point Normalization (Internal)``: Escala los valores XYZ de referencia para alinearlos exactamente con el iluminante D65.

Salidas Esperadas (Outputs)
---------------------------

El proyecto genera resultados en la carpeta ``test_results/[TIMESTAMP]/``:

*   **Reporte Visual (PNG)**: Imagen técnica de 6 paneles:
    *   **Panel A**: Detección original con índices y BBox.
    *   **Panel B**: Parches medidos (Camera Space) con índices 0-23.
    *   **Panel C**: Parches corregidos en AdobeRGB.
    *   **Panel D**: Referencias teóricas neutralizadas.
    *   **Panel E**: Gráfico de error Delta E 2000.
    *   **Panel F**: Previsualización de la imagen completa corregida.
*   **Métricas**: Logs detallados con Delta E 2000 promedio y máximo (Objetivo: ΔE < 3).

---

| **Desarrollo Centrado en Precisión Colorimétrica**
| Implementado para el Laboratorio de Arqueología Digital UC.
| Basado en el ecosistema `Colour Science <https://www.colour-science.org/>`__.
