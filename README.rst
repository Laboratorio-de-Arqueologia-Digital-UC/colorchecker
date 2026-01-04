ColorChecker Pipeline
=====================
.. image:: https://github.com/Laboratorio-de-Arqueologia-Digital-UC/colorchecker/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/Laboratorio-de-Arqueologia-Digital-UC/colorchecker/actions/workflows/ci.yml
    :alt: CI Status

.. image:: https://codecov.io/gh/Laboratorio-de-Arqueologia-Digital-UC/colorchecker/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/Laboratorio-de-Arqueologia-Digital-UC/colorchecker
    :alt: Coverage

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://opensource.org/licenses/Apache-2.0
    :alt: License

.. image:: https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue
    :alt: Python versions

.. image:: https://img.shields.io/github/issues/Laboratorio-de-Arqueologia-Digital-UC/colorchecker
    :target: https://github.com/Laboratorio-de-Arqueologia-Digital-UC/colorchecker/issues
    :alt: Issues

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

**colorchecker-pipeline** es una implementación avanzada para la integración de corrección de color en pipelines de fotogrametría, derivado del proyecto ``colour-checker-detection``. Implementa detección automática, extracción y corrección colorimétrica de cartas **ColorChecker** (optimizado para **ColorChecker Passport post-2014**) a partir de imágenes **RAW**.

Mantenido por el **Laboratorio de Arqueología Digital UC**.

.. image:: https://github.com/Laboratorio-de-Arqueologia-Digital-UC/colorchecker/blob/c73f5f6784c6ac98530a9a526bd5ab79c0c43713/docs/_static/correction_Templated_color_checker_1.png
    :alt: ColorChecker Pipeline
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
*   **Exportación Multi-Formato**: Generación automática de perfiles para integración profesional:
    *   **DCP** (Digital Camera Profile) para Adobe Lightroom/Camera Raw (vía **dcpTool**).
    *   **3D LUT** (.cube) para DaVinci Resolve/Premiere.
    *   **CCM** (Texto) y JSON para análisis científico.
*   **Normalización de Punto Blanco**: Algoritmo propio para asegurar que las referencias y el resultado final sean perfectamente neutros (R=G=B) en parches grises, eliminando tintes verdosos.
*   **Visualización Técnica de 6 Paneles**: Generación automática de reportes visuales detallados.

Créditos y Base
---------------

Este proyecto se basa fuertemente en el excelente trabajo de **Colour Developers** y su librería `colour-checker-detection`.
*   **Proyecto Original**: `https://github.com/colour-science/colour-checker-detection`
*   Reconocemos y agradecemos su contribución fundamental al campo de la ciencia del color open source. 
*   Nuestra implementación extiende su base para soportar flujos de trabajo específicos de Arqueología Digital, imágenes RAW de alta resolución y nuevos formatos de exportación.

Herramientas Externas y Licencias
---------------------------------

*   **dcpTool**: Se utiliza el binario de `dcpTool` (ubicado en `external/dcptool`) para la generación de archivos .dcp.
    *   **Licencia**: GNU General Public License (GPL).
    *   Web: `http://dcptool.sourceforge.net/`
    *   Nota: `dcpTool` se ejecuta como un proceso externo y no se vincula estáticamente con el código Python del proyecto.

Dependencias e Instalación
--------------------------

El proyecto utiliza ``uv`` como gestor de entorno moderno de Python para garantizar velocidad y reproducibilidad.

Sincronización del entorno::

    uv sync

Esto instalará automáticamente todas las dependencias definidas en ``pyproject.toml``, incluyendo:
*   ``colour-science``
*   ``rawpy``
*   ``opencv-python``
*   ``ultralytics`` (opcional, para inferencia YOLO)
*   ``matplotlib``

Ejecutar Tests
--------------

Para validar todo el código, incluyendo los tests unitarios recién implementados que cubren detección, corrección y benchmarks:

.. code-block:: bash

    uv run pytest colour_checker_detection/tests/

**Nota sobre Tests de Benchmark**:
Los tests completos (que procesan imágenes RAW reales) requieren que las imágenes de prueba ("local_test") estén presentes. En CI/CD, esto se maneja vía Git LFS. Si ejecutas localmente sin estas imágenes, los tests se saltarán automáticamente (SKIPPED).

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

5.  **colour_checker_detection/correction_swatches_benchmark.py**
    *   **Función**: Herramienta de evaluación estadística. Compara científicamente los métodos de detección **Segmentation** vs **Templated**.
    *   **Métricas y Tablas Generadas**:

        *   **Valores RGB**: Reporte detallado de valores RGB lineales extraídos por cada método y su precisión ($\Delta E_{2000}$) respecto a la referencia teórica (D65).
        *   **Tiempo (Time)**: Comparativa de tiempos de ejecución por imagen.
        *   **Comparación por Imagen**: Resumen de **Deriva Geométrica** (distancia en píxeles entre centros detectados) y **Diferencia de Color Promedio** entre métodos.
        *   **Significancia Estadística (Paired T-Test)**: Prueba de hipótesis para determinar si los métodos producen resultados colorimétricamente equivalentes ($p < 0.05$).

    *   **Uso**:

        .. code-block:: bash

            uv run python colour_checker_detection/correction_swatches_benchmark.py

6.  **colour_checker_detection/correction_template.py**
    *   **Función**: Script de producción para extracción de datos. Utiliza **exclusivamente** el método de detección por **Plantillas (Templated)** para máxima robustez geométrica.
    *   **Salidas**:

        *   Imagen de visualización (PNG) con 6 paneles.
        *   **Reporte JSON**: Archivo estructurado conteniendo coordenadas de píxeles, valores RGB detectados (Lineal), valores corregidos (AdobeRGB) y referencias teóricas para cada parche.

    *   **Uso**:

        .. code-block:: bash

            uv run python colour_checker_detection/correction_template.py


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

Arquitectura del Software
-------------------------

Visión General
~~~~~~~~~~~~~~

El proyecto sigue una arquitectura de capas:

.. code-block:: text

    ┌─────────────────────────────────────┐
    │     Scripts de Usuario (CLI)       │
    └─────────────┬───────────────────────┘
                  │
    ┌─────────────▼───────────────────────┐
    │  Workflows de Aplicación            │
    │  • correction_template.py           │
    │  • correction_swatches.py           │
    └─────────────┬───────────────────────┘
                  │
    ┌─────────────▼───────────────────────┐
    │  Capa de Dominio                    │
    │  ┌──────────┐  ┌─────────────────┐ │
    │  │Detection │  │  Correction     │ │
    │  │• Templated  │  • CCM Calc     │ │
    │  │• Segment.│  │  • White Bal    │ │
    │  └──────────┘  └─────────────────┘ │
    └─────────────┬───────────────────────┘
                  │
    ┌─────────────▼───────────────────────┐
    │  Infraestructura                    │
    │  NumPy • OpenCV • Colour Science    │
    └─────────────────────────────────────┘

Módulos Principales
~~~~~~~~~~~~~~~~~~~

**Detection (``colour_checker_detection/detection/``)**

* ``templated.py``: Detección robusta por plantillas (Apache 2.0)
* ``segmentation.py``: Método clásico de segmentación (BSD-3)
* ``inference.py``: Deep Learning con YOLOv8 (AGPL-3.0, opcional)

**Utilities (``colour_checker_detection/utilities/``)**

* Operaciones geométricas
* Transformaciones de color
* I/O helpers

Flujo de Datos
~~~~~~~~~~~~~~

1. **Entrada**: Imagen RAW → rawpy → RGB lineal 16-bit
2. **Detección**: Template matching → Geometría refinada
3. **Extracción**: Muestra de swatches (24x3 RGB)
4. **Corrección**: Cálculo CCM → Aplicación → AdobeRGB D65
5. **Salida**: Imagen corregida + Reporte JSON

Para documentación completa de arquitectura, ver ``docs/ARCHITECTURE.md``.

Licencia y Cumplimiento
-----------------------

Este proyecto utiliza un modelo de **Licenciamiento Dual/Aislado** para maximizar la compatibilidad comercial sin violar los términos de las dependencias.

1.  **Código Base (Apache 2.0 / BSD-3-Clause)**:
    El núcleo del proyecto, incluyendo `correction_template.py` y los algoritmos de segmentación, es puramente **Apache 2.0**. No contiene ni importa código AGPL por defecto.

2.  **Módulo de Inferencia (AGPL-3.0 - Opcional)**:
    La funcionalidad de YOLOv8 (`ultralytics`) está aislada en el módulo `inference` y solo se carga si se usa explícitamente `detect_colour_checkers_inference` o el script de inferencia.
    
    .. warning::
        Al activar la inferencia YOLO, el proceso en ejecución queda sujeto a la licencia **AGPL-3.0**.

Consulte el reporte detallado en `LICENSE_COMPLIANCE.md` para más detalles sobre cómo mantener su pipeline "limpio".

**Apache License 2.0**

Copyright 2024 Laboratorio de Arqueología Digital UC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
