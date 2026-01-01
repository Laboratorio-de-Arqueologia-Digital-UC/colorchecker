# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/es/1.0.0/)
Versionado: [Semantic Versioning](https://semver.org/lang/es/)

## [Sin Publicar]

### Añadido
- Tests estabilizados
- Commitizen para changelog automático

### Fix
- **CI/CD**: Solucionado error de sintaxis en `README.rst` (indentación) que impedía la validación de metadatos en PyPI.
- **Pyright**: Solucionada discrepancia de configuración local vs CI desactivando `reportUnnecessaryTypeIgnoreComment` para permitir ignores condicionales en importaciones opcionales (`ultralytics`).

## [0.2.3] - 2024-12-31

### Añadido
- Detección por plantillas (templated)
- Benchmark segmentation vs templated
- Normalización punto blanco CCM

### Cambiado
- Migración a uv
- Actualización dependencias

### Seguridad
- Aislamiento AGPL-3.0 con lazy imports
## v0.3.0 (2025-12-31)

### Feat

- Implement a comprehensive pipeline for color checker detection, sampling, correction, and evaluation from RAW images.
- Add initial Sphinx documentation setup with configuration and basic pages.
- Introduce core utilities for colour checker detection, supported by new benchmarks, tests, and documentation.
- Introduce a templated correction script, add a knowledge base, and update the README image.
- Introduce new modules for colour checker detection, correction, and geometric utilities, alongside a comprehensive test and benchmark script.
- Add colour correction swatches and a benchmark for detection, RAW processing, and Delta E evaluation.
- Introduce colour correction swatch processing and a comprehensive knowledge base for colour checker detection.
- Add `correction_swatches.py` for RAW image color checker detection, swatch sampling, and reference preparation, creating test result images and modifying `pyproject.toml`.
- Add multi-method colour checker detection and swatch extraction from RAW images with visualization and documentation.
- Introduce color checker detection and linear swatch extraction from RAW images, featuring orientation optimization, visual debugging, new geometric utilities, and project documentation.
- Add local traceability test and benchmark script for ColorChecker detection in RAW images, comparing inference, segmentation, and templating methods.
- Add local ColorChecker detection traceability test and benchmark script for RAW images.
- Add local traceability test and benchmark script for ColorChecker detection methods on high-resolution RAW images.
- Add `colour-checker-detection` package, including detection algorithms, a knowledge base for implementation insights, and a local traceability test script.

### Fix

- correct CI workflow YAML syntax

### Refactor

- **ci/cd-and-changelog**: add automatic changelog

## v0.2.1 (2024-10-13)

## v0.2.0 (2024-01-11)

## v0.1.6 (2023-12-21)

## v0.1.5 (2023-08-28)

## v0.1.4 (2022-12-04)

## v0.1.3 (2022-02-27)

## v0.1.2 (2020-11-29)

## v0.1.1 (2019-11-04)

## v0.1.0 (2019-03-24)
