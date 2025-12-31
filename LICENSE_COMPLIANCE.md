# Cumplimiento de Licencias

Este proyecto opera base bajo licencia **Apache-2.0**, derivado de `colour-checker-detection` (BSD-3-Clause).

## ⚠️ ADVERTENCIA IMPORTANTE

La dependencia `ultralytics` (AGPL-3.0) es **opcional** y su código solo se activa al usar la función:
- ❌ `detect_colour_checkers_inference()`
- ❌ O importar el módulo `colour_checker_detection.detection.inference`

El uso de estas funciones conlleva la aplicación de la licencia **AGPL-3.0** a la aplicación en ejecución.

## ✅ Uso seguro (Compatible Apache-2.0 / BSD)

Para mantener el cumplimiento con licencias permisivas y evitar la contaminación viral de la AGPL, utilice solo estas funciones:
- ✅ `detect_colour_checkers_segmentation()` (Método clásico)
- ✅ `detect_colour_checkers_templated()` (Método usado en `correction_template.py`)
- ✅ Cualquier otra función que **NO** involucre inferencia por Deep Learning.

Gracias a la implementación de *Lazy Imports*, el código de inferencia no se carga en memoria a menos que se solicite explícitamente, manteniendo el resto de la librería segura para uso comercial/permisivo.
