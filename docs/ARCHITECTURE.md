# ARQUITECTURA DE SOFTWARE - ColorChecker Pipeline

## 📐 Visión General

**colorchecker-pipeline** es una librería Python para detección y corrección de color en workflows de fotogrametría, optimizada para ColorChecker Passport post-2014.

### Objetivos de Diseño

1. **Modularidad**: Componentes independientes y reutilizables
2. **Aislamiento de licencias**: Código AGPL separado del núcleo Apache 2.0
3. **Extensibilidad**: Fácil añadir nuevos métodos de detección
4. **Performance**: Procesamiento eficiente de imágenes de alta resolución

---

## 🏗️ Arquitectura de Alto Nivel

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE USUARIO                          │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  CLI Scripts    │  │  Python API     │                  │
│  └────────┬────────┘  └────────┬────────┘                  │
└───────────┼─────────────────────┼──────────────────────────┘
            │                     │
            v                     v
┌─────────────────────────────────────────────────────────────┐
│                  CAPA DE APLICACIÓN                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Workflows Completos                                 │  │
│  │  • correction_template.py                            │  │
│  │  • correction_swatches.py                            │  │
│  │  • correction_swatches_benchmark.py                  │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────────┐
│                   CAPA DE DOMINIO                           │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │    Detection     │  │  Correction  │  │    Utils     │ │
│  │  ┌────────────┐  │  │              │  │              │ │
│  │  │ Templated  │  │  │  CCM Calc    │  │  Geometry    │ │
│  │  ├────────────┤  │  │  White Bal   │  │  Color Ops   │ │
│  │  │Segmentation│  │  │  Transform   │  │  I/O         │ │
│  │  ├────────────┤  │  └──────────────┘  └──────────────┘ │
│  │  │ Inference  │◄─┼──► AGPL Isolation                   │
│  │  └────────────┘  │                                      │
│  └──────────────────┘                                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────────┐
│                 CAPA DE INFRAESTRUCTURA                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  NumPy   │  │  OpenCV  │  │ Colour   │  │  RawPy   │  │
│  │          │  │          │  │ Science  │  │          │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │  Ultralytics (YOLOv8) - AGPL-3.0    │                  │
│  │  Lazy Import / Aislado               │                  │
│  └──────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Estructura de Módulos

### Core Detection Module

```
colour_checker_detection/detection/
│
├── common.py                    # Utilidades compartidas
│   ├── detect_contours()
│   ├── quadrilateralise_contours()
│   ├── sample_colour_checker()
│   └── DataDetectionColourChecker  # Estructura de datos
│
├── segmentation.py              # Método clásico (BSD-3)
│   ├── segmenter_default()
│   ├── extractor_segmentation()
│   └── detect_colour_checkers_segmentation()
│
├── templated.py                 # Método robusto (Apache 2.0)
│   ├── segmenter_templated()
│   ├── extractor_templated()
│   ├── detect_colour_checkers_templated()  ← MÉTODO PRINCIPAL
│   └── WarpingData
│
├── inference.py                 # Deep Learning (AGPL-3.0)
│   ├── inferencer_default()     # YOLOv8
│   ├── extractor_inference()
│   └── detect_colour_checkers_inference()
│
├── plotting.py                  # Visualización
│   └── plot_detection_results()
│
└── templates/                   # Plantillas de referencia
    ├── Template
    ├── generate_template()
    └── load_template()
```

### Isolation Pattern (AGPL)

**Problema**: `ultralytics` (AGPL-3.0) contaminaría todo el proyecto

**Solución**: Lazy Import Pattern

```python
# __init__.py
if TYPE_CHECKING:
    # Solo para type checking, no en runtime
    from .inference import detect_colour_checkers_inference

def __getattr__(name: str):
    """Lazy import de módulo AGPL solo si se solicita."""
    if name == "detect_colour_checkers_inference":
        import importlib
        return getattr(
            importlib.import_module(".inference", __package__), 
            name
        )
    raise AttributeError(f"module has no attribute '{name}'")
```

**Beneficio**:
- ✅ `detect_colour_checkers_templated()` → Apache 2.0
- ⚠️ `detect_colour_checkers_inference()` → AGPL-3.0
- ✅ Usuarios comerciales pueden usar librería evitando AGPL

---

## 🔄 Flujo de Datos

### Pipeline de Detección (Templated)

```
┌─────────────────┐
│  Imagen RAW     │
│  (Camera Space) │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  rawpy.imread   │
│  Linear RGB     │
│  16-bit         │
└────────┬────────┘
         │
         v
┌──────────────────────────────────────┐
│  detect_colour_checkers_templated()  │
│  ┌────────────────────────────────┐  │
│  │  1. Reformatear imagen         │  │
│  │  2. Cargar template            │  │
│  │  3. Template matching          │  │
│  │  4. Refinar geometría          │  │
│  │  5. Extraer swatches           │  │
│  └────────────────────────────────┘  │
└──────────────┬───────────────────────┘
               │
               v
┌──────────────────────────────┐
│  DataDetectionColourChecker  │
│  • Corners (4x2)             │
│  • Quadrilateral (4x2)       │
│  • Swatch colors (24x3)      │
│  • Warping data              │
└──────────────┬───────────────┘
               │
               v
┌────────────────────┐
│  CCM Calculation   │
│  (Cheung 2004)     │
└────────┬───────────┘
         │
         v
┌────────────────────┐
│  Color Correction  │
│  RGB → AdobeRGB    │
│  D65 Illuminant    │
└────────┬───────────┘
         │
         v
┌────────────────────┐
│  Output Image      │
│  Corrected         │
└────────────────────┘
```

### Jerarquía de Detección

```
┌───────────────────────────────────────┐
│         ESTRATEGIA DE DETECCIÓN       │
└───────────────────┬───────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        v           v           v
┌────────────┐ ┌─────────┐ ┌──────────┐
│ Templated  │ │Segment- │ │Inference │
│  (Robust)  │ │ation    │ │  (AGPL)  │
│            │ │(Classic)│ │          │
└──────┬─────┘ └────┬────┘ └─────┬────┘
       │            │            │
       v            v            v
┌─────────────────────────────────────┐
│     Extractor Common Interface      │
│  • sample_colour_checker()          │
│  • swatch_masks()                   │
│  • swatch_colours()                 │
└─────────────────────────────────────┘
```

---

## 🎯 Patrones de Diseño

### 1. Strategy Pattern (Métodos de Detección)

```python
# Interface común
def detect_colour_checkers(
    image: NDArray,
    method: Literal["templated", "segmentation", "inference"]
) -> tuple[DataDetectionColourChecker, ...]:
    
    strategies = {
        "templated": detect_colour_checkers_templated,
        "segmentation": detect_colour_checkers_segmentation,
        "inference": detect_colour_checkers_inference,
    }
    
    return strategies[method](image)
```

**Ventaja**: Fácil añadir nuevos métodos sin modificar código existente

### 2. Data Class Pattern

```python
@dataclass
class DataDetectionColourChecker:
    """Encapsula resultado de detección."""
    colour_checker: NDArray[np.float32]
    quadrilateral: NDArray[np.float32]
    # ... otros campos
```

**Ventaja**: 
- Inmutable (con `frozen=True`)
- Type-safe
- Fácil serialización

### 3. Lazy Loading Pattern (Licencias)

```python
# No se importa AGPL hasta que se use
def __getattr__(name):
    if name in AGPL_MODULES:
        return lazy_import(name)
    raise AttributeError
```

**Ventaja**: Aislamiento de dependencias conflictivas

### 4. Template Method Pattern

```python
def detection_pipeline(image):
    # Esqueleto del algoritmo
    formatted = reformat_image(image)      # Paso 1
    detected = _detect(formatted)          # Paso 2 (varía)
    validated = _validate(detected)        # Paso 3
    extracted = _extract_swatches(validated)  # Paso 4
    return extracted
```

### 5. Factory Pattern (Templates)

```python
def load_template(
    template_type: Literal["classic", "nano", "sg"]
) -> Template:
    """Carga template apropiado según tipo."""
    templates = {
        "classic": PATH_TEMPLATE_COLORCHECKER_CLASSIC,
        "nano": PATH_TEMPLATE_COLORCHECKER_NANO,
        "sg": PATH_TEMPLATE_COLORCHECKER_SG,
    }
    return Template.from_file(templates[template_type])
```

---

## 🔐 Gestión de Dependencias

### Capas de Licenciamiento

```
┌─────────────────────────────────────────┐
│  APLICACIÓN USUARIO                     │
│  (Cualquier licencia)                   │
└──────────────┬──────────────────────────┘
               │
               v
┌─────────────────────────────────────────┐
│  CORE LIBRARY                           │
│  License: Apache 2.0 / BSD-3-Clause     │
│  ┌─────────────────────────────────┐   │
│  │  • templated.py                 │   │
│  │  • segmentation.py              │   │
│  │  • correction_template.py       │   │
│  │  • utilities/                   │   │
│  └─────────────────────────────────┘   │
└──────────────┬──────────────────────────┘
               │
               v
┌─────────────────────────────────────────┐
│  DEPENDENCIAS PERMISIVAS                │
│  • NumPy (BSD)                          │
│  • OpenCV (Apache 2.0)                  │
│  • Colour Science (BSD-3)               │
│  • RawPy (MIT)                          │
└──────────────┬──────────────────────────┘
               │
               └──────────────────┐
                                  │
                                  v
┌─────────────────────────────────────────┐
│  MÓDULO OPCIONAL AISLADO                │
│  License: AGPL-3.0                      │
│  ┌─────────────────────────────────┐   │
│  │  • inference.py                 │   │
│  │  • scripts/inference.py         │   │
│  └─────────────────────────────────┘   │
└──────────────┬──────────────────────────┘
               │
               v
┌─────────────────────────────────────────┐
│  ULTRALYTICS                            │
│  License: AGPL-3.0                      │
│  (YOLOv8)                               │
└─────────────────────────────────────────┘
```

### Dependencias por Características

```toml
[project]
dependencies = [
    "colour-science>=0.4.5",      # Core
    "numpy>=2.0.0",
    "opencv-python>=4",
    "rawpy>=0.25.1",
]

[project.optional-dependencies]
# Feature: Deep Learning (AGPL warning)
ultralytics = [
    "ultralytics>=8",
]

# Feature: Docs
docs = [
    "sphinx",
    "pydata-sphinx-theme",
]
```

---

## 🧪 Arquitectura de Testing

```
tests/
├── unit/                        # Tests rápidos, aislados
│   ├── test_common.py
│   ├── test_templated.py
│   └── test_utils_geom.py
│
├── integration/                 # Tests de flujo completo
│   ├── test_detection_pipeline.py
│   └── test_correction_workflow.py
│
├── performance/                 # Benchmarks
│   └── test_detection_speed.py
│
├── fixtures/                    # Datos compartidos
│   └── conftest.py
│
└── data/                        # Imágenes de prueba
    ├── colorchecker_classic.jpg
    └── colorchecker_passport.dng
```

### Estrategia de Testing

1. **Unit Tests** (>80% coverage)
   - Funciones puras
   - Cálculos matemáticos
   - Transformaciones geométricas

2. **Integration Tests**
   - Pipeline completo end-to-end
   - Compatibilidad entre módulos

3. **Property Tests** (Hypothesis)
   - Invariantes matemáticas
   - Robustez ante inputs aleatorios

4. **Performance Tests** (pytest-benchmark)
   - Velocidad de detección
   - Uso de memoria

---

## 📊 Decisiones Arquitectónicas (ADRs)

### ADR-001: Usar Plantillas vs Solo Deep Learning

**Contexto**: YOLOv8 es robusto pero introduce AGPL.

**Decisión**: Implementar método de plantillas como principal.

**Consecuencias**:
- ✅ Sin dependencias AGPL en núcleo
- ✅ Más rápido en condiciones controladas
- ⚠️ Menos robusto con iluminación extrema

### ADR-002: Lazy Import para Inference

**Contexto**: No podemos incluir AGPL directamente.

**Decisión**: Lazy import con `__getattr__`.

**Consecuencias**:
- ✅ Core library Apache 2.0
- ✅ Usuarios comerciales pueden usarla
- ⚠️ Complejidad en imports

### ADR-003: NumPy 2.0+

**Contexto**: NumPy 2.0 rompe compatibilidad pero mejora performance.

**Decisión**: Requerir NumPy >=2.0.0.

**Consecuencias**:
- ✅ +30% velocidad en operaciones matriciales
- ⚠️ Incompatible con ecosistema antiguo

### ADR-004: 16-bit Internal Processing

**Contexto**: Cámaras científicas producen >8-bit.

**Decisión**: Procesar internamente en 16-bit, output configurable.

**Consecuencias**:
- ✅ Sin pérdida de información
- ✅ Mayor precisión colorimétrica
- ⚠️ 2x memoria

---

## 🔮 Evolución Futura

### Roadmap v1.0

```
v0.2.x (Actual)
├── Core detection funcional
├── CCM calculation
└── Templates para Classic/Passport

v0.3.0 (Q1 2025)
├── Batch processing API
├── Progress tracking
└── ColorChecker Nano support

v0.4.0 (Q2 2025)
├── Multiband sensor support
├── Custom illuminants
└── Advanced white balance

v1.0.0 (Q3 2025)
├── API stable
├── Production ready
└── Full documentation
```

### Extensiones Planificadas

1. **Plugin System**
   ```python
   # Usuarios pueden registrar nuevos métodos
   register_detection_method("custom", my_detector)
   ```

2. **Cloud Processing**
   ```python
   # Procesar en cloud para grandes volúmenes
   results = process_batch_cloud(images, api_key=...)
   ```

3. **Integración Nativa con Metashape**
   ```python
   # Plugin directo para Agisoft Metashape
   from colorchecker import MetashapePlugin
   ```

---

## 📚 Recursos de Arquitectura

### Documentos de Referencia

- [C4 Model](https://c4model.com/) - Para diagramas de arquitectura
- [Architectural Decision Records](https://adr.github.io/)
- [The Twelve-Factor App](https://12factor.net/) - Para apps modulares

### Herramientas de Visualización

```bash
# Generar diagrama de dependencias
uv run pydeps colour_checker_detection --max-depth 3

# Análisis de complejidad
uv run radon cc colour_checker_detection/ -a

# Análisis de acoplamiento
uv run cohesion -d colour_checker_detection/
```

---

## 🤝 Contribuir a la Arquitectura

Si propones cambios arquitectónicos:

1. **Crea un ADR** (Architecture Decision Record)
2. **Discute en GitHub Discussions** antes de implementar
3. **Actualiza este documento** con cambios aprobados
4. **Mantén diagramas sincronizados** con código

---

**Última actualización**: 2026-01-01  
**Versión**: 0.3.0  
**Autor**: Laboratorio de Arqueología Digital UC