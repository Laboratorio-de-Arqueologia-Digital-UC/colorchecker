# Guía de Contribución

¡Gracias por tu interés en contribuir a **colorchecker-pipeline**! Este documento te guiará a través del proceso de contribución.

---

## 📋 Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [¿Cómo Puedo Contribuir?](#cómo-puedo-contribuir)
- [Licenciamiento](#licenciamiento)
- [Configuración del Entorno de Desarrollo](#configuración-del-entorno-de-desarrollo)
- [Workflow de Desarrollo](#workflow-de-desarrollo)
- [Convenciones de Código](#convenciones-de-código)
- [Tests](#tests)
- [Commits y Mensajes](#commits-y-mensajes)
- [Pull Requests](#pull-requests)
- [Documentación](#documentación)
- [Contacto](#contacto)

---

## 📜 Código de Conducta

Este proyecto está mantenido por el **Laboratorio de Arqueología Digital UC** y adherimos a principios de colaboración académica respetuosa y constructiva.

### Nuestros Compromisos

- Mantener un ambiente acogedor e inclusivo
- Respetar diferentes puntos de vista y experiencias
- Aceptar críticas constructivas con gracia
- Enfocarse en lo mejor para la comunidad científica
- Mostrar empatía hacia otros miembros de la comunidad

---

## 🤝 ¿Cómo Puedo Contribuir?

### Reportar Bugs

Si encuentras un bug, por favor:

1. **Busca primero** en [Issues existentes](https://github.com/Laboratorio-de-Arqueologia-Digital-UC/colorchecker/issues) para evitar duplicados
2. Usa el template de **Bug Report** al crear un nuevo issue
3. Incluye:
   - Versión de Python y sistema operativo
   - Pasos para reproducir el problema
   - Comportamiento esperado vs observado
   - Screenshots o logs (si aplica)
   - Imágenes de ejemplo (si es relevante para detección/corrección)

### Proponer Features

Para nuevas funcionalidades:

1. **Discute primero**: Abre un issue con el template **Feature Request**
2. Explica el **caso de uso** científico/arqueológico
3. Considera el impacto en:
   - Trazabilidad del color
   - Precisión colorimétrica
   - Performance en pipelines de fotogrametría
   - Compatibilidad con sensores/cámaras

### Mejorar Documentación

La documentación es crítica para un proyecto científico:

- Correcciones de typos o claridad
- Ejemplos adicionales de uso
- Traducciones (español/inglés)
- Documentación de algoritmos y referencias científicas

---

## ⚖️ Licenciamiento

**Este proyecto usa dual licensing** - es importante entender esto antes de contribuir.

### Estructura de Licencias

```
colorchecker-pipeline/
├── colour_checker_detection/
│   ├── detection/
│   │   ├── templated.py      [Apache 2.0]
│   │   ├── segmentation.py   [BSD-3-Clause]
│   │   ├── inference.py      [AGPL-3.0] ⚠️
│   │   └── common.py         [Apache 2.0]
│   ├── correction_*.py       [Apache 2.0]
│   └── utilities/            [Apache 2.0]
```

### Licencias por Módulo

#### 🟢 **Apache 2.0** (Core del Proyecto)
- **Módulos**: Detección (templated, segmentation), corrección, utilities
- **Tu código debe ser**: Apache 2.0, MIT, BSD, o dominio público
- **Dependencias permitidas**: Cualquier licencia permisiva (MIT, BSD, Apache, PSF)

#### 🔴 **AGPL-3.0** (Módulo de Inferencia - Aislado)
- **Módulos**: `detection/inference.py` (YOLOv8)
- **Tu código debe ser**: AGPL-3.0 compatible
- **Dependencias permitidas**: GPL, AGPL, LGPL

### Certificado de Origen del Desarrollador (DCO)

Al contribuir, certificas que:

```
Developer Certificate of Origin
Version 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

**Firma tus commits** con `-s`:
```bash
git commit -s -m "feat: add new feature"
```

### ⚠️ Dependencias: Restricciones Importantes

#### Prohibido Añadir
- ❌ Código propietario o sin licencia
- ❌ Licencias no comerciales (CC BY-NC, etc.)
- ❌ AGPL en módulos core (solo en `inference.py`)
- ❌ GPL en módulos core (contaminaría Apache 2.0)

#### Permitido en Core (Apache 2.0)
- ✅ MIT, BSD-2, BSD-3
- ✅ Apache 2.0
- ✅ PSF (Python Software Foundation)
- ✅ ISC, Unlicense, Public Domain

#### Permitido en Inference (AGPL-3.0)
- ✅ GPL-3.0, AGPL-3.0, LGPL-3.0
- ✅ Ultralytics (AGPL-3.0)

---

## 💻 Configuración del Entorno de Desarrollo

### Requisitos

- Python 3.11, 3.12 o 3.13
- [uv](https://github.com/astral-sh/uv) (gestor de paquetes)
- Git
- Sistema operativo: Linux, macOS o Windows

### Setup Inicial

```bash
# 1. Fork el repositorio en GitHub

# 2. Clonar tu fork
git clone https://github.com/TU_USUARIO/colorchecker.git
cd colorchecker

# 3. Añadir upstream
git remote add upstream https://github.com/Laboratorio-de-Arqueologia-Digital-UC/colorchecker.git

# 4. Instalar uv (si no lo tienes)
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 5. Instalar dependencias
uv sync --all-extras

# 6. Instalar pre-commit hooks
uv run pre-commit install

# 7. Verificar instalación
uv run pytest
uv run ruff check .
uv run pyright
```

### Estructura del Proyecto

```
colorchecker-pipeline/
├── colour_checker_detection/      # Código fuente principal
│   ├── detection/                 # Algoritmos de detección
│   │   ├── templated.py          # Detección por plantillas (PRINCIPAL)
│   │   ├── segmentation.py       # Detección clásica
│   │   ├── inference.py          # YOLOv8 (AGPL-3.0)
│   │   └── common.py             # Utilities compartidas
│   ├── correction_template.py    # Pipeline de corrección (SCRIPT PRINCIPAL)
│   ├── correction_swatches.py    # Corrección alternativa
│   ├── detection_swatches.py     # Detección de swatches
│   ├── utilities/                # Funciones auxiliares
│   └── tests/                    # Tests del proyecto
│       ├── test_correction_template.py
│       ├── test_correction_swatches.py
│       ├── test_detection_swatches.py
│       └── test_correction_swatches_benchmark.py
├── docs/                         # Documentación Sphinx
├── .github/                      # GitHub Actions, templates
├── pyproject.toml               # Configuración del proyecto
└── CHANGELOG.md                 # Historial de cambios
```

---

## 🔄 Workflow de Desarrollo

### 1. Crear Branch

```bash
# Actualizar main
git checkout main
git pull upstream main

# Crear branch descriptivo
git checkout -b feat/descripcion-breve
# o
git checkout -b fix/descripcion-bug
```

**Convención de nombres de branch**:
- `feat/` - Nueva funcionalidad
- `fix/` - Corrección de bug
- `docs/` - Cambios en documentación
- `refactor/` - Refactorización de código
- `test/` - Añadir o mejorar tests
- `chore/` - Tareas de mantenimiento

### 2. Desarrollar

```bash
# Hacer cambios
# ...

# Verificar calidad (pre-commit se ejecuta automáticamente)
uv run ruff format .
uv run ruff check .
uv run pyright

# Ejecutar tests
uv run pytest

# Verificar coverage
uv run pytest --cov=colour_checker_detection --cov-report=html
```

### 3. Commit

**IMPORTANTE**: Usar commitizen para mantener changelog automático.

```bash
# Añadir cambios
git add .

# Commit usando commitizen
uv run cz commit

# Te preguntará:
# - Type: feat, fix, docs, test, refactor, chore, ci
# - Scope: detection, correction, utilities, docs, ci
# - Subject: descripción corta
# - Body: (opcional) descripción larga
# - Breaking change: (opcional) si rompe API
# - Footer: (opcional) referencias a issues

# Firmar commit (DCO)
git commit --amend -s
```

**Ejemplo de commit message**:
```
feat(detection): add support for ColorChecker Nano

Implements template matching for the smaller 20-patch ColorChecker Nano
variant. Includes specialized geometric validation for the 4x5 layout.

BREAKING CHANGE: Detection settings now require 'variant' parameter.

Fixes #123

Signed-off-by: Tu Nombre <tu.email@example.com>
```

### 4. Push y Pull Request

```bash
# Push a tu fork
git push origin feat/descripcion-breve

# Ir a GitHub y crear Pull Request
```

---

## 📐 Convenciones de Código

### Estilo Python

- **PEP 8** con extensiones de NumPy
- **Line length**: 88 caracteres (Black-style)
- **Imports**: Ordenados automáticamente por Ruff
- **Docstrings**: Formato NumPy

### Docstrings (NumPy Style)

```python
def detect_colour_checker_templated(
    image: NDArray[np.uint8],
    settings: dict[str, Any] | None = None,
) -> tuple[DataDetectionColourChecker, ...]:
    """
    Detecta ColorCheckers en una imagen usando template matching.

    Este método es robusto ante variaciones de iluminación y orientación.
    Optimizado para ColorChecker Passport post-2014.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Imagen RGB en espacio sRGB, valores 0-255.
        Dimensiones esperadas: (H, W, 3).
    settings : dict[str, Any] or None, optional
        Configuración de detección. Si es None, usa valores por defecto.
        Claves válidas:
        - 'template_size': Tamaño de plantilla (default: 800)
        - 'threshold': Umbral de matching (default: 0.7)

    Returns
    -------
    tuple[DataDetectionColourChecker, ...]
        Tupla de detecciones encontradas. Puede estar vacía si no se
        detecta ningún ColorChecker.

    Raises
    ------
    ValueError
        Si la imagen no es RGB o tiene dimensiones inválidas.

    Examples
    --------
    >>> import cv2
    >>> img = cv2.imread('photo.jpg')
    >>> detections = detect_colour_checker_templated(img)
    >>> if detections:
    ...     first_detection = detections[0]
    ...     print(f"Found {len(first_detection.swatch_colours)} swatches")

    Notes
    -----
    El algoritmo usa las siguientes etapas:
    1. Conversión a espacio LAB para invariancia de iluminación
    2. Template matching multi-escala
    3. Refinamiento geométrico con RANSAC
    4. Validación de proporción 6:4 (24 swatches)

    Referencias científicas:
    [1] Cheung et al. (2004) "A Comparative Study of the Characterisation
        of Colour Cameras by Means of Neural Networks and Polynomial
        Transforms", Coloration Technology, 120(1), 19–25.

    See Also
    --------
    detect_colour_checker_segmentation : Método alternativo por segmentación
    detect_colour_checker_inference : Método basado en YOLOv8 (AGPL-3.0)
    """
```

### Type Hints

**Requerido** para:
- Todas las funciones públicas (API)
- Clases y métodos públicos
- Parámetros y return types

**Opcional** para:
- Funciones internas privadas
- Variables locales (excepto si mejora claridad)

```python
from typing import Any
from numpy.typing import NDArray
import numpy as np

# Type aliases para tipos comunes
RGBImage: TypeAlias = NDArray[np.uint8]
LinearRGB: TypeAlias = NDArray[np.float32]

def process_raw_image(
    filepath: Path,
    white_balance: bool = True,
) -> LinearRGB:
    """Procesa imagen RAW a RGB lineal."""
    ...
```

### Nombres de Variables

```python
# BIEN - Descriptivos y específicos al dominio
colour_checker_rgb: NDArray[np.uint8]
ccm_matrix: NDArray[np.float64]
delta_e_values: NDArray[np.float32]
swatch_coordinates: list[tuple[int, int]]

# MAL - Genéricos o abreviaciones crípticas
arr: NDArray
mat: NDArray
vals: list
coords: list
```

---

## 🧪 Tests

### Requisitos de Coverage

- **Mínimo general**: 70%
- **Código crítico** (detección, CCM): >85%
- **Nuevo código**: >80%

### Escribir Tests

**Ubicación**: `colour_checker_detection/tests/`

```python
# tests/test_detection_custom.py

import pytest
import numpy as np
from colour_checker_detection.detection import detect_colour_checkers_templated

class TestDetectionCustom:
    """Tests para nueva funcionalidad de detección."""
    
    def test_nano_variant_detection(self):
        """Detecta ColorChecker Nano (20 patches)."""
        # Arrange
        img = np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8)
        settings = {'variant': 'nano'}
        
        # Act
        result = detect_colour_checkers_templated(img, settings)
        
        # Assert
        assert isinstance(result, tuple)
        # Más aserciones específicas...
    
    @pytest.mark.parametrize("size", [400, 800, 1600])
    def test_multi_scale_detection(self, size):
        """Detección funciona en múltiples escalas."""
        img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        result = detect_colour_checkers_templated(img)
        assert isinstance(result, tuple)
    
    @pytest.mark.skipif(
        not Path("test_data/real_image.jpg").exists(),
        reason="Test image not available"
    )
    def test_with_real_image(self):
        """Test con imagen real de ColorChecker."""
        # Solo se ejecuta si existe la imagen
        ...
```

### Ejecutar Tests

```bash
# Todos los tests
uv run pytest

# Con coverage
uv run pytest --cov=colour_checker_detection --cov-report=html

# Solo tests específicos
uv run pytest tests/test_detection_custom.py

# Por marker
uv run pytest -m "not slow"

# Verbose
uv run pytest -v

# Parallel (más rápido)
uv run pytest -n auto
```

### Tests de Integración

Para tests que requieren imágenes reales:

```python
@pytest.fixture
def sample_colorchecker_image():
    """Fixture con imagen real de ColorChecker."""
    path = Path("tests/data/colorchecker_sample.jpg")
    if not path.exists():
        pytest.skip("Sample image not available")
    return cv2.imread(str(path))

def test_full_pipeline(sample_colorchecker_image):
    """Test del pipeline completo end-to-end."""
    detections = detect_colour_checkers_templated(sample_colorchecker_image)
    assert len(detections) > 0
    assert len(detections[0].swatch_colours) == 24
```

---

## 📝 Commits y Mensajes

### Conventional Commits

Usamos [Conventional Commits](https://www.conventionalcommits.org/) con commitizen.

**Formato**:
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat` - Nueva funcionalidad → MINOR version bump
- `fix` - Corrección de bug → PATCH version bump
- `docs` - Solo documentación
- `test` - Añadir o modificar tests
- `refactor` - Refactorización sin cambio de funcionalidad
- `perf` - Mejora de performance
- `ci` - Cambios en CI/CD
- `chore` - Tareas de mantenimiento
- `build` - Cambios en sistema de build

### Scopes

- `detection` - Algoritmos de detección
- `correction` - Algoritmos de corrección
- `utilities` - Funciones auxiliares
- `docs` - Documentación
- `ci` - CI/CD
- `tests` - Tests

### Breaking Changes

Si introduces un cambio que rompe la API:

```bash
feat(detection)!: change detection API to accept settings object

BREAKING CHANGE: Detection functions now require a Settings object
instead of individual parameters.

Before:
  detect_colour_checkers_templated(img, threshold=0.7)

After:
  settings = DetectionSettings(threshold=0.7)
  detect_colour_checkers_templated(img, settings)

Migration guide: https://docs.example.com/migration-v2

Fixes #456
```

---

## 🔀 Pull Requests

### Antes de Abrir un PR

**Checklist**:
- [ ] Código sigue convenciones del proyecto
- [ ] Tests añadidos y pasando (`uv run pytest`)
- [ ] Coverage mantenido o mejorado
- [ ] Documentación actualizada
- [ ] CHANGELOG.md actualizado (si aplica manualmente)
- [ ] Commits firmados con DCO (`-s`)
- [ ] Pre-commit hooks pasando
- [ ] CI pasando en GitHub Actions

### Template de Pull Request

Al crear un PR, el template te pedirá:

```markdown
## Tipo de Cambio

- [ ] Bug fix (cambio que arregla un issue)
- [ ] Nueva funcionalidad (cambio que añade funcionalidad)
- [ ] Breaking change (fix o feature que rompe funcionalidad existente)
- [ ] Documentación

## Descripción

(Descripción clara y concisa del cambio)

## Motivación y Contexto

(¿Por qué es necesario este cambio? ¿Qué problema resuelve?)

## ¿Cómo se ha probado?

- [ ] Tests unitarios
- [ ] Tests de integración
- [ ] Pruebas manuales (describir)

## Screenshots (si aplica)

(Capturas de pantalla de visualizaciones, resultados, etc.)

## Checklist

- [ ] Mi código sigue las convenciones del proyecto
- [ ] He actualizado la documentación
- [ ] He añadido tests que prueban mi cambio
- [ ] Todos los tests pasan localmente
- [ ] He firmado mis commits (DCO)
- [ ] He considerado las implicaciones de licenciamiento
```

### Proceso de Review

1. **Automated checks**: CI debe pasar (tests, linting, type checking)
2. **Code review**: Al menos 1 aprobación de maintainer
3. **Discusión**: Iteración según feedback
4. **Merge**: Squash merge a `main`

---

## 📚 Documentación

### Documentar Nuevo Código

**Requerido**:
- Docstrings en formato NumPy para todas las funciones públicas
- Type hints completos
- Ejemplos de uso en docstring
- Referencias científicas (papers, algoritmos)

### Documentación Sphinx

Para actualizar la documentación oficial:

```bash
cd docs

# Generar API docs
uv run sphinx-apidoc -o api ../colour_checker_detection

# Compilar HTML
uv run make html

# Ver resultado
# Windows: start _build/html/index.html
# macOS: open _build/html/index.html
# Linux: xdg-open _build/html/index.html
```

### Ejemplos Prácticos

Si añades funcionalidad nueva, considera añadir ejemplo en `examples/`:

```python
# examples/03_advanced/detect_custom_variant.py
"""
Detección de Variantes Personalizadas de ColorChecker
======================================================

Este ejemplo muestra cómo detectar variantes custom del ColorChecker
usando plantillas personalizadas.

Caso de uso: Cartas de calibración específicas para arqueología.
"""

import cv2
from colour_checker_detection import detect_colour_checkers_templated

def main():
    # Cargar imagen
    img = cv2.imread('excavation_photo.jpg')
    
    # Configurar detección para variante custom
    settings = {
        'variant': 'custom',
        'patch_count': 18,  # 18 parches en vez de 24
        'layout': (3, 6),   # 3 filas, 6 columnas
    }
    
    # Detectar
    detections = detect_colour_checkers_templated(img, settings)
    
    if detections:
        print(f"✅ Detectado {len(detections)} ColorChecker(s)")
        for i, det in enumerate(detections):
            print(f"  #{i+1}: {len(det.swatch_colours)} parches")
    else:
        print("❌ No se detectó ColorChecker")

if __name__ == '__main__':
    main()
```

---

## 🎓 Mejores Prácticas Científicas

Este proyecto es usado en investigación arqueológica y fotogramétrica. Mantener rigor científico es esencial.

### Trazabilidad del Color

- **Documenta transformaciones**: Cada paso de procesamiento debe estar documentado
- **Preserva metadata**: EXIF, calibración de cámara, condiciones de captura
- **Referencias bibliográficas**: Cita papers para algoritmos implementados

### Validación

- **Delta E**: Siempre reporta métricas de error colorimétrico
- **Ground truth**: Usa ColorChecker físico como referencia
- **Reproducibilidad**: Código debe producir mismos resultados con mismos inputs

### Performance

- **Imágenes grandes**: Fotogrametría usa imágenes de 40-100 MP
- **Batch processing**: Considera procesamiento por lotes
- **Memory efficiency**: Profile y optimiza uso de memoria

---

## 📞 Contacto

### Maintainers

**Laboratorio de Arqueología Digital UC**
- Email: victor.mendez@uc.cl
- GitHub: [@Laboratorio-de-Arqueologia-Digital-UC](https://github.com/Laboratorio-de-Arqueologia-Digital-UC)

### Canales de Comunicación

- **Issues**: Para bugs, features, preguntas técnicas
- **Discussions**: Para discusiones generales, ideas, Q&A
- **Email**: Para asuntos confidenciales o de colaboración

---

## 🙏 Reconocimientos

Este proyecto es un fork de [colour-checker-detection](https://github.com/colour-science/colour-checker-detection) por Color Developers.

Agradecemos a:
- **Color Developers** por el proyecto base
- **Todos los contribuidores** del proyecto original
- **Comunidad de Arqueología Digital** por feedback y testing

---

## 📖 Recursos Adicionales

### Lecturas Recomendadas

- [Cheung et al. (2004)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1478-4408.2004.tb00201.x) - Caracterización de cámaras con CCM
- [X-Rite ColorChecker](https://www.xrite.com/categories/calibration-profiling/colorchecker-classic) - Especificaciones del target físico
- [Colour Science Documentation](https://colour.readthedocs.io/) - Librería de ciencia del color

### Proyectos Relacionados

- [colour-science/colour](https://github.com/colour-science/colour) - Librería base de ciencia del color
- [rawpy](https://github.com/letmaik/rawpy) - Procesamiento de archivos RAW
- [OpenCV](https://opencv.org/) - Computer vision

---

## 📄 Licencia

Al contribuir, aceptas que tu código será licenciado bajo:

- **Apache 2.0** para módulos core
- **AGPL-3.0** solo si contribuyes al módulo `inference.py`

Ver [LICENSE_COMPLIANCE.md](LICENSE_COMPLIANCE.md) para detalles completos.

---

**¡Gracias por contribuir a colorchecker-pipeline!** 🎨📷🔬