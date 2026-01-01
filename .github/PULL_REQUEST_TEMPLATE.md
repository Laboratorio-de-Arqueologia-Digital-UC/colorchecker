# ========================================
# .github/PULL_REQUEST_TEMPLATE.md
# ========================================

## Descripción

<!-- Describe brevemente qué cambios introduces -->

Resuelve #(issue)

## Tipo de cambio

<!-- Marca con [x] lo que aplique -->

- [ ] 🐛 Bug fix (cambio que corrige un problema)
- [ ] ✨ Nueva feature (cambio que añade funcionalidad)
- [ ] ⚠️ Breaking change (cambio que rompe compatibilidad)
- [ ] 📝 Documentación
- [ ] 🎨 Estilo/formato
- [ ] ♻️ Refactorización
- [ ] ✅ Tests
- [ ] 🔧 Configuración/build

## Checklist

<!-- Antes de enviar el PR, verifica: -->

- [ ] Mi código sigue el estilo del proyecto (ejecuté `uv run ruff format`)
- [ ] Ejecuté los linters (`uv run ruff check --fix`)
- [ ] Añadí tests que cubren mis cambios
- [ ] Todos los tests pasan (`uv run pytest`)
- [ ] Actualicé la documentación si es necesario
- [ ] Añadí mi cambio al CHANGELOG.md (sección [Sin Publicar])
- [ ] Mis commits siguen Conventional Commits

## Tests

<!-- Describe qué tests añadiste o modificaste -->

```python
# Ejemplo de nuevo test
def test_batch_processing():
    images = [load_image(f"test{i}.jpg") for i in range(3)]
    results = detect_colour_checkers_batch(images)
    assert len(results) == 3
```

## Capturas de pantalla (si aplica)

<!-- Si tu cambio afecta la UI o los reportes visuales -->

| Antes | Después |
|-------|---------|
| (imagen) | (imagen) |

## Notas adicionales

<!-- Información extra que los revisores deban saber -->

- Este cambio requiere actualizar la versión de NumPy
- He probado en Windows y Linux, pero no en macOS

## Revisores sugeridos

<!-- @menciona a quien creas que debería revisar esto -->

@VmendezM 

---

### Para revisores

<!-- Checklist de revisión -->

- [ ] El código es claro y mantenible
- [ ] Los tests cubren los casos edge
- [ ] La documentación está actualizada
- [ ] No hay regresiones
- [ ] El changelog está actualizado