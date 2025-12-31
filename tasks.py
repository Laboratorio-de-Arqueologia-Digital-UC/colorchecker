"""Invoke - User Maintenance Tasks
=============================
"""

import shutil
from pathlib import Path
from invoke.tasks import task

# Rutas Clave del Usuario
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "colour_checker_detection" / "test_results"


def _delete_pattern(pattern: str, recursive: bool = True):
    """Ayuda para borrar archivos o carpetas por patrón."""
    for path in BASE_DIR.rglob(pattern):
        if path.is_dir() and recursive:
            print(f"Borrando directorio: {path}")
            shutil.rmtree(path)
        elif path.is_file():
            print(f"Borrando archivo: {path}")
            path.unlink()


@task
def clean(ctx, bytecode=True, results=False, pytest=True):
    """
    Limpia el repositorio de archivos temporales y resultados.

    Parámetros:
    -----------
    bytecode : bool
        Borra archivos .pyc y carpetas __pycache__ (Default: True).
    results : bool
        Borra la carpeta de resultados de tus scripts (Default: False).
        USA CON PRECAUCIÓN.
    pytest : bool
        Borra caché de pytest (Default: True).
    """
    print(">>> Iniciando Limpieza...")

    if bytecode:
        print("-> Limpiando Bytecode...")
        _delete_pattern("__pycache__")
        _delete_pattern("*.pyc", recursive=False)

    if pytest:
        print("-> Limpiando Pytest Cache...")
        _delete_pattern(".pytest_cache")

    if results:
        print(f"-> Limpiando Resultados en {RESULTS_DIR}...")
        if RESULTS_DIR.exists():
            shutil.rmtree(RESULTS_DIR)
            print(f"    Eliminado: {RESULTS_DIR}")
        else:
            print("    Nada que limpiar en resultados.")

    print(">>> Limpieza Finalizada.")


@task
def requirements(ctx):
    """
    Exporta requirements.txt usando uv (útil si mueves el script a otro lado).
    """
    print(">>> Exportando requirements.txt...")
    ctx.run("uv export --no-hashes --all-extras --no-dev > requirements.txt")
    print(">>> requirements.txt generado exitosamente.")
