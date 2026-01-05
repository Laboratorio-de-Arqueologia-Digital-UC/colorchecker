
from __future__ import annotations

from pathlib import Path

import numpy as np
import rawpy
from colour.utilities import as_float_array


def load_raw_linear(path: str | Path) -> tuple[np.ndarray, list[float]]:
    """
    Reads a RAW image in linear space (demosaiced, camera WB applied, but no gamma/curve).
    Returns normalized float image (0-1) and the WB multipliers used.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{path} no existe")

    with rawpy.imread(str(path)) as raw:
        # Extraemos el WB que la cámara "pensó" que era correcto.
        as_shot_wb = list(raw.camera_whitebalance)

        # Modo Lineal: 16-bit para preservar detalle en sombras
        img = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=True, # Mantenemos esto PERO retornamos los valores usados
            output_color=rawpy.ColorSpace.raw, # pyright: ignore
            output_bps=16
        )
        return as_float_array(img) / 65535.0, as_shot_wb

def load_raw_visual(path: str | Path, brightness: float = 1.5) -> np.ndarray:
    """
    Reads a RAW image processed for visualization (sRGB, gamma applied).
    Returns normalized float image (0-1).
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{path} no existe")

    with rawpy.imread(str(path)) as raw:
        # Modo Visual
        img = raw.postprocess(
            use_camera_wb=True,
            bright=brightness,
            no_auto_bright=True
        )
        return as_float_array(img) / 255.0
