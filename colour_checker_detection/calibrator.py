
import colour
import numpy as np


def calculate_ccm(measured_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz 3x3 usando Cheung 2004.
    Args:
        measured_rgb: Array (24, 3) lineal.
        reference_rgb: Array (24, 3) lineal (target).
    Returns:
        Matriz (3, 3)
    """
    return colour.characterisation.matrix_colour_correction_Cheung2004(
        measured_rgb, reference_rgb
    )

def calculate_wb_multipliers(
    raw_image: np.ndarray,
    grey_patches_coords: list
) -> list[float]:
    """
    Calcula los multiplicadores RGB necesarios para neutralizar los grises.
    Vital para reemplazar 'use_camera_wb=True'.
    """
    # TODO: Implementar lógica para promediar parches grises y hallar factores R/G/B
    return [1.0, 1.0, 1.0, 1.0]
