
import numpy as np
from scipy.spatial import distance as dist

def order_points(pts):
    """
    Ordena las coordenadas del cuadrilátero: TL, TR, BR, BL.
    Esto previene el espejado (mirroring) en la extracción.
    """
    # Ordenar por coord x (primero izquierda, luego derecha)
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # Puntos de la izquierda/derecha
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # Ordenar los de la izquierda por y (TL, BL)
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # Ordenar los de la derecha:
    # Usamos distancia euclidiana al TL para distinguir TR (más cerca) de BR (más lejos)?
    # O simplemente por Y.
    # Si hay mucha rotación, Y puede fallar. Distance es más robusto.
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (tr, br) = rightMost[np.argsort(D), :]

    # Pero espera, si es TR, Y debería ser menor que BR usualmente?
    # Si está rotado 90 grados??
    # Mejor usar simple sorting si asumimos "upright" canonical.
    # Pero si la imagen viene en cualquier rotación...
    # El método clásico es:
    # TL: min sum(pts)
    # BR: max sum(pts)
    # TR: min diff(y-x)? NO, diff(np.diff(pts, axis=1))
    # Vamos a usar la implementación robusta de PyImageSearch.
    
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL

    return rect
