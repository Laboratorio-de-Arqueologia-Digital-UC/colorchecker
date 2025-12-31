"""
Detection
=========

Colour checker detection algorithms and utilities.

This subpackage provides common utilities, segmentation-based detection
methods, and machine learning inference approaches for identifying colour
checkers in images.
"""

from .common import (
    DTYPE_FLOAT_DEFAULT,
    DTYPE_INT_DEFAULT,
    SETTINGS_CONTOUR_DETECTION_DEFAULT,
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC,
    SETTINGS_DETECTION_COLORCHECKER_SG,
    DataDetectionColourChecker,
    DataSegmentationColourCheckers,
    approximate_contour,
    as_float32_array,
    as_int32_array,
    cluster_swatches,
    contour_centroid,
    detect_contours,
    filter_clusters,
    is_square,
    quadrilateralise_contours,
    reformat_image,
    remove_stacked_contours,
    sample_colour_checker,
    scale_contour,
    swatch_colours,
    swatch_masks,
    transform_image,
)

# isort: split

import typing

if typing.TYPE_CHECKING:
    from .inference import (
        SETTINGS_INFERENCE_COLORCHECKER_CLASSIC,
        SETTINGS_INFERENCE_COLORCHECKER_CLASSIC_MINI,
        detect_colour_checkers_inference,
        extractor_inference,
        inferencer_default,
    )

# isort: split

from .segmentation import (
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC,
    SETTINGS_SEGMENTATION_COLORCHECKER_NANO,
    SETTINGS_SEGMENTATION_COLORCHECKER_SG,
    detect_colour_checkers_segmentation,
    extractor_segmentation,
    segmenter_default,
)

# isort: split

from .templated import (
    SETTINGS_TEMPLATED_COLORCHECKER_CLASSIC,
    WarpingData,
    detect_colour_checkers_templated,
    extractor_templated,
    segmenter_templated,
)
from .templates import (
    PATH_TEMPLATE_COLORCHECKER_CLASSIC,
    PATH_TEMPLATE_COLORCHECKER_CREATIVE_ENHANCEMENT,
    ROOT_TEMPLATES,
    Template,
    generate_template,
    load_template,
)

# isort: split

from .plotting import plot_detection_results

__all__ = [
    "DTYPE_FLOAT_DEFAULT",
    "DTYPE_INT_DEFAULT",
    "SETTINGS_CONTOUR_DETECTION_DEFAULT",
    "SETTINGS_DETECTION_COLORCHECKER_CLASSIC",
    "SETTINGS_DETECTION_COLORCHECKER_SG",
    "DataDetectionColourChecker",
    "DataSegmentationColourCheckers",
    "approximate_contour",
    "as_float32_array",
    "as_int32_array",
    "cluster_swatches",
    "contour_centroid",
    "detect_contours",
    "filter_clusters",
    "is_square",
    "quadrilateralise_contours",
    "reformat_image",
    "remove_stacked_contours",
    "sample_colour_checker",
    "scale_contour",
    "swatch_colours",
    "swatch_masks",
    "transform_image",
]
__all__ += [
    "SETTINGS_INFERENCE_COLORCHECKER_CLASSIC",
    "SETTINGS_INFERENCE_COLORCHECKER_CLASSIC_MINI",
    "detect_colour_checkers_inference",
    "extractor_inference",
    "inferencer_default",
]
__all__ += [
    "SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC",
    "SETTINGS_SEGMENTATION_COLORCHECKER_NANO",
    "SETTINGS_SEGMENTATION_COLORCHECKER_SG",
    "detect_colour_checkers_segmentation",
    "extractor_segmentation",
    "segmenter_default",
]
__all__ += [
    "detect_colour_checkers_templated",
    "extractor_templated",
    "segmenter_templated",
    "SETTINGS_TEMPLATED_COLORCHECKER_CLASSIC",
    "WarpingData",
]
__all__ += [
    "Template",
    "generate_template",
    "load_template",
    "ROOT_TEMPLATES",
    "PATH_TEMPLATE_COLORCHECKER_CLASSIC",
    "PATH_TEMPLATE_COLORCHECKER_CREATIVE_ENHANCEMENT",
]
__all__ += [
    "plot_detection_results",
]


def __getattr__(name: str) -> typing.Any:
    """
    Return the value of the attribute with the given name.

    Parameters
    ----------
    name
        Name of the attribute to return.

    Returns
    -------
    :class:`typing.Any`
        Value of the attribute.
    """
    if name in [
        "SETTINGS_INFERENCE_COLORCHECKER_CLASSIC",
        "SETTINGS_INFERENCE_COLORCHECKER_CLASSIC_MINI",
        "detect_colour_checkers_inference",
        "extractor_inference",
        "inferencer_default",
    ]:
        import importlib

        return getattr(importlib.import_module(".inference", __package__), name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
