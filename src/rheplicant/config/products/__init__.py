"""Deterministic scientific product containers."""

from .extractors import (
    EXTRACTOR_REGISTRY,
    RUN_KIND_SELECTORS,
    ExtractedProduct,
    extract_run_payload,
    numeric_leaves,
)
from .manifest import (
    PRODUCT_FORMATS,
    PRODUCT_SELECTORS,
    build_product_manifest,
    validate_product_bundle,
)
from .types import ProductBundle, ProductFile, ProductOmission

__all__ = [
    "PRODUCT_FORMATS",
    "PRODUCT_SELECTORS",
    "EXTRACTOR_REGISTRY",
    "RUN_KIND_SELECTORS",
    "ExtractedProduct",
    "ProductBundle",
    "ProductFile",
    "ProductOmission",
    "build_product_manifest",
    "extract_run_payload",
    "numeric_leaves",
    "validate_product_bundle",
]
