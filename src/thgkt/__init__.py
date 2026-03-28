"""Top-level package for THGKT."""

from thgkt.config import load_config, load_default_config
from thgkt.schemas.canonical import CanonicalBundle, CanonicalTable
from thgkt.schemas.validators import SchemaValidationError, ValidationReport, validate_bundle

__all__ = [
    "CanonicalBundle",
    "CanonicalTable",
    "SchemaValidationError",
    "ValidationReport",
    "load_config",
    "load_default_config",
    "validate_bundle",
]

__version__ = "0.1.0"
