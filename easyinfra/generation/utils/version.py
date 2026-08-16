import importlib
import importlib.metadata
from packaging import version

def is_version_greater_or_equal(library_name: str, library_version: str):
    return version.parse(importlib.metadata.version(library_name)) >= version.parse(library_version)
