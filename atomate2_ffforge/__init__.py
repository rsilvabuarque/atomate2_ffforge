"""
Top-level package for FFForge.

`atomate2_ffforge` is kept as an alias for backward compatibility.
"""

from importlib import import_module as _imp
import sys as _sys

# expose this package also under the canonical PyPI name
_sys.modules.setdefault("atomate2_ffforge", _imp(__name__))
