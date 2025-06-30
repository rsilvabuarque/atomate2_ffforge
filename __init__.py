"""
atomate2_ffforge top-level package.

Notes
-----
`ffforge` is kept as a backwards-compatibility alias so unit tests or
notebooks that use the shorter name continue to work:

>>> import ffforge.sampling
>>> import atomate2_ffforge.sampling                       # same module
"""

from importlib import import_module as _imp
import sys as _sys

# expose this package also under the short alias "ffforge"
_sys.modules.setdefault("ffforge", _imp(__name__))
