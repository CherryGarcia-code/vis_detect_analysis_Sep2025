"""Compatibility shim: expose I/O helpers at `src.visdetect.io`

This file exists so pickled objects that expect the module path
`src.visdetect.io` can be unpickled by importing the actual
implementations from `visdetect.core.io`.
"""
from .core.io import *
