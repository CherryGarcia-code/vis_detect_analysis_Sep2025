"""Compatibility shim: expose session dataclasses at `src.visdetect.session`

This file exists so pickled objects that expect the module path
`src.visdetect.session` can be unpickled by importing the actual
implementations from `visdetect.core.session`.
"""
from .core.session import *
