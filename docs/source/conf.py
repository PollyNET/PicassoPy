

import os
import sys

from sphinx_pyproject import SphinxConfig
#import recommonmark

sys.path.insert(0, os.path.abspath("../../"))  

import ppcpy

config = SphinxConfig("../../pyproject.toml", globalns=globals())
copyright = f"2025 {author}"