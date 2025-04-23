"""Initialize pyximport for compiling Cython extensions and imports the `get_dist_maps` function."""

import pyximport

pyximport.install(pyximport=True, language_level=3)
# noinspection PyUnresolvedReferences
from ._get_dist_maps import get_dist_maps
