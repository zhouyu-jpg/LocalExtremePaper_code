from setuptools import setup
from Cython.Build import cythonize
import numpy

# Build the extension once; users select any supported odd local-window size
# from 7 through 31 when running the batch driver.
setup(
    name="shash_tmax_local_window",
    ext_modules=cythonize(
        ["ghcnTMAX_shash_cy_windowN.pyx"],
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[numpy.get_include()]
)
