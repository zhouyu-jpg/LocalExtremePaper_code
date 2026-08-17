from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("ghcnTMAX_shash_cy", ["ghcnTMAX_shash_cy.pyx"], include_dirs=[np.get_include()]),
    Extension("ghcnTMIN_shash_cy", ["ghcnTMIN_shash_cy.pyx"], include_dirs=[np.get_include()])
]

setup(
    name="GHCN_SHASH_Modeling",
    ext_modules=cythonize(extensions, language_level=3)
)
