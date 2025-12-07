from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the extension
extensions = [
    Extension(
        "empty_drops_core",
        ["empty_drops_core.pyx"],
        include_dirs=[np.get_include()],
        language='c++'
    )
]

# Setup configuration
setup(
    name="empty_drops_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'nonecheck': False,
            'cdivision': True,
        }
    ),
    zip_safe=False,
) 