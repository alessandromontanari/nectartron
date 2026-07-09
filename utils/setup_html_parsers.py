from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name="html_parsers_cython",
    ext_modules=cythonize(
        Extension(
            "html_parsers_cython",
            sources=["html_parsers_cython.pyx"],
            include_dirs=[numpy.get_include()],
        ),
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'cdivision': True
        }
    ),
)
