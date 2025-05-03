# setup.py

from setuptools import setup, Extension
import numpy

module_name = "encode_seq"
extension = Extension(
    module_name,
    sources=["encode_seq.c"],
    include_dirs=[numpy.get_include()]
)

setup(
    name=module_name,
    ext_modules=[extension],
)


