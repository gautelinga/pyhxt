"""Builds libpyhxt.so into the package (build.sh), so that pip install gives a working
pyhxt. The library has no Python in it, but the wheel is marked as binary."""
import os
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution


class BuildLibrary(build_py):
    def run(self):
        super().run()
        here = os.path.dirname(os.path.abspath(__file__))
        subprocess.check_call(['bash', os.path.join(here, 'build.sh'),
                               os.path.join(self.build_lib, 'pyhxt', 'libpyhxt.so')])


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(cmdclass={'build_py': BuildLibrary}, distclass=BinaryDistribution)
