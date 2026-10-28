"""Builds libpyhxt.so into the package (build.sh), so that pip install gives a working
pyhxt. The library has no Python in it: the wheel is for any Python 3 (py3-none) on one
platform."""
import os
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel


class BuildLibrary(build_py):
    def run(self):
        super().run()
        here = os.path.dirname(os.path.abspath(__file__))
        subprocess.check_call(['bash', os.path.join(here, 'build.sh'),
                               os.path.join(self.build_lib, 'pyhxt', 'libpyhxt.so')])


class BinaryDistribution(Distribution):
    """A shared library in it: platlib, not purelib, so auditwheel can repair the wheel."""

    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        return ('py3', 'none', *super().get_tag()[2:])


setup(cmdclass={'build_py': BuildLibrary, 'bdist_wheel': PlatformWheel},
      distclass=BinaryDistribution)
