#!/usr/bin/env python

from setuptools import setup, find_packages
import re

for line in open('pyea/__init__.py'):
    match = re.match("__version__ *= *'(.*)'", line)
    if match:
        __version__, = match.groups()

setup(name='pyea',
      version=__version__,
      description='Set of pure python libraries including the most common evolutionary algorithms',
      long_description=open('README.rst').read(),
      author='Alejandro CORREA BAHNSEN',
      author_email='al.bahnsen@gmail.com',
      url='https://github.com/albahnsen/pyea',
      license='new BSD',
      packages=find_packages(),
      keywords=['optimization', 'evolutionary', 'genetic'],
      install_requires=['scikit-learn>=0.15.0b2','pandas>=0.14.0','numpy>=1.8.0'],
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Programming Language :: Python',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: OS Independent',],
      )
