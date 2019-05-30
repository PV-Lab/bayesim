from setuptools import setup, find_packages
import os, sys

setup(name='bayesim',
      version='0.9.16',
      description='Fast model fitting via Bayesian inference',
      author='Rachel Kurchin, Giuseppe Romano',
      author_email='rkurchin@mit.edu, romanog@mit.edu',
      classifiers=['Programming Language :: Python :: 3.7'],
      long_description=open('README.rst').read(),
      install_requires=['pandas', 'scipy', 'matplotlib', 'joblib', 'tables']
      license='GPLv2',
      packages = ['bayesim'],
      entry_points = {
     'console_scripts': [
      'bayesim=bayesim.__main__:main'],
      },
      zip_safe=False)
