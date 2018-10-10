from setuptools import setup, find_packages
import os

setup(name='bayesim',
      version='0.9.5',
      description='Fast model fitting via Bayesian inference',
      author='Rachel Kurchin, Giuseppe Romano',
      author_email='rkurchin@mit.edu, romanog@mit.edu',
      classifiers=['Programming Language :: Python :: 3.6'],
      long_description=open('README.rst').read(),
      install_requires=['pandas',
                        'deepdish',
                        'numpy',
                        'scipy',
                        'matplotlib',
                        'joblib'
                         ],
      license='GPLv2',\
      packages = ['bayesim'],
      entry_points = {
     'console_scripts': [
      'bayesim=bayesim.__main__:main'],
      },
      zip_safe=False)
