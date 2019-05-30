from setuptools import setup, find_packages
import os, sys

"""The setup script."""

try: # for pip >= 10
    from pip._internal import req
except ImportError: # for pip <= 9.0.3
    from pip import req

with open('README.rst') as readme_file:
    readme = readme_file.read()

history = ''

links = []
requires = []
try:
    requirements = list(req.parse_requirements(os.path.abspath('requirements.txt')))
except TypeError:

    try:
        from pip import download
    except ImportError:
        from pip._internal import download
    
    # new versions of pip requires a session
    requirements = req.parse_requirements(
        os.path.abspath('requirements.txt'), session=download.PipSession())

for item in requirements:
    req = item.req
    # we want to handle package names and also repo urls
    if getattr(item, 'url', None):  # older pip has url
        links.append(str(item.url))
        if req:
            req = "==".join(str(req).split("-", 1))
    if getattr(item, 'link', None): # newer pip has link
        links.append(str(item.link))
        if req:
            req = "==".join(str(req).split("-", 1))
    if req:
        requires.append(str(req))

setup(name='bayesim',
      version='0.9.16',
      description='Fast model fitting via Bayesian inference',
      author='Rachel Kurchin, Giuseppe Romano',
      author_email='rkurchin@mit.edu, romanog@mit.edu',
      classifiers=['Programming Language :: Python :: 3.7'],
      long_description=open('README.rst').read(),
      install_requires=requires,
      dependency_links=links,
      license='GPLv2',
      packages = ['bayesim'],
      entry_points = {
     'console_scripts': [
      'bayesim=bayesim.__main__:main'],
      },
      zip_safe=False)
