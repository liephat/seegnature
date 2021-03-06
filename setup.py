from setuptools import setup

setup(name='seegnature',
      version='0.3.0',
      description='Module for supporting detection of electrophysiological signatures',
      url='https://github.com/liephat/seegnature',
      author='Mike Imhof',
      author_email='mike.imhof@uni-bamberg.de',
      packages=['seegnature'],
      install_requires=[
          'numpy',
          'matplotlib',
          'pandas'
      ],
      zip_safe=False)