from setuptools import setup

setup(name='seegnature',
      version='0.2.001',
      description='Module for supporting detection of electrophysiological signatures',
      url='https://github.com/liephat/seegnature',
      author='Mike Imhof',
      author_email='mike.imhof@uni-bamberg.de',
      packages=['seegnature'],
      install_requires=[
          'numpy',
          'matplotlib',
          'tensorflow',
          'tflearn'
      ],
      zip_safe=False)