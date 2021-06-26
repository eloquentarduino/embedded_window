import os.path
from distutils.core import setup
from glob import glob
from os.path import isdir

setup(
  name = 'embedded_window',
  packages = ['embedded_window'],
  version = '0.0.1',
  license='MIT',
  description = 'A sliding window that exports to C++',
  author = 'Simone Salerno',
  author_email = 'eloquentarduino@gmail.com',
  url = 'https://github.com/eloquentarduino/embedded_window',
  download_url = 'https://github.com/eloquentarduino/embedded_window/blob/master/dist/embedded_window-0.0.1.tar.gz?raw=true',
  keywords = [
    'ML',
    'microcontrollers',
    'machine learning'
  ],
  install_requires=[
    'numpy',
  ],
  package_data= {},
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Code Generators',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
