#!/usr/bin/env python3
'''Use this to install module'''
from os import path
from setuptools import setup, find_packages

version = '0.1.0'
this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

## TODO: check versions

setup(
    name='rcnn-mnist',
    version=version,
    description='Recurrent Convolutional Neural Network implementation on MNIST.',
    author='Matt Lyon',
    author_email='matthewlyon18@gmail.com',
    long_description='test',
    long_description_content_type='text/markdown',
    python_requires='>=3.7',
    license='MIT License',
    packages=find_packages(),
    install_requires=[
        'tensorflow==2.3.1',
    ],
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows :: Windows 10'
    ],
    keywords=['ai', 'cnn', 'rcnn', 'ml', 'rnn', 'mnist'],
    include_package_data=True
)