#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="David Diaz",
    author_email='david.daniel.diaz@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Implementation of Size-Constrained Region Merging as a graph-based image segmentation routine constructed from scikit-image building blocks.",
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='scrm',
    name='scrm',
    packages=find_packages(include=['scrm', 'scrm.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/d-diaz/scrm',
    version='0.1.0',
    zip_safe=False,
)
