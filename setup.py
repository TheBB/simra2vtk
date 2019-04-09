#!/usr/bin/env python3

from distutils.core import setup

setup(
    name='SIMRA2VTK',
    version='0.1',
    description='Data converter from SIMRA to VTK',
    author='Eivind Fonn',
    author_email='eivind.fonn@sintef.no',
    license='GPL3',
    url='https://github.com/TheBB/simra2vtk',
    py_modules=['simra_to_vtk'],
    entry_points={
        'console_scripts': ['simra2vtk=simra_to_vtk.__main__:main'],
    },
    install_requires=['click', 'scipy', 'tqdm', 'vtk'],
)
