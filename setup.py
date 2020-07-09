#!/usr/bin/env python3

from distutils.core import setup

setup(
    name='SIMRA2VTK',
    version='0.1',
    description='Data converter from SIMRA to VTK and/or OpenFOAM',
    author='Eivind Fonn',
    author_email='eivind.fonn@sintef.no',
    license='GPL3',
    url='https://github.com/TheBB/simra2vtk',
    packages=['simrato', 'simra_to_netcdf'],
    entry_points={
        'console_scripts': [
            'simra2vtk=simrato.vtk:main',
            'simra2openfoam=simrato.openfoam:main',
            'simra2netcdf=simra_to_netcdf.__main__:main',
        ],
    },
    install_requires=['click', 'scipy', 'tqdm', 'vtk', 'netcdf4', 'utm'],
)
