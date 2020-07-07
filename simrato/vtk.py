from os.path import splitext

import click
from tqdm import tqdm
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from . import Simra, reader_args


def vtk_id_list(data):
    ret = vtk.vtkIdList()
    for d in data:
        ret.InsertNextId(int(d))
    return ret


def convert_grid(coords, elems):
    points = vtk.vtkPoints()
    for pt in tqdm(coords, 'Copying points'):
        points.InsertNextPoint(pt[0], pt[1], pt[2])
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    for elem in tqdm(elems, 'Copying elements'):
        grid.InsertNextCell(vtk.VTK_HEXAHEDRON, vtk_id_list(elem))
    return grid


def add_array(pointdata, data, name):
    array = numpy_to_vtk(data, deep=1)
    array.SetName(name)
    pointdata.AddArray(array)


def convert(simra):
    grid = convert_grid(simra.coords, simra.elems)
    if not simra.has_data:
        return grid
    pointdata = grid.GetPointData()
    for k, v in simra.items():
        add_array(pointdata, v, k)
    return grid


@click.command()
@reader_args
@click.option('--out', 'outfile')
def main(outfile, **kwargs):
    simra = Simra(**kwargs)
    grid = convert(simra)

    if outfile is None and simra.resfile is not None:
        name, _ = splitext(simra.resfile)
        outfile = name + '.vtk'
    elif outfile is None:
        outfile = 'cont.vtk'

    print("Writing {}".format(outfile))
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(outfile)
    writer.SetInputData(grid)
    writer.Write()
