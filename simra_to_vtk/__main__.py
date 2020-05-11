import click
import os.path
import sys
from tqdm import tqdm
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from scipy.io import FortranFile


def vtk_id_list(data):
    ret = vtk.vtkIdList()
    for d in data:
        ret.InsertNextId(int(d))
    return ret


def convert_grid(coords, elems):
    coords = np.array(coords, dtype=np.float32)
    elems = np.array(elems, dtype=np.int32)
    points = vtk.vtkPoints()
    for pt in tqdm(coords, 'Copying points'):
        points.InsertNextPoint(pt[0], pt[1], pt[2])
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    for elem in tqdm(elems, 'Copying elements'):
        grid.InsertNextCell(vtk.VTK_HEXAHEDRON, vtk_id_list(elem))
    return grid


def add_array(pointdata, data, name):
    array = numpy_to_vtk(np.array(data, dtype=np.float32), deep=1)
    array.SetName(name)
    pointdata.AddArray(array)


def convert(meshfile, resfile, endian):
    headertype = endian + 'u4'
    floattype = endian + 'f4'
    inttype = endian + 'u4'

    with FortranFile(meshfile, 'r', header_dtype=headertype) as f:
        npts, nelems, imax, jmax, kmax, _ = f.read_ints(inttype)
        coords = f.read_reals(dtype=floattype).reshape(npts, 3)
        elems = f.read_ints(inttype).reshape(nelems, 8) - 1

    grid = convert_grid(coords, elems)

    with FortranFile(resfile, 'r', header_dtype=headertype) as f:
        data = f.read_reals(dtype=floattype)
        time, data = data[0], data[1:].reshape(-1, 11)
        assert data.shape[0] == npts

    pointdata = grid.GetPointData()
    add_array(pointdata, data[:,:3], 'u')
    add_array(pointdata, data[:,3], 'ps')
    add_array(pointdata, data[:,4], 'tk')
    add_array(pointdata, data[:,5], 'td1')
    add_array(pointdata, data[:,6], 'vtef')
    add_array(pointdata, data[:,7], 'pt')
    add_array(pointdata, data[:,8], 'pts1')
    add_array(pointdata, data[:,9], 'rho')
    add_array(pointdata, data[:,10], 'rhos')

    return grid


@click.command()
@click.option('--mesh', 'meshfile', default='mesh.dat')
@click.option('--res', 'resfile', default='cont.res')
@click.option('--endian', type=click.Choice(['native', 'big', 'little']), default='native')
@click.option('--out', 'outfile')
def main(meshfile, resfile, outfile, endian):
    if not os.path.exists(meshfile):
        print("Can't find {} --- please specify mesh file with --mesh FILENAME".format(meshfile), file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(resfile):
        print("Can't find {} --- please specify result file with --res FILENAME".format(resfile), file=sys.stderr)
        sys.exit(1)

    endian = {'native': '=', 'big': '>', 'small': '<'}[endian]
    grid = convert(meshfile, resfile, endian)

    if outfile is None:
        name, _ = os.path.splitext(resfile)
        outfile = name + '.vtk'

    print("Writing {}".format(outfile))
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(outfile)
    writer.SetInputData(grid)
    writer.Write()
