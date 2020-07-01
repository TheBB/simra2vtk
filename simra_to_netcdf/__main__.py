import os
import sys
from scipy.io import FortranFile
import click
from netCDF4 import Dataset


def add_variable(out, name, src, units, long_name=None, standard_name=None, extra_dims=()):
    var = out.createVariable(name, src.dtype, ('zc', 'xc', 'yc', *extra_dims))
    var[:] = src
    var.units = units

    if long_name is not None:
        var.long_name = long_name
    if standard_name is not None:
        var.standard_name = standard_name


def convert(meshfile, resfile, endian, out):
    headertype = endian + 'u4'
    floattype = endian + 'f4'
    inttype = endian + 'u4'

    with FortranFile(meshfile, 'r', header_dtype=headertype) as f:
        npts, nelems, imax, jmax, kmax, _ = f.read_ints(inttype)
        coords = f.read_reals(dtype=floattype).reshape(npts, 3)
        elems = f.read_ints(inttype).reshape(nelems, 8) - 1

    coords = coords.reshape((jmax, imax, kmax, 3)).transpose((2, 0, 1, 3))

    xc = out.createDimension('xc', imax)
    yc = out.createDimension('yc', jmax)
    zc = out.createDimension('zc', kmax)
    dim = out.createDimension('dim', 3)

    add_variable(out, 'x', coords[...,2], 'meters', standard_name='projection_x_coordinate')
    add_variable(out, 'y', coords[...,1], 'meters', standard_name='projection_y_coordinate')
    add_variable(out, 'z', coords[...,0], 'meters', standard_name='altitude')

    with FortranFile(resfile, 'r', header_dtype=headertype) as f:
        data = f.read_reals(dtype=floattype)
        time, data = data[0], data[1:].reshape(-1, 11)
        assert data.shape[0] == npts
        data = data.reshape((jmax, imax, kmax, 11)).transpose((2, 0, 1, 3))

    add_variable(out, 'u', data[..., :3], 'meter/second', long_name='velocity', extra_dims=('dim',))
    add_variable(out, 'ps', data[..., 3], 'pascal', long_name='hydrostatic pressure')
    add_variable(out, 'tk', data[..., 4], 'joule/kg3', long_name='turbulent kinetic energy')
    add_variable(out, 'td', data[..., 5], 'joule/second', long_name='energy dissipation rate')
    # add_variable(out, 'vtef', data[..., 6], '?', long_name='?')
    add_variable(out, 'pt', data[..., 7], 'kelvin', long_name='potential temperature')
    add_variable(out, 'pts', data[..., 8], 'kelvin', long_name='potential temperature due to hydrostatic contribution')
    add_variable(out, 'rho', data[..., 9], 'kg/m3', long_name='density')
    add_variable(out, 'rhos', data[..., 10], 'kg/m3', long_name='density due to hydrostatic contribution')


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

    if outfile is None and resfile is not None:
        name, _ = os.path.splitext(resfile)
        outfile = name + '.nc'
    elif outfile is None:
        outfile = 'cont.nc'

    endian = {'native': '=', 'big': '>', 'small': '<'}[endian]
    with Dataset(outfile, 'w') as f:
        f.Conventions = 'CF-1.7'
        convert(meshfile, resfile, endian, f)
