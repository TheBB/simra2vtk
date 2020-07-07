import click
from scipy.io import FortranFile
from os.path import exists
import sys


def reader_args(func):
    args = [
        click.option('--intwidth', default=4),
        click.option('--floatwidth', default=4),
        click.option('--endian', type=click.Choice(['native', 'big', 'little']), default='native'),
        click.option('--res', 'resfile', default='cont.res'),
        click.option('--mesh', 'meshfile', default='mesh.dat'),
    ]
    for arg in args:
        func = arg(func)
    return func


class Simra:

    FIELDSLICES = {
        'u': slice(3),
        'ps': 3,
        'tk': 4,
        'td': 5,
        'vtef': 6,
        'pt': 7,
        'pts': 8,
        'rho': 9,
        'rhos': 10,
    }

    def __init__(self, meshfile='mesh.dat', resfile='cont.res', endian='native',
                 floatwidth=4, intwidth=4, require_data=False):
        self.meshfile = meshfile
        self.resfile = resfile

        endian = {'native': '=', 'big': '>', 'small': '<'}[endian]
        inttype = f'{endian}u{intwidth}'
        floattype = f'{endian}f{intwidth}'
        headertype = inttype

        if not exists(meshfile):
            raise FileNotFoundError(meshfile)

        with FortranFile(meshfile, 'r', header_dtype=headertype) as f:
            self.npts, self.nelems, self.imax, self.jmax, self.kmax, _ = f.read_ints(dtype=inttype)
            self.coords = f.read_reals(dtype=floattype).reshape(self.npts, 3).astype('f8')
            self.elems = f.read_ints(dtype=inttype).reshape(self.nelems, 8) - 1

        self.has_data = exists(resfile)
        if not self.has_data and require_data:
            raise FileNotFoundError(meshfile)
        elif not self.has_data:
            self.resfile = None
            print("No result file, running in mesh-only mode")
            return

        with FortranFile(resfile, 'r', header_dtype=headertype) as f:
            data = f.read_reals(dtype=floattype)
            self.time, self.data = data[0], data[1:].reshape(-1, 11).astype('f8')
            assert self.data.shape[0] == self.npts

    def __iter__(self):
        yield from self.keys()

    def __getitem__(self, key):
        return self.data[:, self.FIELDSLICES[key]]

    def keys(self):
        yield from self.FIELDSLICES.keys()

    def values(self):
        for k in self:
            yield self[k]

    def items(self):
        for k in self:
            yield k, self[k]
