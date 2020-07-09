from os.path import exists, dirname, join
import sys

import click
from scipy.io import FortranFile
import utm


def utm_convert(x, y, src, tgt):
    src_northern = src[-1].lower() == 'n'
    tgt_northern = tgt[-1].lower() == 'n'
    src_number = int(src[:-1])
    tgt_number = int(tgt[:-1])
    lat, lon = utm.to_latlon(x, y, src_number, northern=src_northern, strict=False)
    x, y, *_ = utm.from_latlon(lat, lon, force_zone_number=tgt_number)
    return x, y


def reader_args(func):
    args = [
        click.option('--intwidth', default=4),
        click.option('--floatwidth', default=4),
        click.option('--translate', type=(float, float), default=(0.0, 0.0)),
        click.option('--auto-translate', flag_value=True),
        click.option('--utmconvert', nargs=2, type=str, default=None),
        click.option('--endian', type=click.Choice(['native', 'big', 'little']), default='native'),
        click.option('--res', 'resfile', default='cont.res'),

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
                 floatwidth=4, intwidth=4, translate=(0, 0), auto_translate=False,
                 utmconvert=None, require_data=False):
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

        if auto_translate:
            infofile = join(dirname(self.meshfile), 'info.txt')
            with open(infofile, 'r') as f:
                translate = tuple(map(float, next(f).split()))
            print("Translating by ({}, {})".format(*translate))
        self.coords[:,0] += translate[0]
        self.coords[:,1] += translate[1]

        if utmconvert:
            x, y = utm_convert(self.coords[:,0], self.coords[:,1], *utmconvert)
            self.coords[:,0] = x
            self.coords[:,1] = y

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

        simrafile = join(dirname(self.meshfile), 'simra.in')
        if exists(simrafile):
            uref, href = 1.0, 1.0
            with open(simrafile, 'r') as f:
                for line in f:
                    if line.startswith(' UREF'):
                        uref = float(line.split('=')[-1][:-2])
                    elif line.startswith(' LENREF'):
                        href = float(line.split('=')[-1][:-2])
            self['u'] = self['u'] * uref
            self['tk'] = self['tk'] * uref**2
            self['ps'] = self['ps'] * uref**2
            self['td'] = self['td'] * uref**3 / href


    def __iter__(self):
        yield from self.keys()

    def __getitem__(self, key):
        return self.data[:, self.FIELDSLICES[key]]

    def __setitem__(self, key, value):
        self.data[:, self.FIELDSLICES[key]] = value

    def keys(self):
        yield from self.FIELDSLICES.keys()

    def values(self):
        for k in self:
            yield self[k]

    def items(self):
        for k in self:
            yield k, self[k]
