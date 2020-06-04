import click
from collections import defaultdict
from itertools import takewhile
import numpy as np
from operator import attrgetter
import os
import os.path
import sys
from tqdm import tqdm
from scipy.io import FortranFile


class Face:

    def __init__(self):
        self.owner = None
        self.neighbour = None
        self.points = None
        self.boundary = None


# Outward-pointing numbering
FACES = [
    np.array([0, 3, 2, 1]),
    np.array([4, 5, 6, 7]),
    np.array([0, 1, 5, 4]),
    np.array([2, 3, 7, 6]),
    np.array([0, 4, 7, 3]),
    np.array([1, 2, 6, 5]),
]
BOUNDARIES = ['kmin', 'kmax', 'jmin', 'jmax', 'imin', 'imax']

# Known physical dimensions of fields
# kg, m, s, K, mol, A, cd
DIMENSIONS = {
    'u': [0, 1, -1, 0, 0, 0, 0],
    'ps': [1, -1, -2, 0, 0, 0, 0],
    'rho': [1, -3, 0, 0, 0, 0, 0],
    'tk': [0, 2, -2, 0, 0, 0, 0],
    'td': [0, 2, -3, 0, 0, 0, 0],
    'pt': [0, 0, 0, 1, 0, 0, 0],
    'pts': [0, 0, 0, 1, 0, 0, 0],
}


def foam_header(cls, obj, note='None'):
    return """FoamFile
{{
    version      2.0;
    format       ascii;
    class        {};
    note         "{}";
    object       {};
}}
""".format(cls, note, obj)


def internalfield_header(cls, obj, loc=0.0):
    return """FoamFile
{{
    version      2.0;
    format       ascii;
    class        {};
    location     "{}";
    object       p;
}}
""".format(cls, loc, obj)


def foam_points(filename, points):
    with open(filename, 'w') as f:
        f.write(foam_header('vectorField', 'points'))
        f.write(str(len(points)) + '\n')
        f.write('(\n')
        for pt in tqdm(points, 'Writing points'):
            f.write('({} {} {})\n'.format(*pt))
        f.write(')\n')


def foam_faces(filename, faces):
    with open(filename, 'w') as f:
        f.write(foam_header('faceList', 'faces'))
        f.write(str(len(faces)) + '\n')
        f.write('(\n')
        for face in tqdm(faces, 'Writing faces'):
            f.write('4({} {} {} {})\n'.format(*face.points))
        f.write(')\n')


def foam_labels(filename, faces, key, note):
    getter = attrgetter(key)
    with open(filename, 'w') as f:
        f.write(foam_header('labelList', key, note=note))
        f.write(str(len(faces)) + '\n')
        f.write('(\n')
        for face in tqdm(faces, 'Writing ' + key + 's'):
            f.write(str(getter(face)) + '\n')
        f.write(')\n')


def foam_boundaries(filename, nint, faces):
    with open(filename, 'w') as f:
        f.write(foam_header('polyBoundaryMesh', 'boundary'))
        f.write('{}\n'.format(len(BOUNDARIES)))
        f.write('(\n')
        while faces:
            boundaryname = faces[0].boundary
            nfaces = sum(1 for _ in takewhile(lambda f: f.boundary == boundaryname, faces))
            faces = faces[nfaces:]
            f.write('{}'.format(boundaryname))
            f.write('{\n')
            f.write('    type patch;\n')
            f.write('    nFaces {};\n'.format(nfaces))
            f.write('    startFace {};\n'.format(nint))
            f.write('}\n')
            nint += nfaces
        f.write(')\n')


def foam_internalfield(filename, fieldname, data):
    data = data.reshape((len(data), -1))
    vectorp = data.shape[-1] != 1
    with open(filename, 'w') as f:
        f.write(internalfield_header(('volVectorField' if vectorp else 'volScalarField'), fieldname))
        if fieldname in DIMENSIONS:
            f.write('dimensions [')
            f.write(' '.join(map(str, DIMENSIONS[fieldname])))
            f.write('];\n')
        if vectorp:
            f.write('internalField nonuniform List<vector>\n')
        else:
            f.write('internalField nonuniform List<scalar>\n')
        f.write(str(len(data)) + '\n')
        f.write('(\n')
        for entry in tqdm(data, 'Writing field ' + fieldname):
            if vectorp:
                f.write('  (' + ' '.join(str(e) for e in entry) + ')\n')
            else:
                f.write('  ' + str(entry[0]) + '\n')
        f.write(');\n')

        if fieldname in ('u', 'tk', 'td', 'ps'):
            f.write('boundaryField\n{\n')
            if fieldname in ('u', 'tk', 'td'):
                f.write('  imin\n  {\n    type fixedValue;\n    value $internalfield;\n  }\n')
            elif fieldname in ('ps',):
                f.write('  imax\n  {\n    type fixedValue;\n    value $internalfield;\n  }\n')
            f.write('}\n')


def convert_grid(meshfile, resfile, outdir, endian):
    headertype = endian + 'u4'
    floattype = endian + 'f4'
    inttype = endian + 'u4'

    with FortranFile(meshfile, 'r', header_dtype=headertype) as f:
        npts, nelems, imax, jmax, kmax, _ = f.read_ints(inttype)
        coords = f.read_reals(dtype=floattype).reshape(npts, 3)
        elems = f.read_ints(inttype).reshape(-1, 8) - 1

    # Compute owner and neighbour IDs for each face
    faces = defaultdict(Face)
    for elemidx, elem in enumerate(tqdm(elems, 'Mapping faces')):
        for faceidx, boundaryname in zip(FACES, BOUNDARIES):
            face_pts = elem[faceidx]
            key = tuple(sorted(face_pts))
            face = faces[key]
            face.boundary = boundaryname
            if face.owner is None:
                face.owner = elemidx
                face.points = face_pts
            else:
                assert face.neighbour is None
                face.neighbour = elemidx
                face.boundary = None

    # Sort faces by owner, neighbour
    faces = list(faces.values())
    bnd_faces = []
    for boundaryname in BOUNDARIES:
        temp = [face for face in faces if face.boundary == boundaryname]
        bnd_faces.extend(sorted(temp, key=attrgetter('owner')))
    int_faces = [face for face in faces if face.neighbour is not None]
    int_faces = sorted(int_faces, key=attrgetter('neighbour'))
    int_faces = sorted(int_faces, key=attrgetter('owner'))
    for face in bnd_faces:
        face.neighbour = -1
    faces = int_faces + bnd_faces

    note = 'nPoints: {} nCells: {} nFaces: {} nInternalFaces: {}'.format(
        len(coords), len(elems), len(faces), len(int_faces)
    )

    foam_points(os.path.join(outdir, 'points'), coords)
    foam_faces(os.path.join(outdir, 'faces'), faces)
    foam_labels(os.path.join(outdir, 'owner'), faces, 'owner', note=note)
    foam_labels(os.path.join(outdir, 'neighbour'), faces, 'neighbour', note=note)
    foam_boundaries(os.path.join(outdir, 'boundary'), len(int_faces), bnd_faces)

    with FortranFile(resfile, 'r', header_dtype=headertype) as f:
        data = f.read_reals(dtype=floattype)
        time, data = data[0], data[1:].reshape(-1, 11)
        assert data.shape[0] == npts

    timedir = os.path.join(outdir, str(time))
    if not os.path.exists(timedir):
        os.makedirs(timedir)

    foam_internalfield(os.path.join(timedir, 'u'), 'u', data[:,:3])
    foam_internalfield(os.path.join(timedir, 'ps'), 'ps', data[:,3])
    foam_internalfield(os.path.join(timedir, 'tk'), 'tk', data[:,4])
    foam_internalfield(os.path.join(timedir, 'td1'), 'td1', data[:,5])
    foam_internalfield(os.path.join(timedir, 'vtef'), 'vtef', data[:,6])
    foam_internalfield(os.path.join(timedir, 'pt'), 'pt', data[:,7])
    foam_internalfield(os.path.join(timedir, 'pts1'), 'pts1', data[:,8])
    foam_internalfield(os.path.join(timedir, 'rho'), 'rho', data[:,9])
    foam_internalfield(os.path.join(timedir, 'rhos'), 'rhos', data[:,10])


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

    if outfile is None:
        outfile, _ = os.path.splitext(resfile)

    if os.path.exists(outfile) and not os.path.isdir(outfile):
        print("Output location {} already exists, and is not a directory".format(outfile), file=sys.stderr)
        sys.exit(1)
    elif not os.path.exists(outfile):
        os.makedirs(outfile)

    endian = {'native': '=', 'big': '>', 'small': '<'}[endian]
    convert_grid(meshfile, resfile, outfile, endian)
