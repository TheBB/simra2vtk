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

from simrato import Simra, reader_args


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
BOUNDARIES = ['ground', 'ceiling', 'wall', 'wall', 'wall', 'wall']

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


def foam_boundaries(filename, nint, faces, boundary_names):
    with open(filename, 'w') as f:
        f.write(foam_header('polyBoundaryMesh', 'boundary'))
        f.write('{}\n'.format(len(boundary_names)))
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


def foam_internalfield(filename, fieldname, data, boundaries=(), faces=()):
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

        if not boundaries:
            return

        f.write('boundaryField\n{\n')
        for boundary in boundaries:
            f.write('  ' + boundary + '\n  {\n    type fixedValue;\n')
            if vectorp:
                f.write('    value nonuniform List<vector>;\n')
            else:
                f.write('    value nonuniform List<scalar>;\n')
            bnd_data = np.array([data[face.owner] for face in faces if face.boundary == boundary])
            f.write('    ' + str(len(bnd_data)) + '\n    (\n')
            for entry in bnd_data:
                if vectorp:
                    f.write('      (' + ' '.join(str(e) for e in entry) + ')\n')
                else:
                    f.write('      ' + str(entry[0]) + '\n')
            f.write('    )\n')
            f.write('  }\n')
        f.write('}\n')


def cellify(data, elems):
    retval = np.zeros((elems.shape[0], data.shape[1]))
    for i, elem in tqdm(enumerate(elems), 'Cellifying', total=len(elems)):
        retval[i] = np.mean(data[elem,:], axis=0)
    return retval


def convert_grid(simra, outdir):
    # Automatic boundary detection based on mean velocity
    tol = 1e-2
    boundary_names = BOUNDARIES.copy()
    mean_velocity = np.mean(simra['u'][:,:2], axis=0)
    mean_velocity /= np.linalg.norm(mean_velocity)
    if np.dot(mean_velocity, [-1, 0]) < -tol:
        print('Detected west-to-east flow')
        boundary_names[4] = 'inflow'
        boundary_names[5] = 'outflow'
    elif np.dot(mean_velocity, [1, 0]) < -tol:
        print('Detected east-to-west flow')
        boundary_names[5] = 'inflow'
        boundary_names[4] = 'outflow'
    if np.dot(mean_velocity, [0, -1]) < -tol:
        print('Detected south-to-north flow')
        boundary_names[2] = 'inflow'
        boundary_names[3] = 'outflow'
    elif np.dot(mean_velocity, [0, 1]) < -tol:
        print('Detected north-to-south flow')
        boundary_names[3] = 'inflow'
        boundary_names[2] = 'outflow'

    # Compute data on cells by averaging
    data = cellify(simra.data, simra.elems)

    # Compute owner and neighbour IDs for each face
    faces = defaultdict(Face)
    for elemidx, elem in enumerate(tqdm(simra.elems, 'Mapping faces')):
        for faceidx, boundaryname in zip(FACES, boundary_names):
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
    bnd_faces, found = [], set()
    for boundaryname in boundary_names:
        if boundaryname in found:
            continue
        found.add(boundaryname)
        temp = [face for face in faces if face.boundary == boundaryname]
        bnd_faces.extend(sorted(temp, key=attrgetter('owner')))
    int_faces = [face for face in faces if face.neighbour is not None]
    int_faces = sorted(int_faces, key=attrgetter('neighbour'))
    int_faces = sorted(int_faces, key=attrgetter('owner'))
    for face in bnd_faces:
        face.neighbour = -1
    faces = int_faces + bnd_faces

    note = 'nPoints: {} nCells: {} nFaces: {} nInternalFaces: {}'.format(
        simra.npts, simra.nelems, len(faces), len(int_faces)
    )

    foam_points(os.path.join(outdir, 'points'), simra.coords)
    foam_faces(os.path.join(outdir, 'faces'), faces)
    foam_labels(os.path.join(outdir, 'owner'), faces, 'owner', note=note)
    foam_labels(os.path.join(outdir, 'neighbour'), faces, 'neighbour', note=note)
    foam_boundaries(os.path.join(outdir, 'boundary'), len(int_faces), bnd_faces, boundary_names)

    timedir = os.path.join(outdir, str(simra.time))
    if not os.path.exists(timedir):
        os.makedirs(timedir)

    foam_internalfield(os.path.join(timedir, 'u'), 'u', simra['u'], boundaries=('inflow',), faces=faces)
    foam_internalfield(os.path.join(timedir, 'ps'), 'ps', simra['ps'], boundaries=('outflow',), faces=faces)
    foam_internalfield(os.path.join(timedir, 'tk'), 'tk', simra['tk'], boundaries=('inflow',), faces=faces)
    foam_internalfield(os.path.join(timedir, 'td1'), 'td1', simra['td'], boundaries=('inflow',), faces=faces)
    foam_internalfield(os.path.join(timedir, 'vtef'), 'vtef', simra['vtef'])
    foam_internalfield(os.path.join(timedir, 'pt'), 'pt', simra['pt'])
    foam_internalfield(os.path.join(timedir, 'pts1'), 'pts1', simra['pts'])
    foam_internalfield(os.path.join(timedir, 'rho'), 'rho', simra['rho'])
    foam_internalfield(os.path.join(timedir, 'rhos'), 'rhos', simra['rhos'])


@click.command()
@reader_args
@click.option('--out', 'outfile')
def main(outfile, **kwargs):
    simra = Simra(**kwargs, require_data=True)

    if outfile is None and simra.resfile is not None:
        outfile, _ = os.path.splitext(simra.resfile)
    elif outfile is None:
        outfile = 'cont'

    if os.path.exists(outfile) and not os.path.isdir(outfile):
        print("Output location {} already exists, and is not a directory".format(outfile), file=sys.stderr)
        sys.exit(1)
    elif not os.path.exists(outfile):
        os.makedirs(outfile)

    convert_grid(simra, outfile)
