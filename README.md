## Installation

Best is to use pip to install to your user directory. Navigate to the
source folder and use:

    pip install --user .

This will install the `simra2vtk` executable to `~/.local/bin`. Make
sure that this folder is in your path.

Note that some distributions may use *pip3* as the name for the *pip*
executable tied to Python 3. In that case:

    pip3 install --user .

## Usage

Full usage is as follows

    simra2vtk --mesh mesh.dat --res cont.res --out cont.vtk
    simra2openfoam --mesh mesh.dat --res cont.res --out cont

The `--mesh` and `--res` options may be omitted, in which case the
default values (as above) apply. The `--out` option may also be
omitted, in which case the output file will have the same name as the
result file, with a *.vtk* extension (in case of *simra2vtk*) or
without the extension (in case of *simra2openfoam*).
