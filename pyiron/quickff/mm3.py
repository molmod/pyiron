import numpy as np
from molmod.periodic import periodic as pt
from molmod.units import angstrom

from yaff import System, log
log.set_level(0)

# This code is largely based on the MM3 code from Juul De Vos

def get_mm3_ff(path):
    mm3_ff = {}
    with open(path, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) > 0 and data[0] == 'vdw':
                mm3_number = int(data[1])
                epsilon = float(data[2])
                sigma = float(data[3])
                mm3_ff[mm3_number] = [epsilon, sigma]
    #print('WARNING: the units of the MM3 parameters in the mm3 dictionary are not atomic units, but kcalmol (epsilon) and angstrom (sigma)')
    return mm3_ff

def get_mm3_indices(fn_sys, fn_xyz, periodic_structure=False):
    sys = System.from_file(fn_sys)
    mm3_ffatypes = {ffatype: None for ffatype in sys.ffatypes}

    # Read xyz file and make some basic sanity checks
    with open(fn_xyz, 'r') as f:
        data = f.readline().strip().split()
        length = int(data[0])
        if periodic_structure:
            assert length/sys.natom==27
            index_range = range(13*sys.natom,14*sys.natom)# assume this is the central cell in a 3x3x3 supercell
        else:
            assert length==sys.natom
            index_range = range(sys.natom)

        lines = f.readlines()

    # Read data from xyz file
    symbols = []
    mm3_ind = []
    pos = np.zeros((sys.natom,3))

    for n,i in enumerate(index_range):
        data = lines[i].strip().split()
        symbols.append(str(data[1]))
        x = float(data[2])*angstrom
        y = float(data[3])*angstrom
        z = float(data[4])*angstrom
        pos[n] = np.array([x, y, z])
        mm3_ind.append(int(data[5]))

    # Center pos
    com = np.sum(pos,axis=0)/sys.natom
    sys_com = np.sum(sys.pos,axis=0)/sys.natom
    pos = pos-com
    sys_pos = sys.pos-sys_com

    # Identify each ffatype with an mm3 type
    for n,_ in enumerate(symbols):
        for j in range(sys.natom):
            if np.linalg.norm(pos[n] - sys_pos[j]) < 1e-4: # only check those atoms that are in the chk file
                assert pt[symbols[n]].number == sys.numbers[j]
                ffatype = sys.get_ffatype(j)
                if mm3_ffatypes.get(ffatype) == None:
                    mm3_ffatypes[ffatype] = mm3_ind[n]
                else:
                    if not mm3_ffatypes.get(ffatype) == mm3_ind[n]:
                        raise RuntimeError('Different MM3 parameters recognized for ffatype {}'.format(ffatype))

    if '/' in fn_sys:
        sys_name = fn_sys.split('/', 1)[1]
    else:
        sys_name = fn_sys
    #print('MM3 ffatypes for System {} found'.format(sys_name))
    return mm3_ffatypes

def write_mm3_pars(mm3_ff, mm3_ffatypes, fn_out):
    max_length = max([len(t) for t in mm3_ffatypes])
    with open(fn_out, 'w') as f:
        f.write("""# van der Waals
#==============
# The following mathemetical form is supported:
#  - MM3:   EPSILON*(1.84e5*exp(-12*r/SIGMA)-2.25*(SIGMA/r)^6)
#  - LJ:    4.0*EPSILON*((SIGMA/r)^12 - (SIGMA/r)^6)

""")
        f.write('MM3:UNIT SIGMA angstrom\n')
        f.write('MM3:UNIT EPSILON kcalmol\n')
        f.write('MM3:SCALE 1 0.0\n')
        f.write('MM3:SCALE 2 0.0\n')
        f.write('MM3:SCALE 3 1.0\n')
        f.write('\n')
        f.write("""# ---------------------------------------------
# KEY      ffatype    SIGMA  EPSILON  ONLYPAULI
# ---------------------------------------------
""")
        for ffatype, mm3_index in mm3_ffatypes.items():
            sigma, epsilon = mm3_ff[mm3_index]
            f.write('MM3:PARS    {:>{w}}    {:4.2f}    {:5.3f}    0 \n'.format(ffatype, sigma, epsilon, w=max_length))
    #print('MM3 parameters written to {}'.format(fn_out))
