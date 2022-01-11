# coding: utf-8
from pyiron.base.generic.parameters import GenericParameters
from pyiron.atomistics.structure.atoms import Atoms
from pyiron.atomistics.job.atomistic import AtomisticGenericJob, GenericOutput
from pyiron.base.settings.generic import Settings


from yaff import System, log, ForceField
from quickff.tools import set_ffatypes
log.set_level(log.silent)
from molmod.units import *
from molmod.constants import *
import subprocess

import os, posixpath, numpy as np, h5py, matplotlib.pyplot as pp, stat, warnings


s = Settings()

def get_plumed_path():
    for resource_path in s.resource_paths:
        p = os.path.join(resource_path, "yaff", "bin", "plumed.sh")
        if os.path.exists(p):
            return p



def write_chk(input_dict,working_directory='.'):
    # collect data and initialize Yaff system
    cell = None
    if 'cell' in input_dict.keys() and input_dict['cell'] is not None and input_dict['cell'].volume > 0:
        cell = input_dict['cell']*angstrom
    system = System(input_dict['numbers'], input_dict['pos']*angstrom, ffatypes=input_dict['ffatypes'], ffatype_ids=input_dict['ffatype_ids'], bonds=input_dict['bonds'], rvecs=cell, masses=input_dict['masses']*amu)

    if input_dict['bonds'] is None:
        system.detect_bonds()
        print('Warning: no bonds could be read and were automatically detected.')
    # write dictionary to MolMod CHK file
    system.to_file(posixpath.join(working_directory,'system.chk'))

def write_pars(input_dict,working_directory='.'):
    with open(posixpath.join(working_directory,'pars.txt'), 'w') as f:
        for line in input_dict['ffpars']:
            f.write(line)

common = """#! /usr/bin/python

from molmod.units import *
from yaff import *
import h5py, numpy as np

#Setting up system and force field
system = System.from_file('system.chk')
ff = ForceField.generate(system, 'pars.txt', rcut={rcut}*angstrom, alpha_scale={alpha_scale}, gcut_scale={gcut_scale}, smooth_ei={smooth_ei})

#Setting up output
f = h5py.File('output.h5', mode='w')
hdf5 = HDF5Writer(f, step={h5step})
r = h5py.File('restart.h5', mode='w')
restart = RestartWriter(r, step=10000)
hooks = [hdf5, restart]

#Setting up simulation
"""

tail = """
"""

def write_yopt(input_dict,working_directory='.'):
    body = common.format(
        rcut=input_dict['rcut']/angstrom, alpha_scale=input_dict['alpha_scale'],
        gcut_scale=input_dict['gcut_scale'], smooth_ei=input_dict['smooth_ei'],
        h5step=1,
    )
    body += "dof = CartesianDOF(ff, gpos_rms={gpos_rms}, dpos_rms={dpos_rms})".format(
        gpos_rms=input_dict['gpos_rms'],dpos_rms=input_dict['dpos_rms']
    )
    body += """
opt = CGOptimizer(dof, hooks=[hdf5])
opt.run({nsteps})
system.to_file('opt.chk')
""".format(nsteps=input_dict['nsteps'])
    body+= tail
    with open(posixpath.join(working_directory,'yscript.py'), 'w') as f:
        f.write(body)

def write_yopt_cell(input_dict,working_directory='.'):
    body = common.format(
        rcut=input_dict['rcut']/angstrom, alpha_scale=input_dict['alpha_scale'],
        gcut_scale=input_dict['gcut_scale'], smooth_ei=input_dict['smooth_ei'],
        h5step=1,
    )
    body += "dof = StrainCellDOF(ff, gpos_rms={gpos_rms}, dpos_rms={dpos_rms}, grvecs_rms={grvecs_rms}, drvecs_rms={drvecs_rms}, do_frozen=False)".format(
        gpos_rms=input_dict['gpos_rms'],dpos_rms=input_dict['dpos_rms'],
        grvecs_rms=input_dict['grvecs_rms'],drvecs_rms=input_dict['drvecs_rms']
    )
    body += """
opt = CGOptimizer(dof, hooks=[hdf5])
opt.run({nsteps})
system.to_file('opt.chk')
""".format(nsteps=input_dict['nsteps'])
    body+= tail
    with open(posixpath.join(working_directory,'yscript.py'), 'w') as f:
        f.write(body)

def __write_ysp(input_dict,working_directory='.'):
    """Deprecated due to not making a trajectory group"""
    body = common.format(
        rcut=input_dict['rcut']/angstrom, alpha_scale=input_dict['alpha_scale'],
        gcut_scale=input_dict['gcut_scale'], smooth_ei=input_dict['smooth_ei'],
        h5step=1,
    )
    body +="""
energy = ff.compute()
system.to_hdf5(f)
f['system/energy'] = energy
"""
    body+= tail
    with open(posixpath.join(working_directory,'yscript.py'), 'w') as f:
        f.write(body)

def write_ysp(input_dict,working_directory='.'):
    body = common.format(
        rcut=input_dict['rcut']/angstrom, alpha_scale=input_dict['alpha_scale'],
        gcut_scale=input_dict['gcut_scale'], smooth_ei=input_dict['smooth_ei'],
        h5step=1,
    )
    body += "dof = CartesianDOF(ff, gpos_rms={gpos_rms}, dpos_rms={dpos_rms})".format(
        gpos_rms=input_dict['gpos_rms'],dpos_rms=input_dict['dpos_rms']
    )
    body += """
opt = CGOptimizer(dof, hooks=[hdf5])
opt.run(0) # just caluculate the energy
"""
    body+= tail
    with open(posixpath.join(working_directory,'yscript.py'), 'w') as f:
        f.write(body)

def write_yhess(input_dict,working_directory='.'):
    body = common.format(
        rcut=input_dict['rcut']/angstrom, alpha_scale=input_dict['alpha_scale'],
        gcut_scale=input_dict['gcut_scale'], smooth_ei=input_dict['smooth_ei'],
        h5step=1,
    )
    body +="""dof = CartesianDOF(ff)

gpos  = np.zeros((len(system.numbers), 3), float)
vtens = np.zeros((3, 3), float)
energy = ff.compute(gpos, vtens)
hessian = estimate_hessian(dof, eps={hessian_eps})

system.to_hdf5(f)
f['system/energy'] = energy
f['system/gpos'] = gpos
f['system/hessian'] = hessian""".format(hessian_eps=input_dict['hessian_eps'])
    body+= tail
    with open(posixpath.join(working_directory,'yscript.py'), 'w') as f:
        f.write(body)

def write_ynve(input_dict,working_directory='.'):
    body = common.format(
        rcut=input_dict['rcut']/angstrom, alpha_scale=input_dict['alpha_scale'],
        gcut_scale=input_dict['gcut_scale'], smooth_ei=input_dict['smooth_ei'],
        h5step=input_dict['h5step'],
    )
    if input_dict['enhanced'] is not None:
        body += """
plumed = ForcePartPlumed(ff.system, fn='plumed.dat')
ff.add_part(plumed)
hooks.append(plumed)
"""
    body += """
hooks.append(VerletScreenLog(step=1000))
md = VerletIntegrator(ff, {timestep}*femtosecond, hooks=hooks)
md.run({nsteps})
""".format(timestep=input_dict['timestep']/femtosecond, nsteps=input_dict['nsteps'])
    body+= tail
    with open(posixpath.join(working_directory,'yscript.py'), 'w') as f:
        f.write(body)

def write_ynvt(input_dict,working_directory='.'):
    body = common.format(
        rcut=input_dict['rcut']/angstrom, alpha_scale=input_dict['alpha_scale'],
        gcut_scale=input_dict['gcut_scale'], smooth_ei=input_dict['smooth_ei'],
        h5step=input_dict['h5step'],
    )
    if input_dict['enhanced'] is not None:
        body += """
plumed = ForcePartPlumed(ff.system, fn='plumed.dat')
ff.add_part(plumed)
hooks.append(plumed)
"""
    body += """
temp = {temp}*kelvin
thermo = NHCThermostat(temp, timecon={timecon_thermo}*femtosecond)
hooks.append(thermo)

hooks.append(VerletScreenLog(step=1000))
md = VerletIntegrator(ff, {timestep}*femtosecond, hooks=hooks)
md.run({nsteps})
""".format(
        temp=input_dict['temp']/kelvin,timestep=input_dict['timestep']/femtosecond,
        timecon_thermo=input_dict['timecon_thermo']/femtosecond, nsteps=input_dict['nsteps']
    )
    body+= tail
    with open(posixpath.join(working_directory,'yscript.py'), 'w') as f:
        f.write(body)

def write_ynpt(input_dict,working_directory='.'):
    body = common.format(
        rcut=input_dict['rcut']/angstrom, alpha_scale=input_dict['alpha_scale'],
        gcut_scale=input_dict['gcut_scale'], smooth_ei=input_dict['smooth_ei'],
        h5step=input_dict['h5step'],
    )
    if input_dict['enhanced'] is not None:
        body += """
plumed = ForcePartPlumed(ff.system, fn='plumed.dat')
ff.add_part(plumed)
hooks.append(plumed)
"""
    body += """
temp = {temp}*kelvin
press = {press}*bar
thermo = NHCThermostat(temp, timecon={timecon_thermo}*femtosecond)
baro = MTKBarostat(ff, temp, press, timecon={timecon_baro}*femtosecond)
TBC = TBCombination(thermo, baro)
hooks.append(TBC)

hooks.append(VerletScreenLog(step=1000))
md = VerletIntegrator(ff, {timestep}*femtosecond, hooks=hooks)
md.run({nsteps})
""".format(
        temp=input_dict['temp']/kelvin,timestep=input_dict['timestep']/femtosecond,
        press=input_dict['press']/bar,timecon_thermo=input_dict['timecon_thermo']/femtosecond,
        timecon_baro=input_dict['timecon_baro']/femtosecond, nsteps=input_dict['nsteps']
    )
    body+= tail
    with open(posixpath.join(working_directory,'yscript.py'), 'w') as f:
        f.write(body)


def write_scan(input_dict,working_directory='.'):
    # Write target positions for each frame to the corresponding h5 file
    scan_data = input_dict['scan']
    h5_path = posixpath.join(working_directory,'structures.h5')
    h5 = h5py.File(h5_path,mode='w')
    grp = h5.create_group("scan")
    grp['positions'] = scan_data['positions']
    if not np.isnan(scan_data['rvecs']).any():
        grp['rvecs'] = scan_data['rvecs']
    h5.close()

    # Write script file which reads and adapts h5 file with energy of each snapshot
    body = """#! /usr/bin/python

from molmod.units import *
from yaff import *
import h5py, numpy as np

# Define scan iterator
class ScanIntegrator(Iterative):
    default_state = [
        AttributeStateItem('counter'),
        AttributeStateItem('epot'),
        PosStateItem(),
        VolumeStateItem(),
        CellStateItem(),
        EPotContribStateItem(),
    ]

    log_name = 'SCAN'

    def __init__(self, ff, h5_path, state=None, hooks=None, counter0=1):
        self.h5 = h5py.File(h5_path,mode='r')
        self.do_cell = 'scan/rvecs' in self.h5
        self.assign_structure(ff, self.h5, self.do_cell, 0)
        Iterative.__init__(self, ff, state, hooks, counter0)

    @staticmethod
    def assign_structure(ff,h5,do_cell,counter):
        ff.update_pos(h5['scan/positions'][counter])
        if do_cell:
            ff.update_rvecs(h5['scan/rvecs'][counter])

    def initialize(self):
        self.epot = self.ff.compute()
        Iterative.initialize(self)

    def propagate(self):
        # counter0 should start from 1 since the counter is updated after evaluation
        self.assign_structure(self.ff, self.h5, self.do_cell, self.counter)
        self.epot = self.ff.compute()
        Iterative.propagate(self)

    def finalize(self):
        self.h5.close()


#Setting up system and force field
system = System.from_file('system.chk')
ff = ForceField.generate(system, 'pars.txt', rcut={rcut}*angstrom, alpha_scale={alpha_scale}, gcut_scale={gcut_scale}, smooth_ei={smooth_ei})

#Setting up output
f = h5py.File('output.h5', mode='w')
hdf5 = HDF5Writer(f, step=1)

#Setting up simulation
scan = ScanIntegrator(ff,'structures.h5',hooks=[hdf5])
scan.run({steps})

    """.format(
        rcut=input_dict['rcut']/angstrom, alpha_scale=input_dict['alpha_scale'],
        gcut_scale=input_dict['gcut_scale'], smooth_ei=input_dict['smooth_ei'],
        steps=scan_data['positions'].shape[0]-1
    )


    with open(posixpath.join(working_directory,'yscript.py'), 'w') as f:
        f.write(body)

def write_plumed_enhanced(input_dict,working_directory='.'):
    #make copy of input_dict['enhanced'] that includes lower case definitions
    #(allowing for case insenstive definition of input_dict['enhanced'])
    enhanced = {}
    for key, value in input_dict['enhanced'].items():
        enhanced[key] = value
        enhanced[key.lower()] = value
    #write plumed.dat file
    with open(posixpath.join(working_directory, 'plumed.dat'), 'w') as f:
        #set units to atomic units
        f.write('UNITS LENGTH=Bohr ENERGY=kj/mol TIME=fs \n')
        #define ics
        for i, kind in enumerate(enhanced['ickinds']):
            if isinstance(kind, bytes):
                kind = kind.decode()
            if len( enhanced['icindices'][i] > 0):
                f.write('ic%i: %s ATOMS=%s \n' %(i, kind.upper(), ','.join([str(icidx) for icidx in enhanced['icindices'][i]])))
            else:
                f.write('ic%i: %s \n' %(i, kind.upper()))

        #define metadynamics run
        if 'sigma' in enhanced.keys():
            if len(enhanced['sigma'])==1:
                sigma = '%.2f' %(enhanced['sigma'])
            else:
                assert len(enhanced['sigma'])>1
                sigma = ','.join(['%.2f' %s for s in enhanced['sigma']])
            if len(enhanced['height'])==1:
                height = '%.2f' %(enhanced['height']/kjmol)
            else:
                assert len(enhanced['height'])>1
                height = ','.join(['%.2f' %h/kjmol for h in enhanced['height']])
            f.write('metad: METAD ARG=%s SIGMA=%s HEIGHT=%s PACE=%i FILE=%s \n' %(
                ','.join([ 'ic%i' %i for i in range(len(enhanced['ickinds'])) ]),
                sigma, height, enhanced['pace'], enhanced['file']
            ))
            #setup printing of colvar
            f.write('PRINT ARG=%s,metad.bias FILE=%s STRIDE=%i \n' %(
                ','.join([ 'ic%i' %i for i in range(len(enhanced['ickinds'])) ]),
                enhanced['file_colvar'], enhanced['stride']
            ))

        # define umbrella sampling run
        if 'kappa' in enhanced.keys():
            kappa = ','.join(['%.2f' %(float(s)/kjmol) for s in enhanced['kappa']])
            loc = ','.join(['%.2f' %(float(h)) for h in enhanced['loc']])

            f.write('umbrella: RESTRAINT ARG=%s KAPPA=%s AT=%s \n' %(
                ','.join([ 'ic%i' %i for i in range(len(enhanced['ickinds'])) ]),
                kappa, loc
            ))
            #setup printing of colvar
            f.write('PRINT ARG=%s,umbrella.bias FILE=%s STRIDE=%i \n' %(
                ','.join([ 'ic%i' %i for i in range(len(enhanced['ickinds'])) ]),
                enhanced['file_colvar'], enhanced['stride']
            ))

def hdf2dict(h5):
    hdict = {}
    hdict['structure/numbers'] = h5['system/numbers'][:]
    hdict['structure/masses'] = h5['system/masses'][:]
    hdict['structure/ffatypes'] = h5['system/ffatypes'][:]
    hdict['structure/ffatype_ids'] = h5['system/ffatype_ids'][:]

    if 'charges' in h5['system'].keys():
        hdict['structure/charges'] = h5['system/charges'][:]
    else:
        warnings.warn("I could not read any charges from the system file. This could break some functionalities!")

    if 'energy' in h5['system'].keys():
        hdict['generic/energy_pot'] = h5['system/energy'][()]/electronvolt
    if 'trajectory' in h5.keys() and 'pos' in h5['trajectory'].keys():
        hdict['generic/positions'] = h5['trajectory/pos'][:]/angstrom
    else:
        hdict['generic/positions'] = np.array([h5['system/pos'][:]/angstrom])
    if 'trajectory' in h5.keys() and 'pos' in h5['trajectory'].keys():
        hdict['structure/positions'] = h5['trajectory/pos'][-1]/angstrom
    else:
        hdict['structure/positions'] = np.array([h5['system/pos'][:]/angstrom])
    if 'trajectory' in h5.keys() and 'cell' in h5['trajectory']:
        hdict['generic/cells'] = h5['trajectory/cell'][:]/angstrom
    elif 'rvecs' in h5['system'].keys():
        hdict['generic/cells'] = np.array([h5['system/rvecs'][:]/angstrom])
    else:
        hdict['generic/cells'] = None
    if 'trajectory' in h5.keys():
        if 'counter' in h5['trajectory'].keys():
            hdict['generic/steps'] = h5['trajectory/counter'][:]
        if 'time' in h5['trajectory'].keys():
            hdict['generic/time'] = h5['trajectory/time'][:]/femtosecond
        if 'volume' in h5['trajectory']:
            hdict['generic/volume'] = h5['trajectory/volume'][:]/angstrom**3
        if 'epot' in h5['trajectory'].keys():
            hdict['generic/energy_pot'] = h5['trajectory/epot'][:]/electronvolt
        if 'ekin' in h5['trajectory'].keys():
            hdict['generic/energy_kin'] = h5['trajectory/ekin'][:]/electronvolt
        if 'temp' in h5['trajectory'].keys():
            hdict['generic/temperature'] = h5['trajectory/temp'][:]
        if 'etot' in h5['trajectory'].keys():
            hdict['generic/energy_tot'] = h5['trajectory/etot'][:]/electronvolt
        if 'econs' in h5['trajectory'].keys():
            hdict['generic/energy_cons'] = h5['trajectory/econs'][:]/electronvolt
        if 'epot_contribs' in h5['trajectory'].keys():
            hdict['generic/epot_contribs'] = h5['trajectory/epot_contribs'][:]/electronvolt
            hdict['generic/epot_contrib_names'] = h5['trajectory/'].attrs.get('epot_contrib_names')
        if 'press' in h5['trajectory'].keys():
            hdict['generic/pressure'] = h5['trajectory/press'][:]/(1e9*pascal)
        if 'gradient' in h5['trajectory'].keys():
            hdict['generic/forces'] = -h5['trajectory/gradient'][:]/(electronvolt/angstrom)
        if 'vel' in h5['trajectory'].keys():
            hdict['generic/velocities'] = h5['trajectory/vel'][:]/(angstrom/femtosecond)
        if 'dipole' in h5['trajectory'].keys():
            hdict['generic/dipole'] = h5['trajectory/dipole'][:]/(angstrom) # unit is e*A
        if 'dipole_vel' in h5['trajectory'].keys():
            hdict['generic/dipole_velocities'] = h5['trajectory/dipole_vel'][:]/angstrom/(angstrom/femtosecond) # unit is (e*A)*(A/fs)

    if 'hessian' in h5['system'].keys():
        hdict['generic/forces'] = -h5['system/gpos'][:]/(electronvolt/angstrom)
        hdict['generic/hessian'] = h5['system/hessian'][:]/(electronvolt/angstrom**2)
    return hdict


def collect_output(output_file):
    # this routine basically reads and returns the output HDF5 file produced by Yaff
    # read output
    h5 = h5py.File(output_file, mode='r')
    # translate to dict
    output_dict = hdf2dict(h5)
    return output_dict


class YaffInput(GenericParameters):
    def __init__(self, input_file_name=None):
        super(YaffInput, self).__init__(input_file_name=input_file_name,table_name="input_inp",comment_char="#")

    def load_default(self):
        '''
        Loading the default settings for the input file.
        '''
        input_str = """\
rcut 28.345892008818783 #(FF) real space cutoff
alpha_scale 3.2 #(FF) scale for ewald alpha parameter
gcut_scale 1.5 #(FF) scale for ewald reciprocal cutoff parameter
smooth_ei True #(FF) smoothen cutoff for real space electrostatics
gpos_rms 1e-8 #(OPT) convergence criterion for RMS of gradients towards atomic coordinates
dpos_rms 1e-6 #(OPT) convergence criterion for RMS of differences of atomic coordinates
grvecs_rms 1e-8 #(OPT) convergence criterion for RMS of gradients towards cell parameters
drvecs_rms 1e-6 #(OPT) convergence criterion for RMS of differences of cell parameters
hessian_eps 1e-3 #(HESS) step size in finite differences for numerical derivatives of the forces
timestep 41.341373336646825 #(MD) time step for verlet scheme
temp None #(MD) temperature
press None #(MD) pressure
timecon_thermo 4134.137333664683 #(MD) timeconstant for thermostat
timecon_baro 41341.37333664683 #(MD) timeconstant for barostat
nsteps 1000 #(GEN) number of steps for opt or md
h5step 5 #(GEN) stores system properties every h5step
"""
        self.load_string(input_str)


class Yaff(AtomisticGenericJob):
    def __init__(self, project, job_name):
        super(Yaff, self).__init__(project, job_name)
        self.__name__ = "Yaff"
        self._executable_activate(enforce=True)
        self.input = YaffInput()
        self.bonds = None
        self.ffatypes = None
        self.ffatype_ids = None
        self.enhanced = None
        self.scan = None


    def calc_minimize(self, cell=False, gpos_tol=1e-8, dpos_tol=1e-6, grvecs_tol=1e-8, drvecs_tol=1e-6, max_iter=1000, n_print=5):
        '''
            Set up an optimization calculation.

            **Arguments**

            cell (bool): Set True if the cell also has to be optimized
            gpos_tol (float): Convergence criterion for RMS of gradients towards atomic coordinates
            dpos_tol (float): Convergence criterion for RMS of differences of atomic coordinates
            grvecs_tol (float): Convergence criterion for RMS of gradients towards cell parameters
            drvecs_tol (float): Convergence criterion for RMS of differences of cell parameters
            max_iter (int): Maximum number of optimization steps
            n_print (int):  Print frequency
        '''
        if cell:
            self.input['jobtype'] = 'opt_cell'
        else:
            self.input['jobtype'] = 'opt'
        self.input['nsteps']     = int(max_iter)
        self.input['h5step']     = int(n_print)
        self.input['gpos_rms']   = gpos_tol
        self.input['dpos_rms']   = dpos_tol
        self.input['grvecs_rms'] = grvecs_tol
        self.input['drvecs_rms'] = drvecs_tol

        super(Yaff, self).calc_minimize(max_iter=max_iter, n_print=n_print)


    def calc_static(self):
        '''
            Set up a static force field calculation.
        '''

        self.input['jobtype'] = 'sp'
        super(Yaff, self).calc_static()


    def calc_md(self, temperature=None, pressure=None, nsteps=1000, time_step=1.0*femtosecond, n_print=5,
                timecon_thermo=100.0*femtosecond, timecon_baro=1000.0*femtosecond):

        '''
            Set an MD calculation within Yaff. Nosé Hoover chain is used by default.

            **Arguments**

            temperature (None/float): Target temperature. If set to None, an NVE calculation is performed.
                                      It is required when the pressure is set
            pressure (None/float): Target pressure. If set to None, an NVE or an NVT calculation is performed.
            nsteps (int): Number of md steps
            time_step (float): Step size between two steps.
            n_print (int):  Print frequency
            timecon_thermo (float): The time associated with the thermostat adjusting the temperature.
            timecon_baro (float): The time associated with the barostat adjusting the temperature.

        '''
        self.input['temp'] = temperature
        self.input['press'] = pressure
        self.input['nsteps'] = int(nsteps)
        self.input['timestep'] = time_step
        self.input['h5step'] = int(n_print)
        self.input['timecon_thermo'] = timecon_thermo
        self.input['timecon_baro'] = timecon_baro

        if temperature is None:
            self.input['jobtype'] = 'nve'
        else:
            if pressure is None:
                self.input['jobtype'] = 'nvt'
            else:
                self.input['jobtype'] = 'npt'

        super(Yaff, self).calc_md(temperature=temperature, pressure=pressure, n_ionic_steps=nsteps,
                                  time_step=time_step, n_print=n_print,
                                  temperature_damping_timescale=timecon_thermo,
                                  pressure_damping_timescale=timecon_baro)

    def calc_scan(self, grid, adapt_structure=None, structures=None):
        '''
            Set a scan calculation within Yaff.

            **Arguments**
            grid (list/array): used as reference, and as input for the adapt structure function if provided
            adapt_structure (function): function which takes a structure object and target value used to identify the structure
            structures (list of structure objects): instead of the adapt_structure function,
                                                    you can also provide the list of structures separately

            **Example**

            grid = np.arange(0,181)
            def adapt_structure(structure,val):
                s = structure.copy()
                s.set_dihedral(0,1,2,3,angle=val)
                return s

            job.structure = ref_structure

            Case 1: using adapt_structure
            job.calc_scan(grid,adapt_structure=adapt_structure)

            Case 2: manually specifying structures
            structures = [adapt_structure(job.structure,val) for val in grid]
            job.calc_scan(grid,structures=structures)

        '''

        self.input['jobtype'] = 'scan'

        positions = np.zeros((len(grid),*self.structure.positions.shape))
        rvecs = np.zeros((len(grid),3,3))

        assert bool(adapt_structure is None) ^ bool(structures is None) # XOR operator (a and not b) or (b and not a)

        for n,val in enumerate(grid):
            if adapt_structure is not None:
                adapted_structure = adapt_structure(self.structure,val)
                positions[n] = adapted_structure.positions
                rvecs[n] = adapted_structure.cell if adapted_structure.cell.volume>0 else np.nan
            else:
                positions[n] = structures[n].positions
                rvecs[n] = structures[n].cell if structures[n].cell.volume>0 else np.nan

        self.scan = {
        'grid'       : grid,
        'positions'  : positions * angstrom,
        'rvecs'      : rvecs * angstrom
        }

    def load_chk(self, fn, bonds_dict=None):
        '''
            Load the atom types, atom type ids and structure by reading a .chk file.

            **Arguments**

            fn      the path to the chk file

            bonds_dict
                    Specify custom threshold for certain pairs of elements. This
                    must be a dictionary with ((num0, num1), threshold) as items.
                    see Yaff.System.detect_bonds() for more information

        '''

        system = System.from_file(fn)
        if len(system.pos.shape)!=2:
            raise IOError("Something went wrong, positions in CHK file %s should have Nx3 dimensions" %fn)
        self.load_system(system,bonds_dict=bonds_dict)

    def load_system(self,system,bonds_dict=None):
        if system.masses is None:
            system.set_standard_masses()
        if system.cell.rvecs is not None and len(system.cell.rvecs)>0:
            self.structure = Atoms(
                positions=system.pos.copy()/angstrom,
                numbers=system.numbers,
                masses=system.masses/amu,
                cell=system.cell.rvecs/angstrom,
            )
        else:
            self.structure = Atoms(
                positions=system.pos.copy()/angstrom,
                numbers=system.numbers,
                masses=system.masses/amu,
            )
        if system.ffatypes is not None:
            self.ffatypes = system.ffatypes
        if system.ffatype_ids is not None:
            self.ffatype_ids = system.ffatype_ids
        if system.bonds is None or bonds_dict is not None:
            # if bonds is None always detect bonds, if they are provided but a bond_dict is specified
            # it is implied the user wants to detect new bonds
            system.detect_bonds(bonds_dict)
        self.bonds = system.bonds

    def set_mtd(self, ics, height, sigma, pace, fn='HILLS', fn_colvar='COLVAR', stride=10, temp=300):
        '''
            Setup a Metadynamics run using PLUMED along the internal coordinates
            defined in the ICs argument.

            **Arguments**

            ics     a list of entries defining each internal coordinate. Each
                    of these entries should be of the form (kind, [i, j, ...])

                    Herein, kind defines the kind of IC as implemented in PLUMED:

                        i.e. distance, angle, torsion, volume, cell, ... see
                        https://www.plumed.org/doc-v2.5/user-doc/html/_colvar.html
                        for more information).

                    and [i, j, ...] is a list of atom indices, starting from 0, involved in this
                    IC. If no atom indices are required for e.g. volume, provide an empty list.

                    An example for a 1D metadynamica using the distance between
                    atoms 2 and 4:

                        ics = [('distance', [2,4])]

            height  the height of the Gaussian hills, can be a single value
                    (the gaussian hills for each IC have identical height) or
                    a list of values, one for each IC defined.

            sigmas  the sigma of the Gaussian hills, can be a single value
                    (the gaussian hills for each IC have identical height) or
                    a list of values, one for each IC defined.

            pace    the number of steps after which the gaussian hills are
                    updated.

            fn      the PLUMED output file for the gaussian hills

            fn_colvar
                    the PLUMED output file for logging of collective variables

            stride  the number of steps after which the internal coordinate
                    values and bias are printed to the COLVAR output file.

            temp    the system temperature
        '''
        for l in ics:
            assert len(l)==2
            assert isinstance(l[0], str)
            assert isinstance(l[1], list) or isinstance(l[1], tuple)
        ickinds = np.array([ic[0] for ic in ics],dtype='S22')
        icindices = np.array([np.array(ic[1])+1 for ic in ics]) # plumed starts counting from 1
        if not isinstance(height,list) and not isinstance(height,np.ndarray):
            height = np.array([height])
        if not isinstance(sigma,list) and not isinstance(sigma,np.ndarray):
            sigma = np.array([sigma])
        self.enhanced= {
            'ickinds': ickinds, 'icindices': icindices, 'height': height, 'sigma': sigma, 'pace': pace,
            'file': fn, 'file_colvar': fn_colvar, 'stride': stride, 'temp': temp
        }


    def set_us(self, ics, kappa, loc, fn_colvar='COLVAR', stride=10, temp=300):
        '''
            Setup an Umbrella sampling run using PLUMED along the internal coordinates
            defined in the ICs argument.

            **Arguments**

            ics     a list of entries defining each an internal coordinate. Each
                    of these entries should be of the form (kind, [i, j, ...])

                    Herein, kind defines the kind of IC as implemented in PLUMED:

                        i.e. distance, angle, torsion, volume, cell, ... see
                        https://www.plumed.org/doc-v2.5/user-doc/html/_colvar.html
                        for more information).

                    and [i, j, ...] is a list of atom indices, starting from 0, involved in this
                    IC. If no atom indices are required for e.g. volume, provide an empty list.

                    An example for a 1D metadynamica using the distance between
                    atoms 2 and 4:

                        ics = [('distance', [2,4])]

            kappa   the value of the force constant of the harmonic bias potential,
                    can be a single value (the harmonic bias potential for each IC has identical kappa)
                    or a list of values, one for each IC defined.

            loc     the location of the umbrella
                    (should have a length equal to the number of ICs)

            fn_colvar
                    the PLUMED output file for logging of collective variables

            stride  the number of steps after which the internal coordinate
                    values and bias are printed to the COLVAR output file.

            temp    the system temperature
        '''
        for l in ics:
            assert len(l)==2
            assert isinstance(l[0], str)
            assert isinstance(l[1], list) or isinstance(l[1], tuple)
        ickinds = np.array([ic[0] for ic in ics],dtype='S22')
        icindices = np.array([np.array(ic[1])+1 for ic in ics]) # plumed starts counting from 1
        if not isinstance(kappa,list) and not isinstance(kappa,np.ndarray):
            kappa = np.array([kappa])
        if not isinstance(loc,list) and not isinstance(loc,np.ndarray):
            loc = np.array([loc])
        assert len(loc)==len(ics)
        self.enhanced= {
            'ickinds': ickinds, 'icindices': icindices, 'kappa': kappa, 'loc': loc,
            'file_colvar': fn_colvar, 'stride': stride, 'temp': temp
        }

    def detect_ffatypes(self, ffatypes=None, ffatype_rules=None, ffatype_level=None, bonds_dict=None):
        '''
            Define atom types by explicitely giving them through the
            ffatypes keyword, defining atype rules using the ATSELECT
            language implemented in Yaff (see the Yaff documentation at
            http://molmod.github.io/yaff/ug_atselect.html) or by specifying
            the ffatype_level employing the built-in routine in QuickFF.
        '''
        numbers = self.structure.get_atomic_numbers()
        if self.structure.cell is not None and self.structure.cell.volume > 0:
            system = System(numbers, self.structure.positions.copy()*angstrom, rvecs=self.structure.cell*angstrom, bonds=self.bonds)
        else:
            system = System(numbers, self.structure.positions.copy()*angstrom, bonds=self.bonds)

        try:
            assert self.bonds is not None
        except AssertionError:
            system.detect_bonds(bonds_dict)
            self.bonds = system.bonds
            print('Warning: no bonds could be read and were automatically detected.')

        def ternary_xor(a,b,c):
            return not (b | c) if a else (b ^ c)

        if not ternary_xor(ffatypes is not None, ffatype_level is not None, ffatype_rules is not None):
            raise ValueError('Only one of ffatypes, ffatype_rules, and ffatype_level can be defined!')

        if ffatypes is not None:
            system.ffatypes = ffatypes
            system.ffatype_ids = None
            system._init_derived_ffatypes()
        if ffatype_rules is not None:
            system.detect_ffatypes(ffatype_rules)
        if ffatype_level is not None:
            set_ffatypes(system, ffatype_level)

        self.ffatypes = system.ffatypes.copy()
        self.ffatype_ids = system.ffatype_ids.copy()


    def set_ffpars(self,fnames=[],ffpars=None):
        '''
            Set the ffpars attribute by providing a list of file names "fnames" which are appended
            or provide the full string which can for instance be created by reading the pars files:

            with open(os.path.join(pr.path, 'pars.txt'), 'r') as f:
                ffpars = f.read()
        '''

        if not sum([fnames==[], ffpars is None])==1:
            raise IOError('Exactly one of fnames and ffpars should be defined')

        # Create the ffpars from the fnames if ffpars was not provided as argument
        if ffpars is None:
            if not isinstance(fnames,list):
                fnames = [fnames]
            # Check if all filenames exist
            assert all([os.path.isfile(fn) for fn in fnames])
            ffpars = ''
            for fn in fnames:
                with open(fn,'r') as f:
                    ffpars += f.read()

        self.input['ffpars'] = ffpars


    def write_input(self):
        # Check whether there are inconsistencies in the parameter file
        self.check_ffpars()

        input_dict = {
            'jobtype': self.input['jobtype'],
            'symbols': self.structure.get_chemical_symbols(),
            'numbers':self.structure.get_atomic_numbers(),
            'bonds': self.bonds,
            'ffatypes': self.ffatypes,
            'ffatype_ids': self.ffatype_ids,
            'ffpars': self.input['ffpars'],
            'pos': self.structure.positions,
            'masses': self.structure.get_masses(),
            'rcut': self.input['rcut'],
            'alpha_scale': self.input['alpha_scale'],
            'gcut_scale': self.input['gcut_scale'],
            'smooth_ei': self.input['smooth_ei'],
            'nsteps': self.input['nsteps'],
            'h5step': self.input['h5step'],
            'gpos_rms': self.input['gpos_rms'],
            'dpos_rms': self.input['dpos_rms'],
            'grvecs_rms': self.input['grvecs_rms'],
            'drvecs_rms': self.input['drvecs_rms'],
            'hessian_eps': self.input['hessian_eps'],
            'timestep': self.input['timestep'],
            'temp': self.input['temp'],
            'press': self.input['press'],
            'timecon_thermo': self.input['timecon_thermo'],
            'timecon_baro': self.input['timecon_baro'],
            'enhanced': self.enhanced,
            'scan': self.scan,
        }

        input_dict['cell'] = None
        if self.structure.cell is not None and self.structure.cell.volume > 0:
             input_dict['cell'] = self.structure.get_cell()
        write_chk(input_dict,working_directory=self.working_directory)
        write_pars(input_dict=input_dict,working_directory=self.working_directory)
        if self.input['jobtype'] == 'sp':
            write_ysp(input_dict=input_dict,working_directory=self.working_directory)
        elif self.input['jobtype'] == 'opt':
            write_yopt(input_dict=input_dict,working_directory=self.working_directory)
        elif self.input['jobtype'] == 'opt_cell':
            write_yopt_cell(input_dict=input_dict,working_directory=self.working_directory)
        elif self.input['jobtype'] == 'hess':
            write_yhess(input_dict=input_dict,working_directory=self.working_directory)
        elif self.input['jobtype'] == 'nve':
            write_ynve(input_dict=input_dict,working_directory=self.working_directory)
        elif self.input['jobtype'] == 'nvt':
            write_ynvt(input_dict=input_dict,working_directory=self.working_directory)
        elif self.input['jobtype'] == 'npt':
            write_ynpt(input_dict=input_dict,working_directory=self.working_directory)
        elif self.input['jobtype'] == 'scan':
            write_scan(input_dict=input_dict,working_directory=self.working_directory)
        else:
            raise IOError('Invalid job type for Yaff job, received %s' %self.input['jobtype'])
        if not self.enhanced is None:
            write_plumed_enhanced(input_dict,working_directory=self.working_directory)

    def collect_output(self):
        output_dict = collect_output(output_file=posixpath.join(self.working_directory, 'output.h5'))

        if self.enhanced is not None:
            # Check if COLVAR file exists
            if 'file_colvar' in self.enhanced and os.path.exists(posixpath.join(self.working_directory,self.enhanced['file_colvar'])):
                data = np.loadtxt(posixpath.join(self.working_directory,self.enhanced['file_colvar']))
                output_dict['enhanced/trajectory/time'] = data[:,0]
                output_dict['enhanced/trajectory/cv'] = data[:,1:-1]
                output_dict['enhanced/trajectory/bias'] = data[:,-1]

        with self.project_hdf5.open("output") as hdf5_output:
            for k, v in output_dict.items():
                hdf5_output[k] = v
            hdf5_output['generic/indices'] = np.vstack([self.structure.indices] * output_dict['generic/positions'].shape[0])

        if self.enhanced is not None:
            # Check if HILLS file exists
            if 'file' in self.enhanced and os.path.exists(posixpath.join(self.working_directory,self.enhanced['file'])):
                self.mtd_sum_hills(fn_out='fes.dat') # calculate MTD free energy profile and store it

    def to_hdf(self, hdf=None, group_name=None):
        super(Yaff, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.structure.to_hdf(hdf5_input)
            self.input.to_hdf(hdf5_input)
            hdf5_input['generic/bonds'] = self.bonds
            hdf5_input['generic/ffatypes'] = np.asarray(self.ffatypes,'S22')
            hdf5_input['generic/ffatype_ids'] = self.ffatype_ids
            if not self.enhanced is None:
                grp = hdf5_input.create_group('generic/enhanced')
                try:
                    for k,v in self.enhanced.items():
                        grp[k] = v
                except TypeError:
                    print(k,v)
                    raise TypeError('Could not save this data to h5 file.')
            if not self.scan is None:
                grp = hdf5_input.create_group('generic/scan')
                try:
                    for k,v in self.scan.items():
                        grp[k] = v
                except TypeError:
                    print(k,v)
                    raise TypeError('Could not save this data to h5 file.')

    def from_hdf(self, hdf=None, group_name=None):
        super(Yaff, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.from_hdf(hdf5_input)
            self.structure = Atoms().from_hdf(hdf5_input)
            self.bonds = hdf5_input['generic/bonds']
            self.ffatypes = np.char.decode(hdf5_input['generic/ffatypes']) # decode byte string literals
            self.ffatype_ids = hdf5_input['generic/ffatype_ids']

            if "enhanced" in hdf5_input['generic'].keys():
                self.enhanced = {}
                for key,val in hdf5_input['generic/enhanced'].items():
                    self.enhanced[key] = val
            if "scan" in hdf5_input['generic'].keys():
                self.scan = {}
                for key,val in hdf5_input['generic/scan'].items():
                    self.scan[key] = val

    def get_structure(self, iteration_step=-1, wrap_atoms=True):
        """
        Overwrite the get_structure routine from AtomisticGenericJob because we want to avoid
        defining a unit cell when one does not exist
        """
        if not (self.structure is not None):
            raise AssertionError()

        positions = self.get("output/generic/positions")
        cells = self.get("output/generic/cells")

        snapshot = self.structure.copy()
        snapshot.positions = positions[iteration_step]
        if cells is not None:
            snapshot.cell = cells[iteration_step]
        indices = self.get("output/generic/indices")
        if indices is not None:
            snapshot.indices = indices[iteration_step]
        if wrap_atoms and cells is not None:
            return snapshot.center_coordinates_in_unit_cell()
        else:
            return snapshot

    # Plot functions are deprecated while yaff is no longer in atomic units!
    def plot(self, ykey, xkey='generic/steps', xunit='au', yunit='au', ref=None, linestyle='-', rolling_average=False):
        warnings.warn('Deprecated! The output units are not necessarily atomic units, such that we can not just divide by the unit.')
        xs = self['output/%s' %xkey]/parse_unit(xunit)
        ys = self['output/%s' %ykey]/parse_unit(yunit)
        if rolling_average:
            ra = np.zeros(len(ys))
            for i, y in enumerate(ys):
                if i==0:
                    ra[i] = ys[0]
                else:
                    ra[i] = (i*ra[i-1]+ys[i])/(i+1)
            ys = ra.copy()

        self._ref(ys,ref)

        pp.clf()
        pp.plot(xs, ys, linestyle)
        pp.xlabel('%s [%s]' %(xkey, xunit))
        pp.ylabel('%s [%s]' %(ykey, yunit))
        pp.show()


    # Plot functions are deprecated while yaff is no longer in atomic units!
    def plot_multi(self, ykeys, xkey='generic/steps', xunit='au', yunit='au', ref=None, linestyle='-', rolling_average=False):
        warnings.warn('Deprecated! The output units are not necessarily atomic units, such that we can not just divide by the unit.')
        # Assume that all ykeys have the same length than the xkey
        xs  = self['output/%s' %xkey]/parse_unit(xunit)
        yss = np.array([self['output/%s' %ykey]/parse_unit(yunit) for ykey in ykeys])

        if rolling_average:
            for ys in yss:
                ra = np.zeros(len(ys))
                for i, y in enumerate(ys):
                    if i==0:
                        ra[i] = ys[0]
                    else:
                        ra[i] = (i*ra[i-1]+ys[i])/(i+1)
                ys = ra.copy()

        if not isinstance(ref,list):
            for ys in yss:
                self._ref(ys,ref)
        else:
            assert len(ref)==len(yss)
            for n in range(len(ref)):
                _ref(yss[n],ref[n])


        pp.clf()
        for n,ys in enumerate(yss):
            pp.plot(xs, ys, linestyle, label=ykeys[n])
        pp.xlabel('%s [%s]' %(xkey, xunit))
        pp.ylabel('[%s]' %(yunit))
        pp.legend()
        pp.show()

    @staticmethod
    def _ref(ys,ref):
        if isinstance(ref, int):
            ys -= ys[ref]
        elif isinstance(ref, float):
            ys -= ref
        elif isinstance(ref,str):
            if ref=='min':
                ys -= min(ys)
            elif ref=='max':
                ys -= max(ys)
            elif ref=='mean':
                ys -= np.mean(ys)

    def log(self):
        with open(posixpath.join(self.working_directory, 'yaff.log')) as f:
            print(f.read())

    def get_yaff_system(self, snapshot=0):
        numbers = self.structure.get_atomic_numbers()
        if snapshot==0:
            struct = self.structure
        else:
            struct = self.get_structure(iteration_step=snapshot, wrap_atoms=False)
        pos = struct.positions.reshape(-1,3)*angstrom
        cell = struct.cell
        if cell is not None and cell.volume>0:
            system = System(numbers, pos, rvecs=cell*angstrom, bonds=self.bonds, ffatypes=self.ffatypes, ffatype_ids=self.ffatype_ids,masses=struct.get_masses()*amu)
        else:
            system = System(numbers, pos, bonds=self.bonds, ffatypes=self.ffatypes, ffatype_ids=self.ffatype_ids,masses=struct.get_masses()*amu) # get masses contains standard masses in atomic mass units
        if system.bonds is None:
            system.detect_bonds()
        return system

    def get_yaff_ff(self, system=None):
        if system is None:
            system = self.get_yaff_system()
        fn_pars = posixpath.join(self.working_directory, 'pars.txt')
        if not os.path.isfile(fn_pars):
            raise IOError('No pars.txt file find in job working directory. Have you already run the job?')
        ff = ForceField.generate(
            system, fn_pars, rcut=self.input['rcut'], alpha_scale=self.input['alpha_scale'],
            gcut_scale=self.input['gcut_scale'], smooth_ei=self.input['smooth_ei']
        )
        return ff

    def mtd_sum_hills(self,fn_out='fes.dat',settings=None):
        '''
            Creates a fes.dat file for plotting the free energy surface after a mtd simulation.
            This is automatically performed for an mtd job.
            If you need to integrate multiple hill files instead,
            or specify certain arguments, you can use this function instead

            **Arguments**

            fn         path to the hills file or hills files (comma separated)
            settings   dictionary with settings for sum_hills command
        '''

        fn = posixpath.join(self.working_directory, self.enhanced['file'])
        fn_out = posixpath.join(self.working_directory, fn_out)

        # Get plumed path for execution
        path = self.path+'_hdf5/'+self.name+'/'
        load_module = get_plumed_path()
        plumed_script = path+'plumed_job.sh'

        if settings is not None:
            kv = " " + " ".join(["--{} {}".format(k,v) for k,v in settings.items()])
        else:
            kv = ""

        with open(plumed_script,'w') as g:
            with open(load_module,'r') as f:
                for line in f:
                    g.write(line)
                g.write("plumed sum_hills --hills {} --outfile {}".format(fn,fn_out) + kv)

        # Change permissions (equal to chmod +x)
        st = os.stat(plumed_script)
        os.chmod(plumed_script, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) # executable by everyone

        # execute wham
        command = ['exec', plumed_script]
        out = s._queue_adapter._adapter._execute_command(command, split_output=False, shell=True)


        def store_fes(data,n_cv):
            # store data
            with self.project_hdf5.open("output") as hdf5_output:
                grp = hdf5_output.create_group('enhanced/fes')
                grp['bins'] = data[...,:n_cv]
                grp['fes'] = data[...,n_cv]
                grp['dfes'] = data[...,n_cv+1:]

        # Check if stride in options, then multiple output files are generated with INDEX.dat appended
        if settings is not None and 'stride' in settings.keys():
            import glob, re
            fnames = glob.glob(os.path.join(path,fn_out)+'*')
            frames = [int(re.findall(r'\d+', fn)[-1]) for fn in fnames] # find last number in filename
            data = np.array([np.loadtxt(fnames[i]) for i in np.argsort(frames)])
        else:
            data = np.loadtxt(fn_out)
            
        n_cv = (data.shape[-1]-1)//2
        store_fes(data,n_cv)


    def check_ffpars(self):
        ffpars = self.input['ffpars'].split('\n')

        # Check DIELECTRIC
        count = ffpars.count('FIXQ:DIELECTRIC 1.0')
        if count>1:
            warnings.warn('Two instances of "FIXQ:DIELECTRIC 1.0" were found, and one will be deleted.')
            idx = ffpars.index('FIXQ:DIELECTRIC 1.0')
            del ffpars[idx]

        # Check UNITS
        units = {}
        for n,line in enumerate(ffpars):
            if 'UNIT' in line:
                spl = line.split()
                unit_tuple = (spl[0],spl[1])
                if not unit_tuple in units:
                    units[unit_tuple] = spl[2]
                else:
                    if not parse_unit(units[unit_tuple]) == parse_unit(spl[2]): # from molmod.units
                        raise ValueError('There was a conflict with your force field parameter units, namely for {} which was defined as {} both {}. Make sure that every unit unique.'.format(unit_tuple,spl[2],units[unit_tuple]))

        self.input['ffpars'] = '\n'.join(ffpars)
