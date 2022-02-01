# coding: utf-8
import os, posixpath, h5py, stat, warnings
import numpy as np
import matplotlib.pyplot as pp

from molmod.units import *
from molmod.constants import *

from yaff import System, log, ForceField
log.set_level(log.silent)

from quickff.tools import set_ffatypes

from pyiron.atomistics.structure.atoms import Atoms
from pyiron.atomistics.job.atomistic import AtomisticGenericJob
from pyiron.base.settings.generic import Settings
from pyiron.yaff.storage import ChunkedStorageParser
from pyiron.yaff.input import YaffInput, InputWriter, LAMMPSInputWriter

import pyiron.yaff.colvar as colvar



s = Settings()

def get_plumed_path():
    for resource_path in s.resource_paths:
        p = os.path.join(resource_path, "yaff", "bin", "plumed.sh")
        if os.path.exists(p):
            return p


def collect_output(output_file):
    # this routine basically reads and returns the output HDF5 file produced by Yaff
    # read output
    h5 = h5py.File(output_file, mode='r')
    # translate to ChunkedStorageParser
    output_parser = ChunkedStorageParser(h5)
    return output_parser


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

    def set_mtd(self, cvs, height, sigma, pace, fn='HILLS', fn_colvar='COLVAR', stride=10, temp=300):
        '''
            Setup a Metadynamics run using PLUMED along the collective variables
            defined in the cvs argument.

            **Arguments**

            cvs     a list of pyiron.yaff.cv.CV objects

            height  the height of the Gaussian hills, can be a single value
                    (the gaussian hills for each CV have identical height) or
                    a list of values, one for each CV defined.

            sigmas  the sigma of the Gaussian hills, can be a single value
                    (the gaussian hills for each CV have identical height) or
                    a list of values, one for each CV defined.

            pace    the number of steps after which the gaussian hills are
                    updated.

            fn      the PLUMED output file for the gaussian hills

            fn_colvar
                    the PLUMED output file for logging of collective variables

            stride  the number of steps after which the internal coordinate
                    values and bias are printed to the COLVAR output file.

            temp    the system temperature
        '''
        for cv in cvs:
            assert isinstance(cv,colvar.CV)
        if not isinstance(height,list) and not isinstance(height,np.ndarray):
            height = np.array([height])
        if not isinstance(sigma,list) and not isinstance(sigma,np.ndarray):
            sigma = np.array([sigma])
        self.enhanced= {
            'cvs': cvs, 'height': height, 'sigma': sigma, 'pace': pace,
            'file': fn, 'file_colvar': fn_colvar, 'stride': stride, 'temp': temp
        }

    def set_us(self, cvs, kappa, loc, fn_colvar='COLVAR', stride=10, temp=300):
        '''
            Setup an Umbrella sampling run using PLUMED along the collective variables
            defined in the cvs argument.

            **Arguments**

            cvs     a list of pyiron.yaff.cv.CV objects

            kappa   the value of the force constant of the harmonic bias potential,
                    can be a single value (the harmonic bias potential for each CV has identical kappa)
                    or a list of values, one for each CV defined.

            loc     the location of the umbrella
                    (should have a length equal to the number of CVs)

            fn_colvar
                    the PLUMED output file for logging of collective variables

            stride  the number of steps after which the collective variable
                    values and bias are printed to the COLVAR output file.

            temp    the system temperature
        '''
        for cv in cvs:
            assert isinstance(cv,colvar.CV)
        if not isinstance(kappa,list) and not isinstance(kappa,np.ndarray):
            kappa = np.array([kappa])
        if not isinstance(loc,list) and not isinstance(loc,np.ndarray):
            loc = np.array([loc])
        assert len(loc)==len(cvs)
        self.enhanced= {
            'cvs': cvs, 'kappa': kappa, 'loc': loc,
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

    def enable_lammps(self,executable='2019_lammps'):
        # Sanity check
        assert self.structure.cell is not None and self.structure.cell.volume > 0
        assert self.input['jobtype'] in ['nve','nvt','npt'], 'LAMMPS coupling is only implemented for MD simulations'
        self.input['use_lammps'] = True
        self.executable.version = executable # will automatically be updated to mpi version if required

    def write_input(self):
        # Check whether there are inconsistencies in the parameter file
        self.check_ffpars()

        input_dict = {
            'jobtype': self.input['jobtype'],
            'use_lammps': self.input['use_lammps'],
            'log_lammps': self.input['log_lammps'],
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
            'tailcorrections': self.input['tailcorrections'],
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

        # Sanity checks
        input_dict['cell'] = None
        if self.structure.cell is not None and self.structure.cell.volume > 0:
             input_dict['cell'] = self.structure.get_cell()

        if not self.input['use_lammps']:
            assert self.server.cores == 1, 'Yaff only supports a single core!'
        else:
            if not 'lammps' in self.executable.version:
                self.executable.version = '2019_lammps' if self.server.cores == 1 else '2019_lammps_mpi'
                warnings.warn('You did not select a yaff/lammps executable. The {} version was automatically assigned.'.format(self.executable.version))

        # Write input files
        input_writer = LAMMPSInputWriter(input_dict,working_directory=self.working_directory) if self.input['use_lammps'] \
                        else InputWriter(input_dict,working_directory=self.working_directory)
        input_writer.write_chk()
        input_writer.write_pars()
        input_writer.write_jobscript(jobtype=self.input['jobtype'])

        if self.input['use_lammps']:
            input_writer.write_jobscript(jobtype='table')
        if not self.enhanced is None:
            input_writer.write_plumed()

    def collect_output(self):
        print('Starting data storage.')
        output_parser = collect_output(output_file=posixpath.join(self.working_directory, 'output.h5'))

        if self.enhanced is not None:
            # Check if COLVAR file exists
            if 'file_colvar' in self.enhanced and os.path.exists(posixpath.join(self.working_directory,self.enhanced['file_colvar'])):
                data = np.loadtxt(posixpath.join(self.working_directory,self.enhanced['file_colvar']))
                output_parser.set_enhanced_data(data)

        for n,k in enumerate(output_parser.function_names):
            print('Storing ', k)
            kpath = output_parser.h5_names[n]
            if kpath in output_parser.trajectory_names:
                # iterative data storing
                trajectory_key = posixpath.join(self.project_hdf5.h5_path, 'output', kpath)
                try:
                    trajectory_shape, trajectory_dtype = getattr(output_parser, k)(info=True)
                except TypeError: # in case there is no data
                    continue

                # Initialize dataset
                with h5py.File(self.project_hdf5.file_name, mode='a') as hdf5_output:
                    maxshape = (None,) + trajectory_shape[1:]
                    shape = (0,) + trajectory_shape[1:]
                    if trajectory_key in hdf5_output: del hdf5_output[trajectory_key]
                    dset = hdf5_output.create_dataset(trajectory_key, shape, maxshape=maxshape, dtype=trajectory_dtype)
                    dset.attrs['TITLE'] = 'ndarray' # we need to set this for h5io reading to work
                    print('\t Created {} dataset, with shape and maxshape '.format(k), shape, maxshape)

                    # Determine slice - assume we can parse data with a size up until 1e7 (arbitrary)
                    its_per_slice = int(np.ceil(1e7/np.prod(trajectory_shape[1:])))
                    num_its = int(np.ceil(trajectory_shape[0]/its_per_slice))
                    print('\t Trajectory length: {}, iterations per slice: {}, number of iterations: {}.'.format(trajectory_shape[0], its_per_slice, num_its, its_per_slice*num_its))
                    for n in range(num_its):
                        start = n*its_per_slice
                        stop = min((n+1)*its_per_slice, trajectory_shape[0])
                        if dset.shape[0] <= stop:
                            # do not over-allocate, hdf5 works with chunks internally.
                            dset.resize(stop, axis=0)
                        dset[start:stop] = getattr(output_parser, k)(slice=(start,stop))
                    print('\t Final {} dataset, with shape'.format(k), dset.shape)
            else:
                # just dump it
                with self.project_hdf5.open('output') as hdf5_output:
                    hdf5_output[kpath] = getattr(output_parser, k)()

        with self.project_hdf5.open("output") as hdf5_output:
            hdf5_output['generic/indices'] = np.vstack([self.structure.indices] * getattr(output_parser, 'structure_numbers')().shape[0])

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
                        # store each cv in the cvs group
                        if k == 'cvs':
                            cvs_grp = grp.create_group('cvs')
                            for cv in v:
                                cv.to_hdf(cvs_grp)
                        else:
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
                    if key == 'cvs':
                        # Load all cv objects
                        cvs = []
                        for cv in hdf5_input['generic/enhanced/cvs'].values():
                            cvs.append(colvar.CV().from_hdf(cv))
                        self.enhanced[key] = cvs
                    else:
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
        if self.input['use_lammps']:
            with open(posixpath.join(self.working_directory, 'table.log')) as f:
                print(f.read())

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
            gcut_scale=self.input['gcut_scale'], smooth_ei=self.input['smooth_ei'], tailcorrections=self.input['tailcorrections']
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


        def _store_fes(data,n_cv):
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
        _store_fes(data,n_cv)

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
