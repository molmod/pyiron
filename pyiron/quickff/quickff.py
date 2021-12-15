from pyiron import Project
from pyiron.base.generic.parameters import GenericParameters
from pyiron.atomistics.structure.atoms import Atoms
from pyiron.atomistics.job.atomistic import AtomisticGenericJob
from pyiron.base.settings.generic import Settings

from collections import OrderedDict

from yaff import System, log as Yafflog, ForceField
Yafflog.set_level(Yafflog.silent)
from quickff import read_abinitio
from quickff.tools import set_ffatypes
from quickff.settings import key_checks
from molmod.units import *
from molmod.constants import *
from molmod.io.chk import load_chk, dump_chk
from molmod.periodic import periodic as pt

import os, posixpath, numpy as np, h5py, matplotlib.pyplot as pp

s = Settings()

def get_mm3_path():
    for resource_path in s.resource_paths:
        p = os.path.join(resource_path, "quickff", "bin", "mm3.prm")
        if os.path.exists(p):
            return p

def get_uff_path():
    for resource_path in s.resource_paths:
        p = os.path.join(resource_path, "quickff", "bin", "uff.prm")
        if os.path.exists(p):
            return p


def write_chk(input_dict, working_directory='.'):
    # collect data and initialize Yaff system
    if 'cell' in input_dict.keys() and input_dict['cell'] is not None and input_dict['cell'].volume > 0:
        system = System(input_dict['numbers'], input_dict['pos']*angstrom, rvecs=input_dict['cell']*angstrom, ffatypes=input_dict['ffatypes_man'], ffatype_ids=input_dict['ffatype_ids_man'])
    else:
        system = System(input_dict['numbers'], input_dict['pos']*angstrom, ffatypes=input_dict['ffatypes_man'], ffatype_ids=input_dict['ffatype_ids_man'])
    # determine masses, bonds and ffaypes from ffatype_rules
    system.detect_bonds()
    system.set_standard_masses()
    # write dictionnairy to MolMod CHK file
    system.to_file(posixpath.join(working_directory,'input.chk'))
    # Reload input.chk as dictionairy and add AI input data
    d = load_chk(posixpath.join(working_directory,'input.chk'))

    assert isinstance(input_dict['aiener'], float), "AI energy not defined in input, use job.read_abintio(...)"
    assert isinstance(input_dict['aigrad'], np.ndarray), "AI gradient not defined in input, use job.read_abintio(...)"
    assert isinstance(input_dict['aihess'], np.ndarray), "AI hessian not defined in input, use job.read_abintio(...)"
    d['energy'] = input_dict['aiener']
    d['grad'] = input_dict['aigrad']
    d['hess'] = input_dict['aihess']
    dump_chk(posixpath.join(working_directory,'input.chk'), d)

def write_pars(pars,fn,working_directory='.'):
    with open(posixpath.join(working_directory,fn), 'w') as f:
        for line in pars:
            f.write(line)

def write_config(input_dict,working_directory='.'):
    with open(posixpath.join(working_directory,'config.txt'), 'w') as f:
        for key in key_checks.keys():
            if key in input_dict.keys():
                value = str(input_dict[key])
                if key=='ffatypes': assert value == 'None'
                print('%s:   %s' %(key+' '*(30-len(key)), value), file=f)

def collect_output(fn_pars, fn_sys):
    # this routine reads the output parameter file containing the covalent pars
    output_dict = {'generic/bond': [], 'generic/bend': [], 'generic/torsion': [], 'generic/oopdist': [], 'generic/cross': []}
    kinds = ['bond', 'bend', 'torsion', 'oopdist', 'cross']
    with open(fn_pars, 'r') as f:
        for line in f.readlines():
            for key in kinds:
                if key in line.lower():
                    output_dict['generic/%s' %key].append(line)
    system = System.from_file(fn_sys)
    output_dict['system/numbers'] = system.numbers
    output_dict['system/pos'] = system.pos/angstrom
    output_dict['system/charges'] = system.charges
    if system.cell is not None:
        output_dict['system/rvecs'] = system.cell.rvecs/angstrom
    output_dict['system/bonds'] = system.bonds
    output_dict['system/ffatypes'] = np.asarray(system.ffatypes,'S22')
    output_dict['system/ffatype_ids'] = system.ffatype_ids
    return output_dict


class QuickFFInput(GenericParameters):
    def __init__(self, input_file_name=None):
        super(QuickFFInput, self).__init__(input_file_name=input_file_name,table_name="input_inp",comment_char="#")

    def load_default(self):
        """
        Loading the default settings for the input file.
        """
        input_str = """\
fn_yaff pars_cov.txt
fn_charmm22_prm None
fn_charmm22_psf None
fn_sys system.chk
plot_traj None
xyz_traj False
fn_traj None
log_level high
log_file quickff.log
program_mode DeriveFF
only_traj PT_ALL
ffatypes None #Define atom types using the built-in routine in QuickFF (see documentation)
ei None
ei_rcut None #default is 20 (periodic) or 50 (non-per) A
vdw None
vdw_rcut 37.79452267842504
covres None
excl_bonds None
excl_bends None
excl_dihs None
excl_oopds None
do_hess_mass_weighting True
do_hess_negfreq_proj False
do_cross_svd True
pert_traj_tol 1e-3
pert_traj_energy_noise None
cross_svd_rcond 1e-8
do_bonds True
do_bends True
do_dihedrals True
do_oops True
do_cross_ASS True
do_cross_ASA True
do_cross_DSS False
do_cross_DSD False
do_cross_DAA False
do_cross_DAD False
consistent_cross_rvs True
remove_dysfunctional_cross True
bond_term BondHarm
bend_term BendAHarm
do_squarebend True
do_bendclin True
do_sqoopdist_to_oopdist True
"""
        self.load_string(input_str)



class QuickFF(AtomisticGenericJob):
    def __init__(self, project, job_name):
        super(QuickFF, self).__init__(project, job_name)
        self.__name__ = "QuickFF"
        self._executable_activate(enforce=True)
        self.input = QuickFFInput()
        self.ffatypes = None
        self.ffatype_ids = None
        #self.aiener = None
        #self.aigrad = None
        #self.aihess = None
        self.fn_ai = None
        self.fn_ei = None
        self.fn_vdw = None

    def read_abinitio(self, fn):
        numbers, coords, energy, grad, hess, masses, rvecs, pbc = read_abinitio(fn,do_hess=False)
        coords /= angstrom
        if rvecs is not None:
            rvecs /= angstrom
        self.structure = Atoms(numbers=numbers, positions=coords, cell=rvecs)
        self.fn_ai = fn

        # We will read these properties later, we do not want to store them in the h5 file
        #self.aiener = energy
        #self.aigrad = grad
        #self.aihess = hess

    def detect_ffatypes(self, ffatypes=None, ffatype_rules=None, ffatype_level=None):
        '''
            Define atom types by explicitely giving them through the
            ffatypes keyword, defining atype rules using the ATSELECT
            language implemented in Yaff (see the Yaff documentation at
            http://molmod.github.io/yaff/ug_atselect.html) or by specifying
            the ffatype_level employing the built-in routine in QuickFF.
        '''
        numbers = np.array([pt[symbol].number for symbol in self.structure.get_chemical_symbols()])
        if self.structure.cell is not None and self.structure.cell.volume > 0:
            system = System(numbers, self.structure.positions.copy()*angstrom, rvecs=self.structure.cell*angstrom)
        else:
            system = System(numbers, self.structure.positions.copy()*angstrom)
        system.detect_bonds()

        def ternary_xor(a,b,c):
            return a if not (b | c) else (b ^ c)

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

    def derive_vdw_mm3(self,fn_vdw='pars_vdw.txt'):
        """
            Derive the VDW parameters from the MM3 force field using the atom types of the current job
        """
        try:
            from openbabel import openbabel
            import pyiron.quickff.mm3 as mm3
        except ImportError:
            raise ImportError('Could not load the openbabel module, make sure the openbabel module is active!')

        # Create working directory to store files in
        self._create_working_directory()

        # Simple sanity checks
        if self.ffatypes is None:
            raise ValueError('You did not yet assign the atom types. Try placing detect_ffatypes() before this command.')

        if self.structure is None:
            raise ValueError('You did not yet assign the structure. Try placing read_abintio() before this command.')

        fn_sys = self.input['fn_sys']
        if not self.status.finished:
            fn_sys = 'input.chk'
            self.write_input()

        # Check if the structure is periodic
        periodic_structure = self.structure.cell is not None and self.structure.cell.volume > 0
        if periodic_structure:
            raise ValueError('The current implementation of openbabel does not parse the bonds correctly for periodic structures, such that the MM3 atomtypes can not be recognized.')

        # Create xyz file of structure
        fn_xyz = os.path.join(self.working_directory,'structure.xyz')
        fn_txyz = os.path.join(self.working_directory,'structure.txyz')
        self.structure.write(fn_xyz)

        # Convert xyz to Tinker xyz with types
        convertor = openbabel.OBConversion()
        convertor.SetInAndOutFormats('xyz','txyz')
        convertor.AddOption('3',convertor.OUTOPTIONS) # set MM3 instead of MM2, option -3 to output options

        structure = openbabel.OBMol()
        convertor.ReadFile(structure, fn_xyz)
        convertor.WriteFile(structure, fn_txyz)

        # Create MM3 pars file
        fn_sys = os.path.join(self.working_directory, fn_sys)
        fn_out = os.path.join(self.working_directory, fn_vdw)

        mm3_atomtypes = mm3.get_mm3_indices(fn_sys, fn_txyz, periodic_structure)
        mm3_ff = mm3.get_mm3_ff(get_mm3_path())
        mm3.write_mm3_pars(mm3_ff, mm3_atomtypes, fn_out)

    def derive_uff(self,ljcross=False,full_ff=False,fn_vdw='pars_vdw.txt',fn_cov='pars_cov_lj.txt'):
        """
            Derive the UFF force field parameters using the atom types of the current job

            ***Arguments***

            ljcross
                True: explicitly calculates all cross terms separately for the vdw part

            full_ff
                False: only calculates the VDW part
                True: derives both the covalent and VDW UFF parameters for the provided atom types
                      in this case the electrostatic part should still be added from Horton

        """
        # Create working directory to store files in
        self._create_working_directory()

        # Simple sanity checks
        if self.ffatypes is None:
            raise ValueError('You did not yet assign the atom types. Try placing detect_ffatypes() before this command.')

        if self.structure is None:
            raise ValueError('You did not yet assign the structure. Try placing read_abintio() before this command.')

        fn_sys = self.input['fn_sys']
        if not self.status.finished:
            fn_sys = 'input.chk'
            self.write_input()

        import pyiron.quickff.uff as uff
        system = System.from_file(os.path.join(self.working_directory,fn_sys))

        fn_out_vdw = os.path.join(self.working_directory, fn_vdw)
        fn_out_cov = os.path.join(self.working_directory, fn_cov)

        uff = uff.UFFMachine(system, get_uff_path())
        uff.build(ljcross=ljcross)
        uff.pars_lj.to_file(fn_out_vdw)
        if full_ff:
            uff.pars_cov.to_file(fn_out_cov)
            print('Dont forget to add the electrostatic part to these ff parts!')

    def set_ei(self, fn):
        self.input['ei'] = fn.split('/')[-1]
        self.fn_ei = fn

    def set_vdw(self, fn):
        self.input['vdw'] = fn.split('/')[-1]
        self.fn_vdw = fn

    def write_input(self):
        #load system related input
        input_dict = {
            'symbols': self.structure.get_chemical_symbols(),
            'numbers': np.array([pt[symbol].number for symbol in self.structure.get_chemical_symbols()]),
            'ffatypes_man': self.ffatypes,
            'ffatype_ids_man': self.ffatype_ids,
            'pos': self.structure.positions,

            #'aiener': self.aiener,
            #'aigrad': self.aigrad,
            #'aihess': self.aihess,
        }

        # Assign ai input data
        _, _, energy, grad, hess, _, _, _ = read_abinitio(self.fn_ai,do_hess=True)
        input_dict['aiener'] = energy
        input_dict['aigrad'] = grad
        input_dict['aihess'] = hess

        for key in self.input._dataset["Parameter"]:
            input_dict[key] = self.input[key]
        input_dict['cell'] = None
        if self.structure.cell is not None and self.structure.cell.volume > 0:
             input_dict['cell'] = self.structure.get_cell()
        #load all input settings from self.input
        for key, value in self.input._dataset.items():
            input_dict[key] = value
        #write input chk file
        write_chk(input_dict,working_directory=self.working_directory)
        #write nonbonded pars and config input files
        if self.fn_ei is not None:
            assert self.input['ei'] is not None
            os.system('cp %s %s/%s'  %(self.fn_ei , self.working_directory, self.input['ei']))
        if self.fn_vdw is not None:
            assert self.input['vdw'] is not None
            os.system('cp %s %s/%s' %(self.fn_vdw, self.working_directory, self.input['vdw']))
        write_config(input_dict,working_directory=self.working_directory)

    def collect_output(self):
        output_dict = collect_output(posixpath.join(self.working_directory, self.input['fn_yaff']), posixpath.join(self.working_directory, self.input['fn_sys']))
        with self.project_hdf5.open("output") as hdf5_output:
            for k, v in output_dict.items():
                hdf5_output[k] = v

    def to_hdf(self, hdf=None, group_name=None):
        super(QuickFF, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.structure.to_hdf(hdf5_input)
            self.input.to_hdf(hdf5_input)
            # Also save other attributes
            hdf5_input['generic/ffatypes'] = np.asarray(self.ffatypes,'S22')
            hdf5_input['generic/ffatype_ids'] = self.ffatype_ids
            hdf5_input['generic/fn_ai'] = self.fn_ai
            hdf5_input['generic/fn_ei'] = self.fn_ei
            hdf5_input['generic/fn_vdw'] = self.fn_vdw


    def from_hdf(self, hdf=None, group_name=None):
        super(QuickFF, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.from_hdf(hdf5_input)
            self.structure = Atoms().from_hdf(hdf5_input)
            self.ffatypes = np.char.decode(hdf5_input['generic/ffatypes']) # decode byte string literals
            self.ffatype_ids = hdf5_input['generic/ffatype_ids']
            self.fn_ai = hdf5_input['generic/fn_ai']
            self.fn_ei = hdf5_input['generic/fn_ei']
            self.fn_vdw = hdf5_input['generic/fn_vdw']


    def get_structure(self, iteration_step=-1, wrap_atoms=True):
        """
        Overwrite the get_structure routine from AtomisticGenericJob because we want to avoid
        defining a unit cell when one does not exist
        """
        raise NotImplementedError

    def log(self):
        with open(posixpath.join(self.working_directory, 'quickff.log')) as f:
            print(f.read())

    def get_yaff_system(self):
        system = System.from_file(posixpath.join(self.working_directory, self.input['fn_sys']))
        return system

    def get_yaff_ff(self, rcut=15*angstrom, alpha_scale=3.2, gcut_scale=1.5, smooth_ei=True):
        system = self.get_yaff_system()
        fn_pars = posixpath.join(self.working_directory, self.input['fn_yaff'])
        if not os.path.isfile(fn_pars):
            raise IOError('No pars.txt file find in job working directory. Have you already run the job?')
        ff = ForceField.generate(
            system, fn_pars, rcut=rcut, alpha_scale=alpha_scale,
            gcut_scale=gcut_scale, smooth_ei=smooth_ei
        )
        return ff
