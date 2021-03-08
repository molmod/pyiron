# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os,subprocess,re,pandas,stat,warnings
import numpy as np
import matplotlib.pyplot as pt

from pyiron.base.settings.generic import Settings
from pyiron.dft.job.generic import GenericDFTJob
from pyiron.base.generic.parameters import GenericParameters
from pyiron.atomistics.structure.atoms import Atoms
from pyiron.atomistics.job.atomistic import Trajectory
from pyiron.qchem.cclib_parser import QChem as QChem_parser

try:
    from molmod.units import amu,angstrom,electronvolt,centimeter,kcalmol
    from molmod.constants import lightspeed
    from molmod.periodic import periodic
except ImportError:
    pass


s = Settings()

# Length dictionary for ICs
ic_len = {'stre':2, 'bend':3, 'outp':4, 'tors':4, 'linc':4, 'linp':4}

class QChem(GenericDFTJob):
    def __init__(self, project, job_name):
        super(QChem, self).__init__(project, job_name)
        self.__name__ = "QChem"
        self._executable_activate(enforce=True)
        self.input = QChemInput()


    def write_input(self):
        input_dict = {'mem': self.server.memory_limit, # per core memory
                      'cores': self.server.cores,
                      'jobtype' : self.input['jobtype'],
                      'lot': self.input['lot'],
                      'basis_set': self.input['basis_set'],
                      'spin_mult': self.input['spin_mult'],
                      'charge': self.input['charge'],
                      'settings': self.input['settings'],
                      'sections': self.input['sections'],
                      'bsse_idx': self.input['bsse_idx'],
                      'symbols': self.structure.get_chemical_symbols().tolist(),
                      'pos': self.structure.positions,
                      }
        write_input(input_dict=input_dict, working_directory=self.working_directory)


    def collect_output(self):
        output_dict = collect_output(output_file=os.path.join(self.working_directory, 'job.out'))
        with self.project_hdf5.open("output") as hdf5_output:
            for k, v in output_dict.items():
                hdf5_output[k] = v
            hdf5_output['generic/indices'] = np.vstack([self.structure.indices] * output_dict['generic/positions'].shape[0])


    def to_hdf(self, hdf=None, group_name=None):
        super(QChem, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.structure.to_hdf(hdf5_input)
            self.input.to_hdf(hdf5_input)


    def from_hdf(self, hdf=None, group_name=None):
        super(QChem, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.from_hdf(hdf5_input)
            self.structure = Atoms().from_hdf(hdf5_input)


    def log(self):
        with open(os.path.join(self.working_directory, 'job.out')) as f:
            print(f.read())

    def pes_scan(self, types, indices, limits, steps):
        '''
            Function to set up a potential energy scan. One and two dimensional scans are supported.

            **Arguments**
                types (list of strings): the IC types
                    stre = interatomic distance (value>0)
                    bend = angle (0<=value<=180); atom2 is in the middle
                    outp = out-of-plane-bend (-180<=value<=180); angle between atom4 and the plane of atoms1-3
                    tors = dihedral angle (-180<=value<=180); angle between plane of atoms1-3 and atoms2-4
                    linc = coplanar bend (-180<=value<=180); angle of atoms1-3 in the plane of atoms2-4
                    linp = perpendicular bend (-180<=value<=180); angle of atoms1-3 perpendicular to the plane of atoms2-4
                indices (list of lists): the atomic indices in play for the types
                limits (list of tuples): the IC limits between which a range is constructed using steps
                steps (list of ints): number of steps between limits (including the limits)
        '''
        self.input['jobtype'] = 'PES-SCAN'

        def sanity_check(arg):
            if not isinstance(arg,list):
                arg = [arg]
            else:
                assert len(arg)<=2 # max two dimensions
            return arg

        # Sanity checks for arguments
        types = sanity_check(types)
        limits = sanity_check(limits)
        steps = sanity_check(steps)

        if not all(isinstance(i, list) for i in indices):
            indices = [indices]

        # Create scan section
        scan = {}
        for n,idx in enumerate(indices):
            assert len(idx)==ic_len[types[n]]
            scan[types[n]+'\t'+'\t'.join([str(i+1) for i in idx])] = list(limits[n]) + [steps[n]] # qchem starts counting from 1, indices are part of key to make it unique

        sections = {'scan':scan}

        if not isinstance(self.input['sections'],dict):
            self.input['sections'] = sections
        else:
            if 'scan' in self.input['sections']:
                warnings.warn('There was already a scan section. This has been overwritten!')
            self.input['sections'].update(sections)


    def set_constraints(self, types, indices, values):
        '''
            Function to assign constraints for geometric optimization.
        '''
        if self.input['jobtype'] is None or not self.input['jobtype'].upper()=='OPT':
            warnings.warn('Constraints are only sensical when used in combination with an optimization job. If you did not yet set the jobtype, please do so.')

        def sanity_check(arg):
            if not isinstance(arg,list):
                arg = [arg]
            return arg

        # Sanity checks for arguments
        types = sanity_check(types)
        values = sanity_check(values)

        if not all(isinstance(i, list) for i in indices):
            indices = [indices]

        # Create constraint section in the opt section
        constraint = {}
        for n,idx in enumerate(indices):
            assert len(idx)==ic_len[types[n]]
            constraint[types[n]+'\t'+'\t'.join([str(i+1) for i in idx])] = values[n]  # qchem starts counting from 1, indices are part of key to make it unique

        # Create fixed section in the opt section
        opt = {'constraint':constraint}
        sections = {'opt':opt}

        if not isinstance(self.input['sections'],dict):
            self.input['sections'] = sections
        elif 'opt' not in self.input['sections']:
            self.input['sections'].update(sections)
        else:
            self.input['sections']['opt'].update(opt)

    def freeze_atoms(self, indices, dimensions=None):
        '''
            Freeze atoms by their indices, specifying which of the dimensions have to stay fixed.
            Fixes the x,y and z coordinate by default for every index.
        '''
        if self.input['jobtype'] is None or not self.input['jobtype'].upper()=='OPT':
            warnings.warn('Freezing atoms is only sensical when used in combination with an optimization job. If you did not yet set the jobtype, please do so.')

        if dimensions is None:
            dimensions = ['xyz']*len(indices)

        fixed = {}
        for n,idx in enumerate(indices):
            fixed[str(idx+1)] = dimensions[n] # qchem starts counting from 1

        # Create fixed section in the opt section
        opt = {'fixed':fixed}
        sections = {'opt':opt}

        if not isinstance(self.input['sections'],dict):
            self.input['sections'] = sections
        elif 'opt' not in self.input['sections']:
            self.input['sections'].update(sections)
        else:
            self.input['sections']['opt'].update(opt)

    def set_connectivity(self, index, indices):
        '''
            Function to set the atom connectivity for atom index when the delocalized internal coordinates fail,
            e.g. in systems with long, weak bonds or in certain transition states where parts of the molecule are rearranging or dissociating
        '''

        connect = {}
        connect[str(index+1)] = [len(indices)] + [i+1 for i in indices] # qchem starts counting from 1

        # Create connect section in the opt section
        opt = {'connect':connect}
        sections = {'opt':opt}

        if not isinstance(self.input['sections'],dict):
            self.input['sections'] = sections
        elif 'opt' not in self.input['sections']:
            self.input['sections'].update(sections)
        elif 'connect' not in self.input['sections']['opt']:
            self.input['sections']['opt'].update(opt)
        else:
            self.input['sections']['opt']['connect'].update(connect)


    def calc_minimize(self, electronic_steps=None, ionic_steps=None, ionic_energy_tolerance=None, ionic_force_tolerance=None, algorithm=None):
        '''
            Function to setup the hamiltonian to perform ionic relaxations using DFT. The convergence goal can be set using
            either the iconic_energy as an limit for fluctuations in energy or the iconic_forces.

            **Arguments**

                algorithm: SCF algorithm
                electronic_steps (int): maximum number of electronic steps per electronic convergence
                ionic_steps (int): maximum number of ionic steps
                ionic_energy_tolerance (int): tolerance = $n \times 10^{-8}$
                ionic_force_tolerance (int): tolerance = $n \times 10^{-6}$.
        '''
        settings = {}

        if electronic_steps is not None:
            settings['MAX_SCF_CYCLES'] = electronic_steps

        if ionic_steps is not None:
            settings['GEOM_OPT_MAX_CYCLES'] = ionic_steps

        if algorithm is not None:
            settings['SCF_ALGORITHM'] = algorithm

        if ionic_energy_tolerance is not None:
            settings['GEOM_OPT_TOL_ENERGY'] = ionic_energy_tolerance

        if ionic_force_tolerance is not None:
            settings['GEOM_OPT_TOL_GRADIENT'] = ionic_force_tolerance

        self.input['jobtype'] = 'OPT'

        if not isinstance(self.input['settings'],dict):
            self.input['settings'] = settings
        else:
            self.input['settings'].update(settings)

        super(QChem, self).calc_minimize(
            electronic_steps=electronic_steps,
            ionic_steps=ionic_steps,
            algorithm=algorithm,
            ionic_energy_tolerance=ionic_energy_tolerance,
            ionic_force_tolerance=ionic_force_tolerance,
        )


    def calc_static(self, electronic_steps=None, algorithm=None):
        '''
            Function to setup the hamiltonian to perform static SCF DFT runs

            **Arguments**

                electronic_steps (int): maximum number of electronic steps, which can be used to achieve convergence
                algorithm (string): SCF algorithm
        '''
        settings = {}
        if electronic_steps is not None:
            settings['MAX_SCF_CYCLES'] = electronic_steps

        if algorithm is not None:
            settings['SCF_ALGORITHM'] = algorithm

        self.input['jobtype'] = 'SP'
        if not isinstance(self.input['settings'],dict):
            self.input['settings'] = settings
        else:
            self.input['settings'].update(settings)

        super(QChem, self).calc_static(
            electronic_steps=electronic_steps,
            algorithm=algorithm,
        )


    def calc_md(self, temperature=None, n_ionic_steps=1000, time_step=None, n_print=100):
        raise NotImplementedError("calc_md() not implemented in QChem.")

    def read_NMA(self):
        '''
            Reads the NMA output from the QChem .log file (they are already parsed)

            Returns:
                    IR frequencies, intensities and corresponding eigenvectors (modes).
        '''
        freqs = self.get('output/generic/nma/freqs') * (lightspeed/centimeter) # put into atomic units
        ints = self.get('output/generic/nma/ir_intensities')
        modes = self.get('output/generic/nma/modes') * angstrom # put into atomic units

        #modes = np.array(modes).reshape(nrat,len(ints),3)
        #modes = np.swapaxes(modes,0,1)

        return freqs,ints,modes



class QChemInput(GenericParameters):
    def __init__(self, input_file_name=None):
        super(QChemInput, self).__init__(input_file_name=input_file_name, table_name="input_inp", comment_char="#")

    def load_default(self):
        '''
        Loading the default settings for the input file.
        '''
        input_str = """\
lot HF
basis_set 6-311G(d,p)
spin_mult 1
charge 0
"""
        self.load_string(input_str)


def write_input(input_dict, working_directory='.'):
    # Comments can be written with ! in Gaussian
    # Load dictionary
    lot          = input_dict['lot']
    basis_set    = input_dict['basis_set']
    spin_mult    = input_dict['spin_mult'] # 2S+1
    charge       = input_dict['charge']
    symbols      = input_dict['symbols']
    pos          = input_dict['pos']

    assert pos.shape[0] == len(symbols)

    # Main settings dictionary
    rem_parameters = {}

    # JOBTYPE
    if input_dict['jobtype'] is None:
        jobtype = 'SP' # default
    else:
        jobtype = input_dict['jobtype'].upper()
        #sanity check
        assert jobtype in ['SP','OPT','TS','FREQ','FORCE','RPATH','NMR','ISSC','BSSE','EDA','PES-SCAN']
        if jobtype == 'FREQ':
            if input_dict['settings'] is None:
                input_dict['settings'] = {}
            input_dict['settings'].update({'HESS_AND_GRAD':'TRUE'}) # otherwise gradient is not printed and this is required for FF derivations
    rem_parameters['JOBTYPE'] = jobtype

    # BASIS
    rem_parameters['BASIS'] = basis_set

    # METHOD
    if isinstance(lot,list):
        rem_parameters['EXCHANGE'] = lot[0]
        rem_parameters['CORRELATION'] = lot[1]
    else:
        rem_parameters['METHOD'] = lot

    # MEMORY
    mem = input_dict['mem'] + 'B' * (input_dict['mem'][-1]!='B') # check if string ends in bytes
    cores = input_dict['cores']
    assert mem[-2:]=='MB' # later this should be converted to a function that takes care of the conversion to MB
    nmem = str(int(int(re.findall("\d+", mem)[0]) * cores))
    rem_parameters['MEM_TOTAL'] = nmem # memory in MBs

    # Add all other options from the settings dict
    if input_dict['settings'] is not None:
        for k,v in input_dict['settings'].items():
            rem_parameters[k.upper()] = v

    def write_section(f, name, parameter_dict, start=None, end=None):
        if start is None:
            start = '${}'.format(name.lower())
        if end is None:
            end='$end'

        f.write(start+'\n')
        if not name.lower()=='opt':
            for k,v in parameter_dict.items():
                if isinstance(v,list):
                    v = [str(vi) for vi in v] # make sure everything is a string
                    f.write('{}\t{}\n'.format(k,"\t".join(v)))
                else:
                    f.write('{}\t{}\n'.format(k,v))
        else:
            for k,v in parameter_dict.items():
                write_section(f,k,v,start=k.upper(),end='END'+k.upper())
        f.write(end+'\n\n')

    if not input_dict['bsse_idx'] is None:
        # sanity check
        assert input_dict['jobtype'].lower()=='bsse'
        bsse_idx = np.asarray(input_dict['bsse_idx'])
        # Check if subsequent elements maximally differ 1 and never decrease
        assert np.all(np.abs(bsse_idx[1:]-bsse_idx[:-1])<=1) and np.all(bsse_idx[1:]-bsse_idx[:-1]>=0)
        bsse_idx_set = set(input_dict['bsse_idx'])

        # Find first occurence of every index
        bsse_idx_loc = {list(input_dict['bsse_idx']).index(n):n for n in bsse_idx}

        charges = input_dict['charge'] if isinstance(input_dict['charge'],list) else [0]*(len(bsse_idx_set)+1) # total + components
        mults = input_dict['spin_mult'] if isinstance(input_dict['spin_mult'],list) else [1]*(len(bsse_idx_set)+1) # total + components
        charge = charges[0]
        spin_mult = mults[0]


    # Write to file
    with open(os.path.join(working_directory, 'job.in'), 'w') as f:
        # Write general job parameters first in correspondence with Gaussian
        write_section(f,'rem',rem_parameters)

        # Write molecule
        f.write('$molecule\n')
        f.write("{} {}\n".format(charge,spin_mult))
        if input_dict['bsse_idx'] is None:
            for n,p in enumerate(pos):
                    f.write(" {}\t{: 1.6f}\t{: 1.6f}\t{: 1.6f}\n".format(symbols[n],p[0],p[1],p[2]))
        else:
            for n,p in enumerate(pos):
                if n in bsse_idx_loc:
                    f.write('--\n')
                    f.write('{} {}\n'.format(charges[bsse_idx_loc[n]+1],mults[bsse_idx_loc[n]+1]))
                f.write(" {}\t{: 1.6f}\t{: 1.6f}\t{: 1.6f}\n".format(symbols[n],p[0],p[1],p[2]))
        f.write('$end\n\n')

        if input_dict['sections'] is not None:
            for sec_name, sec_dict in input_dict['sections'].items():
                write_section(f,sec_name,sec_dict)

        f.write('\n\n')

def qcoutput2dict(qcoutput):
    output_dict = {}

    # Basic information
    output_dict['jobtype']     = qcoutput.user_input['rem']['jobtype']

    # Structure information
    output_dict['structure/numbers']     = qcoutput.atomnos
    output_dict['structure/masses']      = np.array([periodic[n].mass for n in qcoutput.atomnos])
    output_dict['structure/charges']     = qcoutput.atomcharges['mulliken']
    output_dict['structure/dipole']      = qcoutput.moments[1] # [0] is reference 0,0,0

    output_dict['structure/dft/n_electrons']         = qcoutput.nelectrons
    output_dict['structure/dft/n_alpha_electrons']   = qcoutput.nalpha
    output_dict['structure/dft/n_beta_electrons']    = qcoutput.nbeta
    output_dict['structure/dft/n_basis_functions']   = qcoutput.nbasis

    # Orbital information
    output_dict['structure/dft/alpha_orbital_e']     = qcoutput.moenergies[0]
    output_dict['structure/dft/beta_orbital_e']      = qcoutput.moenergies[1] if len(qcoutput.moenergies)==2 else qcoutput.moenergies[0]

    # Specific job information

    # maybe we have to take into account possible energy deviations from scfenergies for the total energy?

    if output_dict['jobtype'] in ['opt', 'ts']:
        output_dict['structure/positions']   = qcoutput.converged_geometries[-1]
        output_dict['generic/positions']     = qcoutput.converged_geometries
        output_dict['generic/forces']        = qcoutput.grads/(electronvolt/angstrom)
        output_dict['generic/energy_tot']    = qcoutput.scfenergies/electronvolt

    if output_dict['jobtype']=='freq':
        output_dict['structure/positions']   = qcoutput.converged_geometries
        output_dict['generic/positions']     = qcoutput.converged_geometries
        output_dict['generic/forces']        = -qcoutput.grads[-1]/(electronvolt/angstrom)
        output_dict['generic/hessian']       = qcoutput.hessian/(electronvolt/angstrom**2)
        output_dict['generic/energy_tot']    = qcoutput.scfenergies[-1]/electronvolt
        output_dict['generic/nma/freqs']             = qcoutput.vibfreqs # in cm**-1
        output_dict['generic/nma/force_constants']   = qcoutput.vibfconsts
        output_dict['generic/nma/ir_intensities']    = qcoutput.vibirs
        if hasattr(qcoutput,'vibramans'):
            output_dict['generic/nma/raman_intensities'] = qcoutput.vibramans
        output_dict['generic/nma/modes']             = qcoutput.vibdisps

        # Thermochemistry
        output_dict['structure/thermochemistry/enthalpy'] = qcoutput.enthalpy/electronvolt
        output_dict['structure/thermochemistry/entropy_298K']  = qcoutput.entropy/electronvolt # entropy x temperature (298.15K)
        output_dict['structure/thermochemistry/free_energy'] = qcoutput.freeenergy/electronvolt

    if output_dict['jobtype']=='sp':
        output_dict['structure/positions']   = qcoutput.converged_geometries
        output_dict['generic/positions']     = qcoutput.converged_geometries
        output_dict['generic/energy_tot']    = qcoutput.scfenergies[-1]/electronvolt

    return output_dict


def collect_output(output_file):
    # Read output
    parser = QChem_parser(output_file)

    # Translate to pyiron dict
    output_dict = qcoutput2dict(parser.parse())

    return output_dict
