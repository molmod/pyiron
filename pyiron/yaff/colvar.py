# coding: utf-8

import os, re, stat, glob, h5py, posixpath, inspect, warnings, numbers, subprocess
import numpy as np

from molmod.units import *

from yaff import System, ForceField, Iterative, log
log.set_level(log.silent)

from pyiron.base.job.generic import GenericJob
from pyiron.base.settings.generic import Settings
from pyiron.atomistics.structure.atoms import Atoms


# This module is a utility module to easily create collective variable definitions to use in PLUMED
s = Settings()

def get_yaff_path():
    for resource_path in s.resource_paths:
        paths = glob.glob(os.path.join(resource_path, "yaff", "bin", "run_yaff_*.sh"))
        if len(paths)>0:
            return paths[-1]

class CV:
    def __init__(self, name=None, plumed_identifier=None, options={}, auxiliary_cvs=[]):
        '''
            Class to create CV objects which can be translated to plumed code.
            You can also create a dummy CV object and specify the plumed lines yourself:

            cv = CV() # dummy CV object
            cv.set_plumed_lines(lines)

            **Arguments**

            name (str)
                    user specified name to refer to this collective variable

            plumed_identifier (str)
                    the identifier as found on the PLUMED site (e.g. TORSION)
                    https://www.plumed.org/doc-v2.5/user-doc/html/_colvar.html

            options (dict, str : list/str)
                    options for the collective variable as specified on the PLUMED site
                    e.g. {'ATOMS': [1,2,3,4]} or {'ATOMS': '1-100'}
                    or {'ATOMS': 'c1,c2'} with c1 and c2 names of  auxiliary_cvs

            auxiliary_cvs (list of CV objects)
                    list of CV objects which are required to specify this CV
                    the plumed lines of these auxiliary_cvs are prepended to
                    the plumed lines of this CV
        '''

        self.name = name
        self.plumed_identifier = plumed_identifier.upper() if plumed_identifier is not None else None
        self.options = options
        self.auxiliary_cvs = auxiliary_cvs # these plumed lines are prepended to the current CV line
        self._format_options()
        self.plumed_lines = self._create_plumed_lines()

    def to_hdf(self, group):
        cv_group = group.create_group(self.name)
        cv_group['name'] = self.name
        cv_group['plumed_identifier'] = self.plumed_identifier
        cv_group['options'] = self.options
        if len(self.auxiliary_cvs)>0:
            auxilary_cv_group = cv_group.create_group('auxiliary_cvs')
            for aux_cv in self.auxiliary_cvs:
                aux_cv.to_hdf(auxilary_cv_group)

    def from_hdf(self, group):
        # Assume group is the created group from the to_hdf function
        # If there is an auxillary CV group load them into a list of CV objects
        auxiliary_cvs = []
        if 'auxiliary_cvs' in group.keys():
            auxiliary_cvs = []
            for cv in group['auxiliary_cvs'].values():
                auxiliary_cvs.append(colvar.CV().from_hdf(cv))
        return CV(name=group['name'], plumed_identifier=group['plumed_identifier'], options=group['options'], auxiliary_cvs=auxiliary_cvs)

    def set_plumed_lines(self,lines):
        """
            Create your own plumed lines if you know what you are doing
        """
        warnings.warn('Your atom indices should start counting from 1 if you are setting your own plumed lines!')
        self.plumed_lines = lines

    def _format_options(self):
        # Make all options upper case
        self.options = {k.upper():v for k,v in self.options.items()}

    def _increment_numbers(self):
        # This should only be executed when creating the plumed lines, such that the values are not adapted continuously
        # Some options may need to be adapted, atom indices start from 1 in plumed
        adapted_options = {}
        for k,v in self.options.items():
            if any([k.startswith(l) for l in ['ATOMS', 'ENTITY',]]):
                # Check type of values to adapt it
                if isinstance(v,(np.ndarray, list)):
                    # if the values are provided as list, assume it has numbers and strings with auxiliary_cvs names
                    assert all([isinstance(val, numbers.Number) or (isinstance(val,str) and val in [aux_cv.name for aux_cv in self.auxiliary_cvs]) for val in v])
                    values_list = [int(val)+1 if isinstance(val,numbers.Number) else val for val in v]
                    # convert to string
                    adapted_options[k] = ",".join([str(val) for val in values_list])
                elif isinstance(v, str):
                    # e.g. ATOMS=1-100
                    values = "".join(v.split()) # remove all whitespacing

                    # remove all auxilary_cv names, as they might contain numbers

                    def increment_match(matchobj):
                        return str(int(matchobj.group(0))+1)
                    def increment(s):
                        return re.sub('\d+', replace, s)

                    adapted_options[k]=increment(values)
                elif isinstance(v, int):
                    adapted_options[k]=v+1
            else:
                adapted_options[k]=v

        return adapted_options

    def _create_plumed_lines(self):
        # Increment the atom indices if necessary
        adapted_options = self._increment_numbers()

        # Initialize the plumed line
        line=""

        # If there any auxiliary_cvs, prepend their lines
        for aux_cv in self.auxiliary_cvs:
            line+=aux_cv.plumed_lines

        # Add the name if specified
        if self.name is not None:
            line+= self.name + ": "

        # Add the plumed identifier
        if self.plumed_identifier is not None:
            line+=self.plumed_identifier + " "

        # Add the options
        line+= " ".join(["{}={}".format(str(k),str(v)) if v is not None else "{}".format(str(k)) for k,v in adapted_options.items()])

        # Add a new line character
        line+='\n'

        return line

    def get_cv_values(self,job):
        '''
        Return the values for this CV based on the provided job
        The corresponding output files will be created in that job's working directory
        '''

        # Create working directory to store files in if it does not exist yet
        job._create_working_directory()

        # Remove all previously generated PLUMED CV files (since they will make a backup if they are overwritten)
        cv_files = ['plumed_cv.dat','cv.log','COLVAR_CV']
        for cvf in cv_files:
            if os.path.exists(posixpath.join(job.working_directory,cvf)):
                os.remove(posixpath.join(job.working_directory,cvf))

        # Determine number of frames for which we need to calculate CVs
        assert isinstance(job, GenericJob)
        atomic_numbers = job.structure.get_atomic_numbers()
        masses = job.structure.get_masses()*amu

        # Check if there is a trajectory
        if job.output is not None:
            positions = job.get('output/generic/positions')
            cells = job.get('output/generic/cells')
        else:
            positions = job.structure.positions.reshape(1,-1,3)*angstrom
            cells = (job.structure.cell).reshape(1,3,3)*angstrom if job.structure.cell is not None and job.structure.cell.volume>0 else None

        # Write inputdata to h5 file
        h5_path = posixpath.join(job.working_directory,'cv_structures.h5')
        with h5py.File(h5_path,mode='w') as h5:
            h5['masses'] = masses
            h5['numbers'] = atomic_numbers
            h5['positions'] = positions
            if cells is not None:
                h5['cells'] = cells

        # Setup PLUMED input
        with open(posixpath.join(job.working_directory,'plumed_cv.dat'), 'w') as f:
            f.write('UNITS LENGTH=Bohr ENERGY=kj/mol TIME=fs \n')
            f.write(self.plumed_lines)
            f.write('PRINT ARG={} FILE={}/COLVAR_cv STRIDE=1 \n'.format(self.name,job.working_directory))

        # Setup Yaff input
        body = inspect.cleandoc("""#! /usr/bin/python

        from molmod.units import *
        from yaff import *
        import h5py, numpy as np

        class SimplePlumedIntegrator(Iterative):
            default_state = [
                # we don't need any properties
            ]

            log_name = 'CVGEN'

            def __init__(self, ff, positions, cells, state=None, hooks=None, counter0=1):
                self.positions = positions
                self.cells = cells
                self.assign_structure(ff, positions, cells, 0)
                self.timestep = 1. # arbitrary timestep, required for PLUMED coupling
                Iterative.__init__(self, ff, state, hooks, counter0)

            @staticmethod
            def assign_structure(ff,positions,cells,counter):
                ff.update_pos(positions[counter])
                if cells is not None:
                    ff.update_rvecs(cells[counter])

            def initialize(self):
                Iterative.initialize(self)

            def propagate(self):
                # counter0 should start from 1 since the counter is updated after evaluation
                self.assign_structure(self.ff, self.positions, self.cells, self.counter)
                Iterative.propagate(self)

            def finalize(self):
                pass

        # Load input data
        with h5py.File('{path}/cv_structures.h5',mode='r') as h5:
            masses = h5['masses'][:]
            numbers = h5['numbers'][:]
            positions = h5['positions'][:]
            if 'cells' in h5:
                cells = h5['cells'][:]
            else:
                cells = None

        # Setting up system and force field
        system = System(numbers,positions[0],rvecs=cells[0] if cells is not None else None,masses=masses)
        ff = ForceField(system,[])

        # Add PLUMED part
        plumed = ForcePartPlumed(ff.system, fn='{path}/plumed_cv.dat',fn_log='{path}/plumed_cv.log')
        ff.add_part(plumed)

        # Setting up simulation
        dummy = SimplePlumedIntegrator(ff,positions,cells,hooks=[plumed])
        dummy.run(positions.shape[0]-1) # we start from 0, so remove one step
        """.format(path=job.working_directory)
        )

        # Create executable script
        script_log_path = posixpath.join(job.working_directory,'cv.log')
        script_path = posixpath.join(job.working_directory,'cv_script.py')
        with open(script_path, 'w') as f:
            f.write(body)

        yaff_executable_path = get_yaff_path()

        job_path = posixpath.join(job.working_directory,'cv_script.sh')
        with open(job_path, 'w') as f:
            # Add Yaff module lines
            with open(yaff_executable_path,'r') as g:
                for line in g:
                    if not line.startswith('python'): # omit the line where the original yaff script is executed
                        f.write(line)
            # Execute script
            f.write('python {} > {}\n'.format(script_path,script_log_path))

        # Change permissions (equal to chmod +x)
        st = os.stat(job_path)
        os.chmod(job_path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) # executable by everyone

        # Execute script
        try:
            out = subprocess.check_output(
                    'exec '+job_path,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    shell=True,
                )
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

        # Read plumed output
        data = np.loadtxt(posixpath.join(job.working_directory,'COLVAR_cv'))

        return data[:,1]
