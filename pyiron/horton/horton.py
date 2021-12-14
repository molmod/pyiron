from pyiron import Project
from pyiron.base.generic.parameters import GenericParameters
from pyiron.base.job.generic import GenericJob
from pyiron.base.settings.generic import Settings

import os, posixpath, h5py, stat, glob


s = Settings()

def get_horton_template():
    for resource_path in s.resource_paths:
        p = os.path.join(resource_path, "horton", "bin", "template.com")
        if os.path.exists(p):
            return p

def get_quickff_path():
    for resource_path in s.resource_paths:
        p = os.path.join(resource_path, "quickff", "bin", "run_*.sh")
        p = glob.glob(p)[0] # get quickff executable
        if os.path.exists(p):
            return p

def get_gaussian_path():
    for resource_path in s.resource_paths:
        p = os.path.join(resource_path, "gaussian", "bin", "cubegen.sh")
        if os.path.exists(p):
            return p


class Horton(GenericJob):
    def __init__(self, project, job_name):
        super(Horton, self).__init__(project, job_name)
        self.__name__ = "Horton"
        self._executable_activate(enforce=True)
        self.input = HortonInput()
        self.structure = None
        self.scheme = None
        self.fchk = None
        self.pars_file = os.path.join(self.working_directory, 'pars_ei.txt')


    def write_input(self):
        input_dict = {'bci': self.input['bci'],
                      'gaussian': self.input['gaussian'],
                      'ffatypes': self.input['ffatypes'],
                      'ei-scales': self.input['ei-scales'],
                      'bci-constraints': self.input['bci-constraints'],
                      'verbose': self.input['verbose'],
                      'scheme': self.scheme,
                      'numbers': list(set(self.structure.numbers)),
                      'lot': self.lot,
                      'basis_set': self.basis_set,
                      }
        write_input(input_dict=input_dict, working_directory=self.working_directory)
        
    def detect_ffatypes(self, ffatypes=None, ffatype_level=None):
        '''
            Define atom types by explicitely giving them through the
            ffatypes keyword, or by specifying the ffatype_level employing
            the built-in routine in QuickFF.
            Loading the atom types from an input file is not supported in pyiron.
        '''
        if ffatypes is not None ^ ffatype_level is not None:
            raise ValueError('Only one of ffatypes and ffatype_level can be defined!')
        
        if ffatype_level is not None:
            self.input['ffatypes'] = ffatype_level
        
        if ffatypes is not None:
            assert isinstance(ffatypes,list)
            self.input['ffatypes'] = ",".join(ffatypes)

    def calculate_AIM_charges(self,job,scheme='mbis'):
        try:
            self.restart_file_list.append(
                posixpath.join(job.working_directory, "input.fchk")
            )
        except IOError:
            self.logger.warning(
                msg="The fchk file is missing from: {}, therefore it can't be read.".format(
                    job.job_name
                )
            )

        try:
            assert scheme in ['b','h','hi','is','he','mbis']
        except AssertionError:
            raise ValueError('Your scheme should be one of the following: b,h,hi,is,he,mbis.')

        self.fchk = posixpath.join(job.working_directory, "input.fchk")
        self.scheme = scheme
        self.structure = job.structure
        self.lot = job.input['lot']
        self.basis_set = job.input['basis_set']

    def collect_output(self):
        output_dict = collect_output(output_file=os.path.join(self.working_directory, 'horton_out.h5'))
        with self.project_hdf5.open("output") as hdf5_output:
            for k, v in output_dict.items():
                hdf5_output[k] = v

    def to_hdf(self, hdf=None, group_name=None):
        super(Horton, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.to_hdf(hdf5_input)
            hdf5_input['generic/scheme'] = self.scheme
            hdf5_input['generic/fchk'] = self.fchk
            hdf5_input['generic/pars_file'] = self.pars_file

    def from_hdf(self, hdf=None, group_name=None):
        super(Horton, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.from_hdf(hdf5_input)
            self.scheme = hdf5_input['generic/scheme']
            self.fchk = hdf5_input['generic/fchk']
            self.pars_file = hdf5_input['generic/pars_file']


    def log(self):
        logtext = ""
        with open(os.path.join(self.working_directory, 'horton.log')) as f:
            logtext+= f.read()

        logtext+='\n\n'

        with open(os.path.join(self.working_directory, 'quickff.log')) as f:
            logtext+= f.read()

        print(logtext)


def write_input(input_dict,working_directory='.'):
    options=[]

    if input_dict['verbose'] is not None and input_dict['verbose']:
        options+= ['--verbose']

    if input_dict['ffatypes'] is not None:
        options+= ['--ffatypes {}'.format(input_dict['ffatypes'])]

    if input_dict['gaussian']:
        options+= ['--gaussian']

    if input_dict['bci']:
        options+= ['--bci']

    if input_dict['ei-scales'] is not None:
        options+= ['--ei-scales {}'.format(','.join([str(i) for i in input_dict['ei-scales']]))]

    if input_dict['bci-constraints'] is not None:
        options+= ['--bci-constraints {}'.format(input_dict['bci-constraints'])]

    import_statement = """#! /usr/bin/python \nfrom quickff.scripts import qff_input_ei\n"""

    body = import_statement + 'qff_input_ei("{} input.fchk horton_out.h5:/charges")'.format(' '.join(options))
    with open(posixpath.join(working_directory,'qff_input_ei.py'), 'w') as f:
        f.write(body)

    with open(posixpath.join(working_directory,'template.com'), 'w') as g:
        with open(get_horton_template(),'r') as f:
            for n,line in enumerate(f):
                if n==1:
                    g.write(line.format(lot=input_dict['lot'],basis=input_dict['basis_set']))
                else:
                    g.write(line)


    horton_script = posixpath.join(working_directory,'horton_job.sh')
    with open(horton_script,'w') as g:
        g.write('#!/bin/bash\n')

        if not input_dict['scheme'] in ['b','is','mbis']:
            g.write("mkdir atomdb; cp template.com atomdb/; cd atomdb/\n")
            g.write("horton-atomdb.py input g09 {} template.com\n\n".format(",".join([str(n) for n in input_dict['numbers']])))
            with open(get_gaussian_path(),'r') as f:
                for line in f:
                    if not line.startswith('#'):
                        g.write(line)
                g.write('\n')
            g.write("sed -i 's/g09/g16/g' run_g09.sh\n") # replace gaussian09 by gaussian16
            g.write("./run_g09.sh\n\n")

            with open(get_horton_path(),'r') as f:
                for n,line in enumerate(f):
                    if not line.startswith('#'):
                        g.write(line)
                g.write('\n')
            g.write("horton-atomdb.py convert\n")
            g.write("mv atoms.h5 ..; cd ..\n")
            g.write("horton-wpart.py --grid veryfine input.fchk horton_out.h5 {} atoms.h5 > horton.log\n\n".format(input_dict['scheme']))
        else:
            g.write("horton-wpart.py --grid veryfine input.fchk horton_out.h5 {} > horton.log\n\n".format(input_dict['scheme']))

        with open(get_quickff_path(),'r') as f:
            for line in f:
                if not line.startswith('#') and (line.startswith('ml') or line.startswith('module')):
                    g.write(line)
            g.write('\n')
        g.write("python qff_input_ei.py > quickff.log\n")

        # Change permissions (equal to chmod +x)
        st = os.stat(horton_script)
        os.chmod(horton_script, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) # executable by everyone

def collect_output(output_file):
    # this routine basically reads and returns the output HDF5 file produced by Yaff
    # read output
    h5 = h5py.File(output_file, mode='r')
    # translate to dict
    output_dict = {}
    output_dict['charges'] = h5['charges'][:]
    # read colvar file if it is there
    return output_dict

class HortonInput(GenericParameters):
    def __init__(self, input_file_name=None):
        super(HortonInput, self).__init__(input_file_name=input_file_name, table_name="input_inp", comment_char="#")

    def load_default(self):
        '''
        Loading the default settings for the input file.
        '''
        input_str = """\
bci True # Convert averaged atomic charges to bond charge increments, i.e. charge transfers along the chemical bonds in the system.
gaussian True # Use gaussian smeared charges
ffatypes high # {None,list_of_atypes,low,medium,high,highest}
ei-scales 1,1,1 # A comma-seperated list representing the electrostatic neighborscales
bci-constraints None # A file containing constraints for the charge to bci fit in a master: slave0,slave1,...: sign format
verbose True # set False if you don't want a log file
"""
        self.load_string(input_str)
