from pyiron import Project
from pyiron.base.generic.parameters import GenericParameters
from pyiron.base.job.generic import GenericJob
from pyiron.base.settings.generic import Settings

import os, posixpath, h5py, stat


s = Settings()

def get_horton_path():
    for resource_path in s.resource_paths:
        p = os.path.join(resource_path, "horton", "bin", "horton.sh")
        if os.path.exists(p):
            return p

def get_quickff_path():
    for resource_path in s.resource_paths:
        p = os.path.join(resource_path, "quickff", "bin", "quickff.sh")
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
        print('Warning: Horton jobs can only be performed on the golett, swalot and phanpy clusters!')


    def write_input(self):
        input_dict = {'bci': self.input['bci'],
                      'gaussian': self.input['gaussian'],
                      'ffatypes': self.input['ffatypes'],
                      'ei-scales': self.input['ei-scales'],
                      'bci-constraints': self.input['bci-constraints'],
                      'scheme': self.scheme,
                      }
        write_input(input_dict=input_dict, working_directory=self.working_directory)

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

    def collect_output(self):
        output_dict = collect_output(output_file=os.path.join(self.working_directory, 'horton_out.h5'))
        with self.project_hdf5.open("output") as hdf5_output:
            for k, v in output_dict.items():
                hdf5_output[k] = v

    def to_hdf(self, hdf=None, group_name=None):
        super(Horton, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.to_hdf(hdf5_input)

    def from_hdf(self, hdf=None, group_name=None):
        super(Horton, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.from_hdf(hdf5_input)

    def log(self):
        with open(os.path.join(self.working_directory, 'horton.log')) as f:
            print(f.read())


def write_input(input_dict,working_directory='.'):
    options=[]
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


    horton_script = posixpath.join(working_directory,'horton_job.sh')
    with open(horton_script,'w') as g:
        with open(get_horton_path(),'r') as f:
            for line in f:
                g.write(line)
        if input_dict['scheme'] in ['b','is','mbis']:
            g.write("horton-wpart.py --grid veryfine input.fchk horton_out.h5 {} > horton.log".format(input_dict['scheme']))
        else:
            raise NotImplementedError('These schemes have not been implemented. Try again later.')
            g.write("horton-wpart.py --grid veryfine input.fchk horton_out.h5 {} atoms.h5 > horton.log".format(input_dict['scheme']))
        g.write('\n\n')
        with open(get_quickff_path(),'r') as f:
            for line in f:
                g.write(line)

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
"""
        self.load_string(input_str)
