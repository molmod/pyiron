# coding: utf-8
import os, posixpath, h5py, stat, warnings, inspect, re, stat
import numpy as np
import matplotlib.pyplot as pp

from molmod.units import *
from molmod.constants import *

from pyiron.base.generic.parameters import GenericParameters

class GPXRDInput(GenericParameters):
    def __init__(self, input_file_name=None):
        super(GPXRDInput, self).__init__(input_file_name=input_file_name, table_name="input_inp", comment_char="#")

    def load_default(self):
        '''
        Loading the default settings for the input file.
        '''
        input_str = """\
wavelength 1.54056
peakwidth 0.14
numpoints 1001
max2theta 50
detail_fhkl False
save_fhkl True
rad_type xray
"""
        self.load_string(input_str)

class InputWriter():
    """
        The GPXRD input writer creates a run.py script with the required code for the diffraction calculation
    """

    common_import = inspect.cleandoc("""#! /usr/bin/python

    import sys, os, posixpath
    import numpy as np

    import pyobjcryst
    from pyobjcryst.powderpattern import *
    from pyobjcryst.crystal import *
    from pyobjcryst.radiation import RadiationType

    deg = np.pi/180.
    """) +'\n\n'

    common_utils = inspect.cleandoc("""
    # Code to redirect stdout
    from contextlib import contextmanager

    def fileno(file_or_fd):
        fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
        if not isinstance(fd, int):
            raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
        return fd

    @contextmanager
    def stdout_redirected(to=os.devnull, stdout=None):
        if stdout is None:
           stdout = sys.stdout

        stdout_fd = fileno(stdout)
        # copy stdout_fd before it is overwritten
        #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
        with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
            stdout.flush()  # flush library buffers that dup2 knows nothing about
            try:
                os.dup2(fileno(to), stdout_fd)  # $ exec >&to
            except ValueError:  # filename
                with open(to, 'wb') as to_file:
                    os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
            try:
                yield stdout # allow code to be run with the redirected stdout
            finally:
                # restore stdout to its previous value
                #NOTE: dup2 makes stdout_fd inheritable unconditionally
                stdout.flush()
                os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied
    """) +'\n\n'


    common_frame_calc = inspect.cleandoc("""
    def calculate_frame(input_name,output_name,refpattern,skiprows,full_pattern,wavelength,rad_type,max2theta,numpoints,peakwidth,save_fhkl,detail_fhkl):
        c = CreateCrystalFromCIF(input_name)

        px = PowderPattern()
        px.SetWavelength(wavelength)
        px.SetRadiationType(rad_type)

        if refpattern is None:
            px.SetPowderPatternX(np.linspace(0, max2theta*deg, numpoints))
        else:
            px.ImportPowderPattern2ThetaObs(refpattern,skiprows) # skip no lines
            if full_pattern:
                # use an identical range as the reference pattern, but with minimum of 0
                ttheta = px.GetPowderPatternX()
                step = np.round((ttheta[1]-ttheta[0])/deg,5)
                full_ttheta = np.arange(0, max(ttheta/deg)+step, step)*deg
                px.SetPowderPatternX(full_ttheta)

        diffData = px.AddPowderPatternDiffraction(c)
        diffData.SetReflectionProfilePar(ReflectionProfileType.PROFILE_PSEUDO_VOIGT,(peakwidth*deg)**2)

        #Export data - calculated reflections
        calc = px.GetPowderPatternComponent(0)

        stdout_fd = sys.stdout.fileno()
        if save_fhkl:
         with open(output_name + '_fhkl.dat', 'w') as f, stdout_redirected(f):
             calc.PrintFhklCalc()

        if save_fhkl and detail_fhkl:
         with open(output_name + '_fhkl_detail.dat', 'w') as f, stdout_redirected(f):
             calc.PrintFhklCalcDetail()

        # Export data - 2theta space
        ttheta = px.GetPowderPatternX()
        icalc = px.GetPowderPatternCalc()

        with open(output_name + '.dat','w') as f:
            f.write('# 2theta \\t ICalc \\n')
            for i in range(len(ttheta)):
                f.write("{:7.5f}\\t{:10.8f}\\n".format(ttheta[i]/deg,icalc[i]))

        # Calculate - q space
        wavelength = calc.GetWavelength()
        qspace = 4.*np.pi/wavelength*np.sin(ttheta/2.)
        with open(output_name + '_q.dat','w') as f:
            f.write('# Q \\t ICalc \\n')
            for i in range(len(ttheta)):
                f.write("{:7.5f}\\t{:10.8f}\\n".format(qspace[i],icalc[i]))

    """)+'\n\n'

    input_static = inspect.cleandoc("""
    if __name__ == '__main__':
        # Calculate the frame
        fname_in = sys.argv[1]
        fname_out = 'output'
        calculate_frame(input_name=fname_in, output_name=fname_out, refpattern={refpattern},skiprows={skiprows},
                        full_pattern={full_pattern},wavelength={wavelength},rad_type=RadiationType.RAD_{rad_type},
                        max2theta={max2theta},numpoints={numpoints},peakwidth={peakwidth},save_fhkl={save_fhkl},detail_fhkl={detail_fhkl}
                    )
    """) +'\n\n'

    input_dynamic = inspect.cleandoc("""
    if __name__ == '__main__':
        # Calculate the frame
        frame_nr = int(sys.argv[1])
        fname_in = posixpath.join('cifs', '{{}}.cif'.format(frame_nr))
        fname_out = posixpath.join('frames', '{{}}'.format(frame_nr))
        calculate_frame(input_name=fname_in,output_name=fname_out,refpattern={refpattern},skiprows={skiprows},
                        full_pattern={full_pattern},wavelength={wavelength},rad_type=RadiationType.RAD_{rad_type},
                        max2theta={max2theta},numpoints={numpoints},peakwidth={peakwidth},save_fhkl={save_fhkl},detail_fhkl={detail_fhkl}
                    )
    """) +'\n\n'


    def __init__(self,input_dict,working_directory='.'):
        self.input_dict = input_dict
        self.working_directory = working_directory

    def write_calcscript(self):
        if self.input_dict['jobtype'] == 'static':
            self.write_static_calcscript()
        elif self.input_dict['jobtype'] == 'dynamic':
            self.write_dynamic_calcscript()
        else:
            raise ValueError

    def write_static_calcscript(self):
        # This assume only a single cif file was provided
        body  = InputWriter.common_import
        body += InputWriter.common_utils
        body += InputWriter.common_frame_calc
        body += InputWriter.input_static.format(
                            refpattern="'{}'".format("ref.tsv") if self.input_dict['refpattern'] is not None else None,
                            skiprows=self.input_dict['skiprows'] if self.input_dict['skiprows'] is not None else 0,
                            full_pattern=self.input_dict['full_pattern'],wavelength=self.input_dict['wavelength'],
                            rad_type=self.input_dict['rad_type'].upper(),max2theta=self.input_dict['max2theta'],
                            numpoints=self.input_dict['numpoints'],peakwidth=self.input_dict['peakwidth'],
                            save_fhkl=self.input_dict['save_fhkl'],
                            detail_fhkl=self.input_dict['detail_fhkl'],
                            )

        with open(posixpath.join(self.working_directory,'run.py'), 'w') as f:
            f.write(body)

    def write_dynamic_calcscript(self):
        body  = InputWriter.common_import
        body += InputWriter.common_utils
        body += InputWriter.common_frame_calc
        body += InputWriter.input_dynamic.format(
                            refpattern="'{}'".format("ref.tsv") if self.input_dict['refpattern'] is not None else None,
                            skiprows=self.input_dict['skiprows'] if self.input_dict['skiprows'] is not None else 0,
                            full_pattern=self.input_dict['full_pattern'],wavelength=self.input_dict['wavelength'],
                            rad_type=self.input_dict['rad_type'].upper(),max2theta=self.input_dict['max2theta'],
                            numpoints=self.input_dict['numpoints'],peakwidth=self.input_dict['peakwidth'],
                            save_fhkl=self.input_dict['save_fhkl'],
                            detail_fhkl=self.input_dict['detail_fhkl'],
                            )

        with open(posixpath.join(self.working_directory,'run.py'), 'w') as f:
            f.write(body)

    def write_jobscript(self):
        if self.input_dict['jobtype'] == 'static':
            self.write_static_jobscript()
        elif self.input_dict['jobtype'] == 'dynamic':
            self.write_dynamic_jobscript()
        else:
            raise ValueError

        # make the runscript executable, change permissions (equal to chmod +x)
        run_script = posixpath.join(self.working_directory,'run.sh')
        st = os.stat(run_script)
        os.chmod(run_script, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) # executable by everyone

    def write_static_jobscript(self):
        body = inspect.cleandoc("""
        #!/bin/bash

        python run.py input.cif > gpxrd.log
        """)

        with open(posixpath.join(self.working_directory,'run.sh'), 'w') as f:
            f.write(body)

    def write_dynamic_jobscript(self):
        # Literal curly brackets need to be doubled
        body = inspect.cleandoc("""
        #!/bin/bash

        # Make directory
        mkdir -p frames

        for n in {{0..{num_frames}}}; do
            python run.py ${{n}} > gpxrd.log
        done

        """).format(
            num_frames=self.input_dict['num_frames']-1,
        )

        with open(posixpath.join(self.working_directory,'run.sh'), 'w') as f:
            f.write(body)
