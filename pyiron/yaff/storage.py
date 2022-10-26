# coding: utf-8

# This module is a utility module to efficiently store the data from Yaff jobs to the H5 data container

import warnings
import numpy as np
from molmod.units import *

class ChunkedStorageParser:
    structure_names = [
        'structure/numbers',
        'structure/masses',
        'structure/ffatypes',
        'structure/ffatype_ids',
        'structure/charges',
        'structure/positions',
    ]

    trajectory_names = [
        'generic/positions',
        'generic/cells',
        'generic/steps',
        'generic/time',
        'generic/volume',
        'generic/temperature',
        'generic/pressure',
        'generic/forces',
        'generic/hessian',
        'generic/velocities',
        'generic/dipole',
        'generic/dipole_velocities',
        'generic/energy_pot',
        'generic/energy_kin',
        'generic/energy_tot',
        'generic/energy_cons',
        'generic/epot_contribs',
    ]

    trajectory_attrs_names = [
        'generic/epot_contrib_names',
    ]

    enhanced_names = [
        'enhanced/trajectory/time',
        'enhanced/trajectory/cv',
        'enhanced/trajectory/bias',
    ]

    h5_names = structure_names+trajectory_names+trajectory_attrs_names+enhanced_names
    function_names = [hn.replace('/','_') for hn in h5_names]

    def __init__(self,h5):
        self.h5 = h5
        self.enhanced_data = None # will be set

    # Define structure functions
    def structure_numbers(self):
        return self.h5['system/numbers'][:]
    def structure_masses(self):
        return self.h5['system/masses'][:]
    def structure_ffatypes(self):
        return self.h5['system/ffatypes'][:]
    def structure_ffatype_ids(self):
        return self.h5['system/ffatype_ids'][:]
    def structure_charges(self):
        if 'charges' in self.h5['system'].keys():
            return self.h5['system/charges'][:]
        else:
            warnings.warn("I could not read any charges from the system file. This could break some functionalities!")
    def structure_positions(self):
        if 'trajectory' in self.h5.keys() and 'pos' in self.h5['trajectory'].keys():
            return self.h5['trajectory/pos'][-1]/angstrom
        else:
            return np.array([self.h5['system/pos'][:]/angstrom])

    # Define generic functions
    def generic_positions(self,slice=None,info=False):
        if 'trajectory' in self.h5.keys() and 'pos' in self.h5['trajectory'].keys():
            data = self.h5['trajectory/pos']
        else:
            data = self.h5['system/pos']
            data = np.array(data).reshape((1,*data.shape))

        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]/angstrom
        return data[:]/angstrom


    def generic_cells(self,slice=None,info=False):
        if 'trajectory' in self.h5.keys() and 'cell' in self.h5['trajectory']:
            data = self.h5['trajectory/cell']
        elif 'rvecs' in self.h5['system'].keys():
            data = self.h5['system/rvecs']
            data = np.array(data).reshape((1,*data.shape))
        else:
            return None
        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]/angstrom
        return data[:]/angstrom

    def generic_steps(self,slice=None,info=False):
        if 'trajectory' in self.h5.keys() and 'counter' in self.h5['trajectory'].keys():
            data = self.h5['trajectory/counter']
        else:
            return None
        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]
        return data[:]

    def generic_time(self,slice=None,info=False):
        if 'trajectory' in self.h5.keys() and 'time' in self.h5['trajectory'].keys():
            data = self.h5['trajectory/time']
        else:
            return None
        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]/femtosecond
        return data[:]/femtosecond

    def generic_volume(self,slice=None,info=False):
        if 'trajectory' in self.h5.keys() and 'volume' in self.h5['trajectory'].keys():
            data = self.h5['trajectory/volume']
        else:
            return None
        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]/angstrom**3
        return data[:]/angstrom**3

    def generic_temperature(self,slice=None,info=False):
        if 'trajectory' in self.h5.keys() and 'temp' in self.h5['trajectory'].keys():
            data = self.h5['trajectory/temp']
        else:
            return None
        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]
        return data[:]

    def generic_pressure(self,slice=None,info=False):
        if 'trajectory' in self.h5.keys() and 'press' in self.h5['trajectory'].keys():
            data = self.h5['trajectory/press']
        else:
            return None
        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]/(1e9*pascal)
        return data[:]/(1e9*pascal)

    def generic_forces(self,slice=None,info=False):
        if 'trajectory' in self.h5.keys() and 'gradient' in self.h5['trajectory'].keys():
            data = -self.h5['trajectory/gradient']
        elif 'hessian' in self.h5['system'].keys():
            data = -self.h5['system/gpos'][:]
            data = np.array(data).reshape((1,*data.shape))
        else:
            return None
        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]/(electronvolt/angstrom)
        return data[:]/(electronvolt/angstrom)

    def generic_hessian(self,slice=None,info=False):
        if 'hessian' in self.h5['system'].keys():
            data = self.h5['system/hessian'][:]
        else:
            return None
        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]/(electronvolt/angstrom**2)
        return data[:]/(electronvolt/angstrom**2)


    def generic_velocities(self,slice=None,info=False):
        if 'trajectory' in self.h5.keys() and 'vel' in self.h5['trajectory'].keys():
            data = self.h5['trajectory/vel']
        else:
            return None
        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]/(angstrom/femtosecond)
        return data[:]/(angstrom/femtosecond)

    def generic_dipole(self,slice=None,info=False):
        if 'trajectory' in self.h5.keys() and 'dipole' in self.h5['trajectory'].keys():
            data = self.h5['trajectory/dipole'] # unit is e*A
        else:
            return None
        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]/(angstrom)
        return data[:]/(angstrom)

    def generic_dipole_velocities(self,slice=None,info=False):
        if 'trajectory' in self.h5.keys() and 'dipole_vel' in self.h5['trajectory'].keys():
            data = self.h5['trajectory/dipole_vel'] # unit is (e*A)*(A/fs)
        else:
            return None
        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]/angstrom/(angstrom/femtosecond)
        return data[:]/angstrom/(angstrom/femtosecond)

    def generic_energy_pot(self,slice=None,info=False):
        if 'trajectory' in self.h5.keys() and 'epot' in self.h5['trajectory'].keys():
            data = self.h5['trajectory/epot']
        elif 'energy' in self.h5['system'].keys():
            data = self.h5['system/energy']
            data = np.array(data).reshape((1,*data.shape))
        else:
            return None
        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]/electronvolt

        return data[...]/electronvolt # using ... solves issue with possible empty shape

    def generic_energy_kin(self,slice=None,info=False):
        if 'trajectory' in self.h5.keys() and 'ekin' in self.h5['trajectory'].keys():
            data = self.h5['trajectory/ekin'][:]
        else:
            return None
        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]/electronvolt
        return data[:]/electronvolt

    def generic_energy_tot(self,slice=None,info=False):
        if 'trajectory' in self.h5.keys() and 'etot' in self.h5['trajectory'].keys():
            data = self.h5['trajectory/etot']
        else:
            return None
        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]/electronvolt
        return data[:]/electronvolt

    def generic_energy_cons(self,slice=None,info=False):
        if 'trajectory' in self.h5.keys() and 'econs' in self.h5['trajectory'].keys():
            data = self.h5['trajectory/econs']
        else:
            return None
        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]/electronvolt
        return data[:]/electronvolt

    def generic_epot_contribs(self,slice=None,info=False):
        if 'trajectory' in self.h5.keys() and 'epot_contribs' in self.h5['trajectory'].keys():
            data = self.h5['trajectory/epot_contribs']
        else:
            return None
        if info:
            return data.shape, data.dtype
        if slice is not None:
            return data[slice[0]:slice[1],...]/electronvolt
        return data[:]/electronvolt

    def generic_epot_contrib_names(self):
        if 'trajectory' in self.h5.keys() and 'epot_contribs' in self.h5['trajectory'].keys():
            return self.h5['trajectory/'].attrs.get('epot_contrib_names')




    #Define enhanced functions
    def set_enhanced_data(self,data):
        self.enhanced_data = data

    def enhanced_trajectory_time(self):
        if self.enhanced_data is not None:
            return self.enhanced_data[:,0]

    def enhanced_trajectory_cv(self):
        if self.enhanced_data is not None:
            return self.enhanced_data[:,1:-1]

    def enhanced_trajectory_bias(self):
        if self.enhanced_data is not None:
            return self.enhanced_data[:,-1]
