# coding: utf-8
import numpy as np
import matplotlib.pyplot as pt
from molmod.units import *
import subprocess, os, stat, warnings

from pyiron.base.settings.generic import Settings
from pyiron.base.master.parallel import JobGenerator
from pyiron.atomistics.structure.atoms import Atoms
from pyiron.atomistics.master.parallel import AtomisticParallelMaster

import pyiron.yaff.colvar as colvar


s = Settings()

def get_wham_path():
    for resource_path in s.resource_paths:
        p = os.path.join(resource_path, "yaff", "bin", "wham.sh")
        if os.path.exists(p):
            return p

def convert_val(val,unit=None):
    scale = 1 if unit is None else unit
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return [str(l/scale) for l in val]
    else:
        return str(val/scale)

class InputError(Exception):
    """Simple error class with clear meaning."""
    pass


class USJobGenerator(JobGenerator):
    @property
    def parameter_list(self):
        '''

        Returns:
            (list)
        '''
        # Check if all post processing parameters are correctly defined
        assert self._job.input['h_min'] is not None
        assert self._job.input['h_max'] is not None
        assert self._job.input['h_bins'] is not None

        # Create parameter list
        parameter_lst = []
        for n,_ in enumerate(self._job.input['cv_grid']):
            parameter_lst.append([n])
        return parameter_lst

    @staticmethod
    def job_name(parameter):
        return 'us_{}'.format(parameter[0]) # avoid minus signs by just using index

    def modify_job(self, job, parameter):
        if self._job.traj_job is not None:
            job.structure = self._job.traj_job.get_structure(self._job.traj_job_idx[parameter[0]])
        elif self._job.ref_f is not None:
            job.structure = self._job.ref_f(self._job.structure,  self._job.input['cv_grid'][parameter[0]])
        else:
            warnings.warn("You did not provide a trajectory or cv function, so each US simulation will start from the initial structure!")
            assert self._job.structure is not None
            job.structure = self._job.structure

        job.set_us(self._job.cvs, self._job.input['kappas'][parameter[0]], self._job.input['cv_grid'][parameter[0]],
                                    fn_colvar='COLVAR', stride=self._job.input['stride'], temp=self._job.input['temp'])
        return job


class US(AtomisticParallelMaster):
    def __init__(self, project, job_name='us'):
        '''

        Args:
            project:
            job_name:
        '''
        super(US, self).__init__(project, job_name)
        self.__name__ = 'us'
        self.__version__ = '0.1.0'

        # define default input
        self.input['kappa']       = (1.*kjmol, 'force constant of the harmonic bias potential')
        self.input['stride']      = (10, 'step for output printed to COLVAR output file.')
        self.input['temp']        = (300*kelvin, 'the system temperature')
        self.input['cv_grid']     = ([0], 'cv grid')

        self.input['h_min']       = (None , 'lowest value(s) of the cv(s) for WHAM')
        self.input['h_max']       = (None , 'highest value(s) of the cv(s) for WHAM')
        self.input['h_bins']      = (None , 'bins between h_min and h_max for WHAM')

        self.input['periodicity'] = (None , 'periodicity of cv(s)')
        self.input['tol']         = (0.00001 , 'WHAM converges if free energy changes < tol')

        self.cvs = None            # 'list of pyiron.yaff.colvar.CV objects'
        self.traj_job = None       # reference job with trajectory, does need to be initialized, it is automatically an attribute
        self.traj_job_idx = None   # idx which are selected by generate_structures_traj
        self.ref_f = None          # function to transform template structure to structure with correct cv

        self._job_generator = USJobGenerator(self)

    def set_us(self,cvs,kappas,cv_grid,stride=10,temp=300*kelvin,h_min=None,h_max=None,h_bins=None,periodicity=None,tol=0.00001):
        '''
            Setup an Umbrella sampling run using PLUMED along the collective variables
            defined in the CVs argument.

            **Arguments**

            cvs     a list of pyiron.yaff.cv.CV objects

            kappas  the value(s) of the force constant of the harmonic bias potential,
                    requires a dimension of (-1,len(cvs))
                    if the dimension is (len(cvs),) the kappa value will be identical for all locs

            cv_grid
                    the locations of the umbrellas

            stride  the number of steps after which the colelctive variable
                    values and bias are printed to the COLVAR output file.

            temp    the system temperature

            h_min   lowest value(s) of the cv(s) for WHAM, defaults to lowest cv value in cv_grid

            h_max   highest value(s) of the cv(s) for WHAM, defaults to highest cv value in cv_grid

            h_bins  number of bins between h_min and h_max for WHAM, defaults to length of cv_grid

            periodicity
                    periodicity of cv(s)

            tol     WHAM converges if free energy changes < tol
        '''

        self.cvs = cvs
        self.input['kappas']  = np.asarray(kappas).tolist() # input attributes can't be numpy arrays, but can be lists
        self.input['cv_grid'] = np.asarray(cv_grid).tolist()
        self.input['stride']  = stride
        self.input['temp']    = temp

        # Sanity check for cvs
        for cv in self.cvs:
            assert isinstance(cv,colvar.CV)

        # Check if cv_grid is well defined
        try:
            _ = iter(self.input['cv_grid'])
            if isinstance(self.input['cv_grid'],str):
                print(self.input['cv_grid'])
                raise TypeError
        except TypeError:
            raise InputError('Your cv_grid is not iterable! Please define a proper cv_grid.')

        # Check the shape of the cv_grid
        try:
            shape = np.asarray(self.input['cv_grid']).shape
            if len(shape)>1:
                assert len(shape)==2
                assert shape[1]==len(self.cvs)
        except AssertionError:
            raise InputError('Your cv_grid has the wrong shape. It should either be a flat array/list or an array with shape (N,#cvs)')

        # Check the shape of kappas
        try:
            shape = np.asarray(self.input['kappas']).shape
            if len(shape)==0: # if it is just a number
                warnings.warn('You only provided a single kappa value. All kappa values will be equal to this value.')
                self.input['kappas'] = (np.ones((len(self.input['cv_grid']),len(self.cvs))) * self.input['kappas']).tolist()
            else:
                assert shape[-1]==len(self.cvs)
                if len(shape)==1:
                    warnings.warn('You only provided a single kappa value for each cv. The kappa values at each location will be equal to these values.')
                    self.input['kappas'] = (np.ones((len(self.input['cv_grid']),len(self.cvs))) * self.input['kappas']).tolist()
                elif len(shape)>1:
                    assert len(shape)==2
                    assert shape[0]==len(self.input['cv_grid'])
        except AssertionError:
            raise InputError('Your kappas has the wrong shape. It should either be a flat array/list or an array with shape (N,#cvs) or (#cvs,)')


        if h_min is None:
            if len(cvs)>1:
                h_min = np.array([np.min(cvg) for cvg in cv_grid]).tolist()
            else:
                h_min = np.min(cv_grid)

        if h_max is None:
            if len(cvs)>1:
                h_max = np.array([np.max(cvg) for cvg in cv_grid]).tolist()
            else:
                h_max = np.max(cv_grid)

        if h_bins is None:
            if len(cvs)>1:
                h_bins = np.array([len(cvg) for cvg in cv_grid]).tolist()
            else:
                h_bins = len(cv_grid)

        self.input['h_min']       = h_min
        self.input['h_max']       = h_max
        self.input['h_bins']      = h_bins

        self.input['periodicity'] = periodicity
        self.input['tol']         = tol


    def generate_structures_traj(self,job):
        '''
            Generates structure list based on cv grid and provided CV objects using the trajectory data from another job (e.g. MD or MTD job)

            **Arguments**

            job      job object which contains enough snapshots in the region of interest
        '''
        cv_values = np.array([cv.get_cv_values(job) for cv in self.cvs]).T
        idx = np.zeros(len(self.input['cv_grid']),dtype=int)
        max_deviation = 0.
        for n,loc in enumerate(self.input['cv_grid']):
            idx[n] = np.argmin(np.linalg.norm(loc-cv_values,axis=-1))
            max_deviation = max(max_deviation,np.linalg.norm(loc-cv_values,axis=-1)[idx[n]])
        print('The largest deviation is equal to {}'.format(max_deviation))
        self.traj_job = job
        self.traj_job_idx = idx

    def generate_structures_ref(self,f):
        '''
            Generates structure list based on cv grid and reference structure

            **Arguments**

            f     function object f(structure, cv_values) that takes the reference structure object and the target cv values as input and returns the altered structure
        '''

        # Ref job corresponds to template
        assert self.structure is not None
        self.ref_f = f


    def check_overlap(self):
        '''
            Checks overlap between different umbrella simulations. Only works for 1D!

            **Arguments**
        '''
        pt.figure()
        for job_id in self.child_ids:
            job = self.project_hdf5.inspect(job_id)
            pt.plot(job['output/enhanced/trajectory/cv'])
        pt.show()


    def wham(self, h_min, h_max, bins, f_metadata, f_fes, periodicity=None, tol=0.00001):
        '''
            Performs the weighted histogram analysis method to calculate the free energy surface

            **Arguments**

            h_min   lowest value that is taken into account, float or list if more than one cv is biased

            h_max   highest value that is taken into account, float or list if more than one cv is biased
                    if one whole trajectory is outside of these borders an error occurs

            bins    number of bins between h_min and h_max

            periodicity
                    periodicity of the collective variable
                    1D: either a number, 'pi' for angles (2pi periodic) or an empty string ('') for periodicity of 360 degrees
                    2D: either a number, 'pi' for angles (2pi periodic) or 0 if no periodicity is required

            tol     if no free energy value changes between iteration for more than tol, wham is converged
        '''


        # Get wham path for execution
        path = self.path+'_hdf5/'+self.name+'/'
        load_module = get_wham_path()
        wham_script = path+'wham_job.sh'

        if isinstance(h_min,(int,float)):
            cmd = 'wham'
            cmd += ' '
            if not periodicity is None:
                cmd += 'P{} '.format(periodicity)
            cmd += ' '.join(map(str,[h_min,h_max,int(bins),tol,self.input['temp'],0,f_metadata,f_fes]))

        elif isinstance(h_min,list) and len(h_min) == 2:
            cmd =  'wham-2d'
            cmd += ' '
            periodic = ['Px='+str(periodicity[0]) if not periodicity[0] is None else '0', 'Py='+str(periodicity[1]) if not periodicity[1] is None else '0']
            for i in range(2):
                cmd += ' '.join([periodic[i],h_min[i],h_max[i],int(bins[i])])
            cmd += ' '.join([tol,self.input['temp'],0,f_metadata,f_fes,1])
        else:
            raise NotImplementedError()

        with open(wham_script,'w') as g:
            with open(load_module,'r') as f:
                for line in f:
                    g.write(line)
            g.write(cmd)

        # Change permissions (equal to chmod +x)
        st = os.stat(wham_script)
        os.chmod(wham_script, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) # executable by everyone

        # execute wham
        command = ['exec', wham_script]
        out = s._queue_adapter._adapter._execute_command(command, split_output=False, shell=True)

    def get_structure(self, iteration_step=-1):
        '''

        Returns: Structure at free energy minimum

        '''

        # Read minimal energy from fes
        # Read corresponding job
        # return average structure

        raise NotImplementedError()

    def to_hdf(self, hdf=None, group_name=None):
        super(US, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open('input') as hdf5_input:
            self.input.to_hdf(hdf5_input)
            grp = hdf5_input.create_group('cvs')
            for cv in self.cvs:
                cv.to_hdf(grp)

    def from_hdf(self, hdf=None, group_name=None):
        super(US, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open('input') as hdf5_input:
            self.input.from_hdf(hdf5_input)
            cvs = []
            for cv in hdf5_input['cvs'].values():
                cvs.append(colvar.CV().from_hdf(cv))
            self.cvs = cvs

    def collect_output(self):
        # Files to store data of wham
        f_metadata = os.path.join(self.working_directory, 'metadata')
        f_fes      = os.path.join(self.working_directory, 'fes.dat')


        with open(f_metadata, 'w') as f:
            for job_id in self.child_ids:
                job = self.project_hdf5.inspect(job_id)
                print('job_id: ', job_id, job.status)
                loc = convert_val(job['input/generic/enhanced/loc'])
                kappa = convert_val(job['input/generic/enhanced/kappa'],unit=kjmol)
                f.write('{}/COLVAR\t'.format(job.working_directory) + '\t'.join(loc) + '\t' + '\t'.join(kappa) + '\n') # format of colvar needs to be TIME CV1 (CV2)

        # Execute wham code
        self.wham(self.input['h_min'], self.input['h_max'], self.input['h_bins'], f_metadata, f_fes, periodicity=self.input['periodicity'], tol=self.input['tol'])

        # Process output of wham code
        data = np.loadtxt(os.path.join(self.working_directory,'fes.dat'))

        if len(loc) == 1:
            bins = data[:,0]
            fes = data[:,1]
        elif len(loc) == 2:
            bins = data[:,0:2]
            fes = data[:,2]

        with self.project_hdf5.open('output') as hdf5_out:
            hdf5_out['bins'] = bins
            hdf5_out['fes'] = fes

    def plot_1D_FES(self,equilibration_time=None,ax=None,xunit=None,yunit='kjmol',label=None,smooth=False,window_length=11,poly_order=3,verbose=True):
        '''
            Plots the free energy surface for your calculation to the axes object.
            If ax is None a figure object is created for you.

            In case you want a certain equilibration time to be taken into account,
            you can specify it as well, and the free energy is recalculated (but not stored).

            You can also smooth the obtained free energy profile using the Savitzky-Golay filter

            **Arguments**

            equilibration_time
                    the minimal time required for each simulation to equilibrate (in femtoseconds)

            ax      an Axes object of matplotlib in which the generated data is plotted

            xunit,yunit
                    units in which the data is plotted (default unit of y-axis is kJ/mol)

            label   label for the plotted data

            smooth  smooths the obtained free energy profile

            window_length, poly_order
                    argument of the Savitzky-Golay filter

            verbose
                    if True, prints the number of samples that are taken into account
        '''

        # Create figure is no Axes object is parsed
        if ax is None:
            pt.figure()
            ax = pt.gca()

        # If equilibration_time is not None, calculate the new FES
        # Files to store data of wham
        if equilibration_time is not None:
            f_metadata = os.path.join(self.working_directory, 'metadata_eq')
            f_fes      = os.path.join(self.working_directory, 'fes_eq.dat')

            with open(f_metadata, 'w') as f:
                for job_id in self.child_ids:
                    job = self.project_hdf5.inspect(job_id)

                    time = job['output/enhanced/trajectory/time'].reshape(-1,1)
                    cv = job['output/enhanced/trajectory/cv']
                    bias = job['output/enhanced/trajectory/bias'].reshape(-1,1)

                    time_idx = np.argmin(np.abs(time-equilibration_time))
                    if verbose:
                        print('{}/{} samples taken into account.'.format(len(time)-time_idx, len(time)))
                        verbose = False # avoid printing this many times

                    time = time[time_idx:]
                    cv = cv[time_idx:]
                    bias = bias[time_idx:]

                    data = np.concatenate((time,cv,bias),axis=-1)

                    # Write data to COLVAR_eq file
                    np.savetxt(os.path.join(job.working_directory,'COLVAR_eq'), data, fmt='%.5e', delimiter='\t', newline='\n')

                    # Write metadata file
                    loc = convert_val(job['input/generic/enhanced/loc'])
                    kappa = convert_val(job['input/generic/enhanced/kappa'],unit=kjmol)
                    f.write('{}/COLVAR_eq\t'.format(job.working_directory) + '\t'.join(loc) + '\t' + '\t'.join(kappa) + '\n') # format of colvar needs to be TIME CV1 (CV2)


            # Execute wham code
            self.wham(self.input['h_min'], self.input['h_max'], self.input['h_bins'], f_metadata, f_fes, periodicity=self.input['periodicity'], tol=self.input['tol'])

            # Process output of wham code
            data = np.loadtxt(f_fes)

            if len(self.cvs) == 1:
                bins = data[:,0]
                fes = data[:,1]
            elif len(self.cvs) == 2:
                bins = data[:,0:2]
                fes = data[:,2]
        else:
            bins = self['output/bins']
            fes = self['output/fes']

        # Smooth data if needed
        if smooth:
            from scipy.signal import savgol_filter
            fes = savgol_filter(fes, window_length, poly_order)

        # Scale data if needed
        if xunit is not None:
            bins = bins/eval(xunit)

        if yunit is not None:
            fes = fes*kjmol/eval(yunit)

        # Plot the data on the Axes
        ax.plot(bins,fes,label=label)


    def restart(self):
        for job in self.iter_jobs():
            if not job.status.finished:
                job.run(delete_existing_job=True)
        self.run_static() # refresh state
