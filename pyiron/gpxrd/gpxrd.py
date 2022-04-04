# coding: utf-8
import os, posixpath, h5py, stat, warnings, bisect, glob
import numpy as np
import matplotlib.pyplot as pt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from molmod.units import *
from molmod.constants import *
from scipy.optimize import least_squares

from pyiron.atomistics.structure.atoms import Atoms
from pyiron.atomistics.job.atomistic import AtomisticGenericJob
from pyiron.base.settings.generic import settings

from pyiron.gpxrd.input import GPXRDInput, InputWriter

"""
This is a plugin for the pyobjcryst code, based on the gpxrdpy wrapper found at
https://github.com/SanderBorgmans/gpxrdpy
"""


class GPXRD(AtomisticGenericJob):
    def __init__(self, project, job_name):
        super(GPXRD, self).__init__(project, job_name)
        self.__name__ = "GPXRD"
        self._executable_activate(enforce=True)
        self.input = GPXRDInput()

    @property
    def reference_pattern(self):
        if hasattr(self, '_reference_pattern'):
            return self._reference_pattern
        else:
            print('The reference pattern was not yet assigned.')

    def set_reference_pattern(self, pattern, skiprows=0):
        """
            Sets the reference pattern attribute based on a provided 2D array or filename
            When it is a 2D array, also creates a tsv file and assigns this new tsv file
            as reference_pattern input dict value if it was not set

            ***Args***
            pattern
                2D array (ttheta, intensity) or fname (string)
            skiprows
                when providing a filename, skip this number of rows (e.g. header lines)
                when reading the data
        """
        # Small sanity check
        if skiprows is None:
            skiprows = 0

        if isinstance(pattern,str):
            if os.path.exists(pattern):
                self.input['refpattern'] = pattern
                self.input['skiprows'] = skiprows
                self._reference_pattern = np.loadtxt(pattern,skiprows=skiprows)
            else:
                raise ValueError('The file does not exist {}'.format(pattern))
        elif isinstance(pattern,np.ndarray):
            fname = posixpath.join(self.working_directory,'ref.tsv')
            self.input['refpattern'] = fname
            self.input['skiprows'] = 0
            self._reference_pattern = pattern
            np.savetxt(fname, pattern, fmt='%.5f', delimiter='\t')
        else:
            raise ValueError('Did not receive the correct data type {}, expected str or np.ndarray.'.format(type(pattern)))

    def plot(self, scale='optimal', verbose=True, fname=None):
        """
            Check whether there are significant peaks outside the reference XRD range
            Assumes that you have used the full_pattern option

            ***Args***
            plot
                whether to plot the XRD
            fname
                file location for the plot
        """
        fig = pt.figure()
        ax1 = pt.gca()

        ttheta_calc = self.get("output/ttheta_calc")
        int_calc = self.get("output/int_calc")

        ttheta_ref = self.get("output/ttheta_ref")
        int_ref = self.get("output/int_ref")

        ax1.plot(ttheta_calc,int_calc-int_calc.min(),lw=1,label='calc')
        if int_ref is not None:
            # If there is a reference pattern, perform a comparison
            stat_res = self.compare(np.array([ttheta_calc,int_calc]).T, np.array([ttheta_ref,int_ref]).T,scale=scale,verbose=verbose,plot=False)
            ax1.plot(ttheta_ref,(int_ref-int_ref.min())/stat_res['scalefactor'],lw=1,label='reference')
        ax1.set_xlabel('2θ (°)')
        ax1.set_ylabel('Intensity (a.u.)')

        ax1.tick_params(
            axis='y',          # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the left edge are off
            right=False,         # ticks along the right edge are off
            labelleft=False) # labels along the left edge are off

        ax1.ticklabel_format(axis='y', style='plain', useOffset=False) # avoid scientific notation and offsets
        ax1.legend(bbox_to_anchor=(1.1,.5), loc='center left',frameon=False)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        if fname is None:
            fig.tight_layout()
            pt.show()
        else:
            pt.savefig(posixpath.join(self.working_directory, fname+'.pdf'),bbox_inches='tight')
        pt.close()

    @staticmethod
    def _unaccounted_peaks_function(pattern1,pattern2,PEAK_FACTOR=3):
        # Pattern2 is the reference
        ttheta1, p1 = pattern1[:,0], pattern1[:,1]
        ttheta2, p2 = pattern2[:,0], pattern2[:,1]

        if np.array_equal(ttheta1,ttheta2):
            return False # if the arrays are equal, we don't need to check anything
        else:
            if np.any(ttheta1<min(ttheta2)) and max(p1[ttheta1<min(ttheta2)]) > max(p1[ttheta1>min(ttheta2)])/PEAK_FACTOR:
                warnings.warn("There is a significant diffraction peak in the omitted 2theta range! Statistical comparison can give biased results.", UserWarning)
                return True
        return False

    def unaccounted_peaks(self,PEAK_FACTOR=3):
        """
            Check whether there are significant peaks outside the reference XRD range
            Assumes that you have used the full_pattern option

            ***Args***
            plot
                whether to plot the XRD
            fname
                file location for the plot
        """
        calc_pattern = np.array([self.get("output/ttheta_calc"), self.get("output/int_calc")]).T
        ref_pattern = np.array([self.get("output/ttheta_ref"), self.get("output/int_ref")]).T

        return self._unaccounted_peaks_function(calc_pattern,ref_pattern,PEAK_FACTOR=PEAK_FACTOR)

    @staticmethod
    def compare(pattern1,pattern2,scale='optimal',verbose=True,plot=False,fname=None):
        """
            Compare two patterns and calculate the relevant metrics

            ***Args***
            pattern1,pattern2
                two arrays with (ttheta,int)

            scale
                one of "false", "optimal" , "max", will scale the two patterns such that:
                    - false: no scale
                    - optimal: a minimal Rwp is obtained
                    - max: both maxima will be normalized to 1

            plot
                whether to plot the comparison
            fname
                file location for the plot
        """

        ttheta1, p1 = pattern1[:,0], pattern1[:,1]
        ttheta2, p2 = pattern2[:,0], pattern2[:,1]

        # If the two theta ranges do not match, check if they are commensurable (and whether there are significant unaccounted peaks)
        unaccounted_peaks = False
        if not ttheta1.size==ttheta2.size or not np.allclose(ttheta1,ttheta2):
            unaccounted_peaks = GPXRD._unaccounted_peaks_function(pattern1,pattern2)

            # take overlapping range
            minimum = max(ttheta1.min(),ttheta2.min())
            maximum = min(ttheta1.max(),ttheta2.max())
            step = max(ttheta1[1]-ttheta1[0], ttheta2[1]-ttheta2[0])
            ttheta1,p1,ttheta2,p2 = GPXRD.make_commensurable(ttheta1,p1,ttheta2,p2,minimum,maximum,step)

        # Calculate scale factor (p2 is reference)
        if scale is not 'false':
            scalefactor = p2.max()/p1.max()
            if scale is 'optimal':
                scalefactor = FitScaleFactorForRw(p1,p2,scalefactor,verbose=verbose)
        else:
            scalefactor = 1.

        p1 *= scalefactor

        # Compare data
        stat_res = GPXRD.statistical_comparison(ttheta1,p1,p2,verbose=verbose,unaccounted_peaks=unaccounted_peaks)
        stat_res['scalefactor'] = scalefactor
        stat_res['unaccounted_peaks'] = unaccounted_peaks

        if plot:
            fig = pt.figure()
            ax1 = pt.gca()
            ax1.plot(ttheta1,p1-p1.min(),lw=1,label='pattern 1')
            ax1.plot(ttheta2,p2-p2.min(),lw=1,label='pattern 2')
            ax1.set_xlabel('2θ (°)')
            ax1.set_ylabel('Intensity (a.u.)')

            ax1.tick_params(
                axis='y',          # changes apply to the y-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the left edge are off
                right=False,         # ticks along the right edge are off
                labelleft=False) # labels along the left edge are off

            ax1.ticklabel_format(axis='y', style='plain', useOffset=False) # avoid scientific notation and offsets

            ax1.legend(bbox_to_anchor=(1.1,.5), loc='center left',frameon=False)
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

            if fname is None:
                fig.tight_layout()
                pt.show()
            else:
                pt.savefig(fname+'.pdf',bbox_inches='tight')
            pt.close()

        return stat_res

    @staticmethod
    def remove_background(reference,locs=None,bkg_range=None,bkg_points=10,uniform=True,plot=False,fname=None,skiprows=0):
        """
            Remove the background the provided reference pattern

            ***Args***
            reference
                file name of reference data (file with two columns, no header)
                or a 2D array with (ttheta,intensity)
            skiprows
                number of rows to skip in reading file (when reference is a filename)
            locs
                node locations in degrees, default linspace in range with bkg_points
            bkg_range
                range for background points, default full ttheta range of reference pattern
            bkg_points
                set number of background points in fit
            uniform
                if false, the grid points are locally optimized towards the minima for a better interpolation
            plot
                plot the analysis and difference plot
            fname
                file location for the plot

            ***Returns***
                a new reference pattern (ttetha,intensity) array
        """

        # Interactively use the pyobjcryst code here, it should not be necessary to run a separate job for this
        try:
            import pyobjcryst
        except ImportError:
            raise ImportError("The pyobjcryst package is required for this")

        pp = pyobjcryst.powderpattern.PowderPattern()

        # Add the reference data
        if isinstance(reference,str) and os.path.exists(reference):
            pp.ImportPowderPattern2ThetaObs(reference,skiprows) # skip no lines
        elif isinstance(reference,np.ndarray):
            pp.SetPowderPatternX(reference[:,0]*deg)
            pp.SetPowderPatternObs(reference[:,1])
        ttheta = pp.GetPowderPatternX()/deg
        reference = pp.GetPowderPatternObs()

        # Background
        if locs is not None:
            bx = np.array(locs)
        else:
            if bkg_range is not None:
                assert len(bkg_range)==2
                bx=np.linspace(bkg_range[0],bkg_range[1],bkg_points)
            else:
                bx=np.linspace(ttheta.min(),ttheta.max(),bkg_points)
            if not uniform:
                # adapt bx to minima of reference pattern in each neighbourhood (optional)
                idx = [np.argmin(np.abs(ttheta-bxi)) for bxi in bx]
                step = (idx[1] - idx[0])//4
                for n in range(len(bx)):
                    mn = -step if idx[n]>step else 0
                    mx = step if n<(len(idx)-1) else 0
                    bx[n] = ttheta[idx[n]+ mn + np.argmin(reference[idx[n]+mn:idx[n]+mx])]

        bx*=deg
        by=np.zeros(bx.shape)
        b=pp.AddPowderPatternBackground()
        b.SetInterpPoints(bx,by)
        b.UnFixAllPar()
        b.OptimizeBayesianBackground()

        no_bg = pp.GetPowderPatternObs()-pp.GetPowderPatternCalc()
        no_bg -= np.min(no_bg)

        # Plot the difference
        if plot:
            # Consider the difference for the delta plot
            height = no_bg.max()-no_bg.min()

            fig = pt.figure()
            ax1 = pt.gca()
            ax1.plot(ttheta,pp.GetPowderPatternCalc(),lw=1,label='background')
            ax1.plot(ttheta,reference - reference.min(),lw=1,label='reference')
            ax1.plot(ttheta,no_bg-height*0.1,color='g',lw=1,label=r'$\Delta$')
            ax1.set_xlabel('2θ (°)')
            ax1.set_ylabel('Intensity (a.u.)')

            # Format the plot
            lims = pt.xlim()
            ax1.hlines(0,lims[0],lims[1],lw=0.1)
            ax1.set_xlim(lims)

            lims = pt.ylim()
            ax1.vlines(bx/deg,lims[0],lims[1],lw=0.1)

            ax1.tick_params(
                axis='y',          # changes apply to the y-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the left edge are off
                right=False,         # ticks along the right edge are off
                labelleft=False) # labels along the left edge are off

            ax1.ticklabel_format(axis='y', style='plain', useOffset=False) # avoid scientific notation and offsets

            ax1.legend(bbox_to_anchor=(1.1,.5), loc='center left',frameon=False)
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

            if fname is None:
                fig.tight_layout()
                pt.show()
            else:
                pt.savefig(fname+'.pdf',bbox_inches='tight')
            pt.close()

        if fname is not None:
            with open(fname + '.tsv','w') as f:
                for i in range(len(ttheta)):
                    f.write("{:4.3f}\t{:10.8f}\n".format(ttheta[i],no_bg[i]))

        return np.array([ttheta, no_bg]).T

    @staticmethod
    def make_commensurable(ttheta1,p1,ttheta2,p2,minimum,maximum,step):
        # Create new arrays
        trange = np.arange(minimum,maximum+step,step)
        ttheta1_commensurate = np.zeros_like(trange)
        ttheta2_commensurate = np.zeros_like(trange)
        p1_commensurate = np.zeros_like(trange)
        p2_commensurate = np.zeros_like(trange)

        for n,t in enumerate(trange):
            index1 = np.argmin(np.abs(ttheta1-t))
            index2 = np.argmin(np.abs(ttheta2-t))

            ttheta1_commensurate[n] = ttheta1[index1]
            ttheta2_commensurate[n] = ttheta2[index2]

            p1_commensurate[n] = p1[index1]
            p2_commensurate[n] = p2[index2]

        if not np.allclose(ttheta1_commensurate,ttheta2_commensurate):
            warnings.warn('Tried to make the two theta ranges commensurate, but they still differ! Interpolating values')

            for n,t in enumerate(trange):
                # find surrounding indices
                idx1_right = bisect.bisect_left(ttheta1,t)
                idx2_right = bisect.bisect_left(ttheta2,t)

                ttheta1_commensurate[n] = t
                ttheta2_commensurate[n] = t

                # do linear interpolation
                p1_interp = np.interp(t,ttheta1[np.clip(idx1_right-1,a_min=0,a_max=None):np.clip(idx1_right+1,a_min=None,a_max=len(ttheta1))],
                                             p1[np.clip(idx1_right-1,a_min=0,a_max=None):np.clip(idx1_right+1,a_min=None,a_max=len(ttheta1))])
                p2_interp = np.interp(t,ttheta2[np.clip(idx2_right-1,a_min=0,a_max=None):np.clip(idx2_right+1,a_min=None,a_max=len(ttheta2))],
                                             p2[np.clip(idx2_right-1,a_min=0,a_max=None):np.clip(idx2_right+1,a_min=None,a_max=len(ttheta2))])
                p1_commensurate[n] = p1_interp
                p2_commensurate[n] = p2_interp


        return ttheta1_commensurate,p1_commensurate,ttheta2_commensurate,p2_commensurate

    @staticmethod
    def statistical_comparison(ttheta,p1,p2,verbose=True,unaccounted_peaks=False):
        """
            Statistical comparison, using p2 as reference data
            ttheta should be provided in degrees
        """
        p1 -= np.min(p1)
        p2 -= np.min(p2)

        if np.sum(np.abs(p2)) == 0: return # this happens for virtual reference pattern

        low_p1 = p1[ttheta<=10]
        low_p2 = p2[ttheta<=10]

        high_p1 = p1[ttheta>=10]
        high_p2 = p2[ttheta>=10]

        if verbose:
            print("")
            print("Statistical comparison")
            if unaccounted_peaks:
                print("WARNING: the calculated pattern has significant peaks outside the shown range!")
            print("------------------------------------------------------------------")
            print("Quantity \t\t |     Full    | LA (min-10) | HA (10-max)")
            print("------------------------------------------------------------------")
            print("R factor (abs) \t\t | {:10.9f} | {:10.9f} | {:10.9f}".format(R_Factor(p1,p2,abs=True),R_Factor(low_p1,low_p2,abs=True),R_Factor(high_p1,high_p2,abs=True)))
            print("R factor (squared)\t | {:10.9f} | {:10.9f} | {:10.9f}".format(R_Factor(p1,p2),R_Factor(low_p1,low_p2),R_Factor(high_p1,high_p2)))
            print("Weighted R factor \t | {:10.9f} | {:10.9f} | {:10.9f}".format(R_Factor(p1,p2,weighted=True),R_Factor(low_p1,low_p2,weighted=True),R_Factor(high_p1,high_p2,weighted=True)))
            print("Similarity index \t | {:10.9f} | {:10.9f} | {:10.9f}".format(similarity_index(p1,p2),similarity_index(low_p1,low_p2),similarity_index(high_p1,high_p2)))

        results = {
            'R_factor_abs' : R_Factor(p1,p2,abs=True),
            'R_factor_sq': R_Factor(p1,p2),
            'R_wp': R_Factor(p1,p2,weighted=True),
            'SI' : similarity_index(p1,p2)
        }

        return results

    def load_h5_files(self, h5_files, start, stop, num_frames):
        """
            Load a list of h5 files into the job for a dynamic calculation
            Assume these are yaff h5 files (if only using pyiron this should not be used)
        """

        # Load the initial frame as the job.structure (so that we can access the symbols,indices,... at any frame)
        # Then only save the positions and cell vectors

        with h5py.File(h5_files[0], mode='r') as h5:
            self.structure = Atoms(
                positions=h5['system/pos']/angstrom,
                numbers=h5['system/numbers'],
                cell=h5['system/rvecs']/angstrom,
            )

        positions = np.zeros((0,len(self.structure),3))
        cells = np.zeros((0,3,3))

        for h5_fname in h5_files:
            with h5py.File(h5_fname, mode='r') as h5:
                positions = np.vstack((positions, h5['trajectory/pos']))
                cells = np.vstack((cells, h5['trajectory/cell']))

        if stop < 0:
            stop = positions.shape[0] + stop + 1
        else:
            stop = min(stop, positions.shape[0])
        if start < 0:
            start = positions.shape[0] + start + 1

        if num_frames is None:
            num_frames = start-stop

        self.input['num_frames'] = num_frames
        frames = np.linspace(start,stop,num_frames,dtype=int,endpoint=False)

        # We will store the 'input' trajectory data as output so that we can visualize the trajectory
        with self._hdf5.open("output") as hdf5_output:
            hdf5_output['generic/positions'] = positions[frames]
            hdf5_output['generic/cells'] = cells[frames]
            hdf5_output['generic/indices'] = np.vstack([self.structure.indices] * positions.shape[0])

    def load_cifs(self, cif_files, start, stop, num_frames):
        """
            Load a list of cif files into the job for a dynamic calculation
        """
        from ase.io import read
        from pyiron import ase_to_pyiron

        self.structure = ase_to_pyiron(read(cif_files[0]))

        if stop < 0:
            stop = len(cif_files) + stop + 1
        else:
            stop = min(stop, len(cif_files))
        if start < 0:
            start = len(cif_files) + start + 1

        if num_frames is None:
            num_frames = start-stop

        self.input['num_frames'] = num_frames
        frames = np.linspace(start,stop,num_frames,dtype=int,endpoint=False)
        cif_files = [cif_files[n] for n in frames]

        positions = np.zeros((len(cif_files),len(self.structure),3))
        cells = np.zeros((len(cif_files),3,3))

        for n,cif_fname in enumerate(cif_files):
            cif = ase_to_pyiron(read(cif_fname))
            positions[n] = cif.positions
            cells[n] =  cif.cell.array

        # We will store the 'input' trajectory data as output so that we can visualize the trajectory
        with self._hdf5.open("output") as hdf5_output:
            hdf5_output['generic/positions'] = positions
            hdf5_output['generic/cells'] = cells
            hdf5_output['generic/indices'] = np.vstack([self.structure.indices] * positions.shape[0])

    def load_structures(self, structures, start, stop, num_frames):
        """
            Load a list of structure objects into the job for a dynamic calculation
        """

        self.structure = structures[0]

        if stop < 0:
            stop = len(structures) + stop + 1
        else:
            stop = min(stop, len(structures))
        if start < 0:
            start = len(structures) + start + 1

        if num_frames is None:
            num_frames = start-stop

        self.input['num_frames'] = num_frames
        frames = np.linspace(start,stop,num_frames,dtype=int,endpoint=False)
        structures = [structures[n] for n in frames]

        positions = np.zeros((len(structures),len(self.structure),3))
        cells = np.zeros((len(structures),3,3))

        for n,structure in enumerate(structures):
            positions[n] = structure.positions
            cells[n] =  structure.cell.array

        # We will store the 'input' trajectory data as output so that we can visualize the trajectory
        with self._hdf5.open("output") as hdf5_output:
            hdf5_output['generic/positions'] = positions
            hdf5_output['generic/cells'] = cells
            hdf5_output['generic/indices'] = np.vstack([self.structure.indices] * positions.shape[0])

    def load_jobs(self, jobs, start, stop, num_frames):
        """
            Load a list of job objects into the job for a dynamic calculation
        """

        self.structure = jobs[0].structure

        positions = np.zeros((0,len(self.structure),3))
        cells = np.zeros((0,3,3))

        for job in jobs:
            positions = np.vstack((positions, job.output.positions))
            cells = np.vstack((cells, job.output.cells))

        if stop < 0:
            stop = positions.shape[0] + stop + 1
        else:
            stop = min(stop, positions.shape[0])
        if start < 0:
            start = positions.shape[0] + start + 1

        if num_frames is None:
            num_frames = start-stop

        self.input['num_frames'] = num_frames
        frames = np.linspace(start,stop,num_frames,dtype=int,endpoint=False)

        # We will store the 'input' trajectory data as output so that we can visualize the trajectory
        with self._hdf5.open("output") as hdf5_output:
            hdf5_output['generic/positions'] = positions[frames]
            hdf5_output['generic/cells'] = cells[frames]
            hdf5_output['generic/indices'] = np.vstack([self.structure.indices] * positions.shape[0])

    def load_trajectory(self, data, start=0, stop=-1, num_frames=None):
        """
            Load trajectory data into the job object for a dynamic calculation
            The data can be:
                - h5 file name
                - a pyiron job object with trajectory data
                - a list of (h5 files, cif files, structure objects or pyiron jobs), all list elements should have the same type
            Based on the provided data, the relevant load function will be executed

            start,stop,num_frames
                defines the structures that are taken into account
        """

        # Adapt the jobtype automatically
        self.input['jobtype'] = 'dynamic'

        if isinstance(data,str):
            assert os.path.exists(data)
            assert os.path.splitext(data)[1] == '.h5'
            self.load_h5_files([data], start, stop, num_frames)
        elif isinstance(data,AtomisticGenericJob):
            self.load_jobs([data], start, stop, num_frames)
        elif isinstance(data,list):
            if all([isinstance(d,str) for d in data]):
                if all([os.path.splitext(d)[1] == '.h5' for d in data]):
                    self.load_h5_files(data, start, stop, num_frames)
                elif all([os.path.splitext(d)[1] == '.cif' for d in data]):
                    self.load_cifs(data, start, stop, num_frames)
                else:
                    raise ValueError("The provided data list contains elements with a non-supported data type (only h5 filenames, cif filenames, or structure objects)")
            elif all([isinstance(d,Atoms) for d in data]):
                self.load_structures(data, start, stop, num_frames)
            elif all([isinstance(d,AtomisticGenericJob) for d in data]):
                self.load_jobs(data, start, stop, num_frames)
            else:
                raise ValueError("The provided data list contains elements with a non-supported data type (only h5 filenames, cif filenames, or structure objects)")
        else:
            raise ValueError("The provided data did not have a correct type, {}".format(type(data)))

    def write_input(self):
        input_dict = {
            'jobtype': self.input['jobtype'], # static or dynamic
            'wavelength': self.input['wavelength'],
            'peakwidth': self.input['peakwidth'],
            'numpoints': self.input['numpoints'],
            'max2theta': self.input['max2theta'],
            'refpattern': self.input['refpattern'],
            'skiprows': self.input['skiprows'],
            'detail_fhkl': self.input['detail_fhkl'],
            'rad_type': self.input['rad_type'],
            'full_pattern': self.input['full_pattern'],
            'save_fhkl': self.input['save_fhkl'],
            'num_frames': self.input['num_frames'],
            }

        # Sanity checks
        assert self.input['jobtype'] in ['static', 'dynamic']
        assert self.input['rad_type'].upper() in ['NEUTRON', 'XRAY']

        # Write input files
        input_writer = InputWriter(input_dict,working_directory=self.working_directory)
        input_writer.write_calcscript()
        input_writer.write_jobscript()

        # If there is a reference pattern, copy it to the working directory
        if self.input['refpattern'] is not None:
            import shutil
            shutil.copyfile(self.input['refpattern'], posixpath.join(self.working_directory,'ref.tsv'))

            # if this dict value was set directly, still set the property
            if not hasattr(self, '_reference_pattern'):
                self.set_reference_pattern(self.input['refpattern'], skiprows=self.input['skiprows'])

        # Write structure(s) to cif file(s) for reading
        if self.input['jobtype'] == 'static':
            self.structure.write(posixpath.join(self.working_directory,'input.cif'))
        elif self.input['jobtype'] == 'dynamic':
            # Write structures for each frame
            os.makedirs(posixpath.join(self.working_directory,'cifs'),exist_ok=True)
            structure = self.structure.copy()
            for n in range(self.input['num_frames']):
                structure.positions = self['output/generic/positions'][n]
                structure.cells = self['output/generic/cells'][n]
                structure.write(posixpath.join(self.working_directory,'cifs','{}.cif'.format(n)))

    def collect_output(self):
        # Read the relevant tsv files generated by pyobjcryst
        output_dict = {}

        def _load_frame(output_dict,fname,save_hkl,key_prefix=""):
            ttheta_data = np.loadtxt(fname + '.dat',skiprows=1)
            q_data = np.loadtxt(fname + '_q.dat',skiprows=1)

            output_dict[key_prefix+'ttheta_calc'] = ttheta_data[:,0]
            output_dict[key_prefix+'int_calc'] = ttheta_data[:,1]
            output_dict[key_prefix+'q_calc'] = q_data[:,0]

            if self.input['save_fhkl']:
                fhkl_data = np.loadtxt(fname + '_fhkl.dat',skiprows=2)
                output_dict[key_prefix+'fhkl/hkl'] = fhkl_data[:,:3]
                output_dict[key_prefix+'fhkl/f_hkl_2'] = fhkl_data[:,3]
                output_dict[key_prefix+'fhkl/re_f'] = fhkl_data[:,4]
                output_dict[key_prefix+'fhkl/im_f'] = fhkl_data[:,5]
                output_dict[key_prefix+'fhkl/theta'] = fhkl_data[:,6]
                output_dict[key_prefix+'fhkl/1_2d'] = fhkl_data[:,7]

        if self.input['jobtype'] == 'static':
            _load_frame(output_dict,posixpath.join(self.working_directory,'output'),self.input['save_fhkl'])

        elif self.input['jobtype'] == 'dynamic':
            for n in range(self.input['num_frames']):
                # can't use numbers as h5 group name
                _load_frame(output_dict,posixpath.join(self.working_directory,'frames/{}'.format(n)),self.input['save_fhkl'],key_prefix="frames/frame_{}/".format(n))

            # Average diffraction data to store in main folder
            output_dict['ttheta_calc'] = output_dict['frames/frame_0/ttheta_calc']
            output_dict['int_calc'] = np.average([output_dict['frames/frame_{}/int_calc'.format(n)] for n in range(self.input['num_frames'])],axis=0)
            output_dict['q_calc'] = np.average([output_dict['frames/frame_{}/q_calc'.format(n)] for n in range(self.input['num_frames'])],axis=0)

        else:
            raise ValueError("This jobtype is not supported")

        # Read the reference pattern if it exists and store it in the output
        if hasattr(self, '_reference_pattern'):
            output_dict['ttheta_ref'] = self.reference_pattern[:,0]
            output_dict['int_ref'] = self.reference_pattern[:,1]

            # if there is an reference pattern, also store the statistical comparison
            stat_res = self.compare(np.array([output_dict['ttheta_calc'],output_dict['int_calc']]).T,
                                    np.array([output_dict['ttheta_ref'],output_dict['int_ref']]).T)

            for k,v in stat_res.items():
                output_dict['statistical_comparison/{}'.format(k)] = v

        with self.project_hdf5.open("output") as hdf5_output:
            for k, v in output_dict.items():
                hdf5_output[k] = v

    def to_hdf(self, hdf=None, group_name=None):
        super(GPXRD, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.structure.to_hdf(hdf5_input)
            self.input.to_hdf(hdf5_input)

    def from_hdf(self, hdf=None, group_name=None):
        super(GPXRD, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.from_hdf(hdf5_input)
            self.structure = Atoms().from_hdf(hdf5_input)

            # Also initialise the reference pattern if it exists, use the copied version since we know this file must exist
            if self.input['refpattern'] is not None:
                self.set_reference_pattern(posixpath.join(self.working_directory,'ref.tsv'),skiprows=self.input['skiprows'])

    def log(self):
        # the log file is likely empty
        with open(posixpath.join(self.working_directory, 'gpxrd.log')) as f:
            print(f.read())

# Utility functions
def similarity_index(p1,p2):
    return (p1*p2).sum()/(np.sqrt((p1*p1).sum()) * np.sqrt((p2*p2).sum()))

def R_Factor(icalc,iobs,weighted=False,abs=False):
    # R factor of 0 means a perfect fit
    if weighted:
        wi = [1./iobsi if iobsi>0 else 0. for iobsi in iobs]
        return np.sqrt((wi*(iobs-icalc)*(iobs-icalc)).sum() /(wi*(iobs*iobs)).sum())
    else:
        if abs:
            return (np.abs(iobs-icalc)).sum() / iobs.sum()
        else:
            return np.sqrt(((iobs-icalc)*(iobs-icalc)).sum() /((iobs*iobs)).sum())

def FitScaleFactorForRw(p1,p2,guess,verbose=False):
    """
        Fit scale factor, while keeping reference pattern p2 fixed as reference
    """
    p1 -= np.min(p1)
    p2 -= np.min(p2)
    if np.sum(np.abs(p2)) == 0: return guess # this happens for virtual reference pattern

    def error(x,p1,p2):
        #return R_Factor(p1*x[0],p2,abs=True)
        return R_Factor(p1*x[0],p2,weighted=True)
        #return similarity_index(p1*x[0],p2)

    res = least_squares(error,1.,args=(p1*guess,p2),bounds=(1e-7,np.inf))
    if verbose:
        print("Guess = {}, fit = {}".format(guess,guess*res.x[0]))
    return guess*res.x[0]
