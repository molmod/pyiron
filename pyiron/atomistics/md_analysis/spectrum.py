# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.


import numpy as np
import matplotlib.pyplot as pt

from yaff.log import log
from molmod.units import *
from molmod.constants import *


class Spectrum(object):
    def __init__(self, job, start=0, end=-1, step=1, bsize=4096, select=None, path='output/generic/velocities', weights=None, unit=angstrom/femtosecond):
        """
           **Arguments**

           job
               job object that contains an MD trajectory of the structure

           **Optional arguments:**

           start
                The first sample to be considered for analysis. This may be
                negative to indicate that the analysis should start from the
                -start last samples.

           end
                The last sample to be considered for analysis. This may be
                negative to indicate that the last -end sample should not be
                considered.

           step
                The spacing between the samples used for the analysis

           bsize
                The size of the blocks used for individual FFT calls.

           select
                A list of atom indexes that are considered for the computation
                of the spectrum. If not given, all atoms are used.

           path
                The path of the dataset that contains the time dependent data in
                the HDF5 file. The first axis of the array must be the time
                axis. The spectra are summed over the other axes.

           weights
                If not given, the spectrum is just a simple sum of contributions
                from different time-dependent functions. If given, a linear
                combination is made based on these weights.
           unit
                The unit of the path data

           The max_sample argument from get_slice is not used because the choice
           step value is an important parameter: it is best to choose step*bsize
           such that it coincides with a part of the trajectory in which the
           velocities (or other data) are continuous.

           The block size should be set such that it corresponds to a decent
           resolution on the frequency axis, i.e. 33356 fs of MD data
           corresponds to a resolution of about 1 cm^-1. The step size should be
           set such that the highest frequency is above the highest relevant
           frequency in the spectrum, e.g. a step of 10 fs corresponds to a
           frequency maximum of 3336 cm^-1. The total number of FFT's, i.e.
           length of the simulation divided by the block size multiplied by the
           number of time-dependent functions in the data, determines the noise
           reduction on the (the amplitude of) spectrum. If there is sufficient
           data to perform 10K FFT's, one should get a reasonably smooth
           spectrum.

           Depending on the FFT implementation in numpy, it may be interesting
           to tune the bsize argument. A power of 2 is typically a good choice.
        """

        self.job = job

        # Compute spectrum
        self._calculate(self.job, start, end, step, bsize, select, path, weights, unit)


    def _calculate(self, job, start, end, step, bsize, select, path, weights, unit):
        spectrum = Spectrum_calculator(job, start, end, step, bsize, select, path, weights, unit)
        self.freqs = spectrum.freqs
        self.amps = spectrum.amps
        self.ac = spectrum.ac
        self.nfft = spectrum.nfft


    def plot(self, fn_png=None, xlim=None, verticals=None, thermostat=None, ndof=None):
        """
           **Optional arguments**

           fn_png
                location to save the resulting figure

            xlim
               Provide the xlim for the plot in units of [1/cm]

            verticals
               Array containing the thermostat timeconstant of the system at the
               first index (in atomic units), followed by the wavenumbers (in 1/cm)
               of the system
        """
        xunit = lightspeed/centimeter
        xlabel = 'Wavenumber [1/cm]'

        pt.clf()
        pt.plot(self.freqs/xunit, self.amps)
        if verticals is not None:
            thermo_freq = 1.0/verticals[0]/lightspeed*centimeter
            #plot frequencies original system, and coupling to thermostat
            for i in np.arange(1, len(verticals)):
                pt.axvline(verticals[i], color='r', ls='--')
                pt.axvline(verticals[i] + thermo_freq, color='g', ls='--')
                pt.axvline(verticals[i] - thermo_freq, color='g', ls='--')
        if thermostat is not None and ndof is not None:
            thermo_freq = 1.0/thermostat/lightspeed*centimeter
            pt.axvline(thermo_freq, color='k', ls='--')
            pt.axvline(thermo_freq/np.sqrt(ndof), color='k', ls='--')
            pt.axvline(thermo_freq+thermo_freq/np.sqrt(ndof), color='r', ls='--')
            pt.axvline(thermo_freq+2.0*thermo_freq/np.sqrt(ndof), color='r', ls='--')

        if xlim is not None:
            pt.xlim(xlim[0], xlim[1])
        else:
            pt.xlim(0, self.freqs[-1]/xunit)

        pt.xlabel(xlabel)
        pt.ylabel('Amplitude')
        if fn_png is not None:
            pt.savefig(fn_png)
        pt.show()

    def plot_ac(self, fn_png=None, time_unit=femtosecond, xlabel='Time [a.u.]'):
        pt.clf()
        pt.plot(self.time/time_unit, self.ac/self.ac[0])
        pt.xlabel(xlabel)
        pt.ylabel('Autocorrelation')
        if fn_png is not None:
            pt.savefig(fn_png)
        pt.show()


class Spectrum_calculator(object):
    """
        This class allows for the spectral analysis and calculation of autocorrelation functions for a job object.
    """

    def __init__(self, job, start, end, step, bsize, select, path, weights, unit):
        self.job = job
        self.bsize = bsize
        self.select = select
        self.weights = weights
        self.ssize = self.bsize//2+1 # the length of the spectrum array
        self.amps = np.zeros(self.ssize, float)
        self.nfft = 0 # the number of fft calls, for statistics

        runlength = len(self.job['output/generic/time'])
        self.start = start if start>=0 else start+runlength
        self.end = end if end >=0 else end+runlength
        self.step = step

        self.timestep = (self.job['output/generic/time'][self.start+self.step] - self.job['output/generic/time'][self.start])*femtosecond # time is in femtoseconds
        self.work = np.zeros(self.bsize)
        self.freqs = np.arange(self.ssize)/(self.timestep*self.bsize)
        self.time = np.arange(self.ssize)*self.timestep

        # Compute the amplitudes of the spectrum
        ds_signal = job[path]*unit

        current = self.start
        stride = self.step*self.bsize
        work = np.zeros(self.bsize, float)
        while current <= self.end - stride:
            for indexes in self._iter_indexes(ds_signal):
                work = ds_signal[(slice(current, current+stride, self.step),) + indexes]
                self.amps += self._get_weight(indexes)*abs(np.fft.rfft(work))**2
                self.nfft += 1
            current += stride
        # Compute related arrays
        self.ac = np.fft.irfft(self.amps)[:self.ssize]

    def _iter_indexes(self, array):
        if self.select is None:
            for indexes in np.ndindex(array.shape[1:]):
                yield indexes
        else:
            for i0 in self.select:
                for irest in np.ndindex(array.shape[2:]):
                    yield (i0,) + irest

    def _get_weight(self, indexes):
        if self.weights is None:
            return 1.0
        else:
            return self.weights[indexes]
