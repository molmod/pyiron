# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron.atomistics.job.atomistic import AtomisticGenericJob
import warnings

__author__ = "Jan Janssen"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH - " \
                "Computational Materials Design (CM) Department"
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class GenericDFTJob(AtomisticGenericJob):
    def __init__(self, project, job_name):
        super(GenericDFTJob, self).__init__(project, job_name)
        self._generic_input['fix_symmetry'] = True

    @property
    def encut(self):
        return self.plane_wave_cutoff

    @encut.setter
    def encut(self, val):
        self.plane_wave_cutoff = val

    @property
    def xc(self):
        return self.exchange_correlation_functional

    @xc.setter
    def xc(self, val):
        self.exchange_correlation_functional = val

    @property
    def plane_wave_cutoff(self):
        raise NotImplementedError("The encut property is not implemented for this code.")

    @plane_wave_cutoff.setter
    def plane_wave_cutoff(self, val):
        raise NotImplementedError("The encut property is not implemented for this code.")

    @property
    def spin_constraints(self):
        raise NotImplementedError("The spin_constraints property is not implemented for this code.")

    @spin_constraints.setter
    def spin_constraints(self, val):
        raise NotImplementedError("The spin_constraints property is not implemented for this code.")

    @property
    def exchange_correlation_functional(self):
        raise NotImplementedError("The exchange property is not implemented for this code.")

    @exchange_correlation_functional.setter
    def exchange_correlation_functional(self, val):
        raise NotImplementedError("The exchange property is not implemented for this code.")

    def get_k_mesh_by_cell(self, cell=None, kpoints_per_angstrom=1):
        if cell is None:
            cell = self.structure.cell
        latlens = [np.linalg.norm(lat) for lat in cell]
        kmesh = np.rint(np.array([2 * np.pi / ll for ll in latlens]) * kpoints_per_angstrom)
        if kmesh.min()<=0:
           warnings.warn('kpoint per angstrom too low')
        return [int(k) for k in kmesh]

    @property
    def fix_spin_constraint(self):
        return self._generic_input['fix_spin_constraint']

    @fix_spin_constraint.setter
    def fix_spin_constraint(self, boolean):
        if not isinstance(boolean, bool):
            raise AssertionError()
        self._generic_input['fix_spin_constraint'] = boolean

    @property
    def fix_symmetry(self):
        return self._generic_input['fix_symmetry']

    @fix_symmetry.setter
    def fix_symmetry(self, boolean):
        if not isinstance(boolean, bool):
            raise AssertionError()
        self._generic_input['fix_symmetry'] = boolean

    def get_structure(self, iteration_step=-1, wrap_atoms=True):
        """
        Gets the structure from a given iteration step of the simulation (MD/ionic relaxation). For static calculations
        there is only one ionic iteration step
        Args:
            iteration_step (int): Step for which the structure is requested
            wrap_atoms (bool): True if the atoms are to be wrapped back into the unit cell

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: The required structure


        """
        snapshot = super(GenericDFTJob).get_structure(iteration_step=iteration_step, wrap_atoms=wrap_atoms)
        spins = self.get("output/generic/dft/atom_spins")
        if spins is not None:
            snapshot.set_initial_magnetic_moments(spins[iteration_step])
        return snapshot

    def set_mixing_parameters(self, method=None, n_pulay_steps=None, density_mixing_parameter=None, spin_mixing_parameter=None):
        raise NotImplementedError("set_mixing_parameters is not implemented for this code.")

    #def set_kmesh_density(self, kspace_per_in_ang=0.10):
    #    mesh = self._get_k_mesh_by_cell(self.structure, kspace_per_in_ang)
    #    self.set_kpoints(mesh=mesh, scheme='MP', center_shift=None, symmetry_reduction=True, manual_kpoints=None,
    #                     weights=None, reciprocal=True)

    def _set_kpoints(self, mesh=None, scheme='MP', center_shift=None, symmetry_reduction=True, manual_kpoints=None,
                     weights=None, reciprocal=True):
        raise NotImplementedError("The set_kpoints function is not implemented for this code.")

    def set_kpoints(self, mesh=None, scheme='MP', center_shift=None, symmetry_reduction=True, manual_kpoints=None,
                    weights=None, reciprocal=True, kpoints_per_angstrom=None):
        """
        Function to setup the k-points

        Args:
            mesh (list): Size of the mesh (ignored if scheme is not set to 'MP' or kpoints_per_angstrom is set)
            scheme (str): Type of k-point generation scheme (MP/GP(gamma point)/Manual/Line)
            center_shift (list): Shifts the center of the mesh from the gamma point by the given vector in relative coordinates
            symmetry_reduction (boolean): Tells if the symmetry reduction is to be applied to the k-points
            manual_kpoints (list/numpy.ndarray): Manual list of k-points
            weights(list/numpy.ndarray): Manually supplied weights to each k-point in case of the manual mode
            reciprocal (bool): Tells if the supplied values are in reciprocal (direct) or cartesian coordinates (in
            reciprocal space)
            kpoints_per_angstrom (float): Number of kpoint per angstrom in each direction
        """
        if kpoints_per_angstrom is not None:
            if mesh is not None:
                warnings.warn('mesh value is overwritten by kpoints_per_angstrom')
            mesh = self.get_k_mesh_by_cell(kpoints_per_angstrom=kpoints_per_angstrom)
        if mesh is not None:
            if np.min(mesh) <= 0:
                raise ValueError('mesh values must be larger than 0')
        if center_shift is not None:
            if np.min(center_shift)<0 or np.max(center_shift)>1:
                warnings.warn('center_shift is given in relative coordinates')
        self._set_kpoints(mesh=mesh, scheme=scheme, center_shift=center_shift, symmetry_reduction=symmetry_reduction, manual_kpoints=manual_kpoints,
                          weights=weights, reciprocal=reciprocal)

    def calc_static(self, electronic_steps=400, algorithm=None, retain_charge_density=False,
                    retain_electrostatic_potential=False):
        self._generic_input['fix_symmetry'] = True
        super(GenericDFTJob, self).calc_static()

    def calc_minimize(self, electronic_steps=400, ionic_steps=100, max_iter=None, pressure=None, algorithm=None,
                      retain_charge_density=False, retain_electrostatic_potential=False, ionic_energy=None,
                      ionic_forces=None, volume_only=False):
        self._generic_input['fix_symmetry'] = True
        super(GenericDFTJob, self).calc_minimize(max_iter=max_iter, pressure=pressure)

    def calc_md(self, temperature=None, n_ionic_steps=1000, n_print=1, time_step=1.0, retain_charge_density=False,
                retain_electrostatic_potential=False, **kwargs):
        self._generic_input['fix_symmetry'] = False
        super(GenericDFTJob, self).calc_md(temperature=temperature, n_ionic_steps=n_ionic_steps, n_print=n_print,
                                           time_step=time_step)

    # Backward compatibility
    def get_encut(self):
        return self.encut

    def set_encut(self, encut):
        """
        Sets the plane-wave energy cutoff
        Args:
            encut (float): The energy cutoff in eV
        """
        self.plane_wave_cutoff = encut

    def set_exchange_correlation_functional(self, exchange_correlation_functional):
        self.exchange_correlation_functional = exchange_correlation_functional

    def set_empty_states(self, n_empty_states=None):
        raise NotImplementedError("The set_empty_states function is not implemented for this code.")
