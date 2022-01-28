# coding: utf-8

# Module defining Yaff units for each property


import pint
import numpy as np

from pyiron.generic.units import PyironUnitRegistry, BaseUnitConverter



class SimpleYaffUnits:
    # simple class to instantiate all required units for the Yaff class
    def __init__(self, unit_system='atomic'):
        # Possible systems: 'Planck', 'SI', 'US', 'atomic', 'cgs', 'imperial', 'mks'
        base_registry = pint.UnitRegistry(system=unit_system)
        code_registry = PyironUnitRegistry()


        # Define code quantities
        code_registry.add_quantity(quantity="dimensionless_integer", unit=base_registry.dimensionless, data_type=int)
        code_registry.add_quantity(quantity="dimensionless_float", unit=base_registry.dimensionless, data_type=float)
        code_registry.add_quantity(quantity="masses", unit=base_registry.amu, data_type=float)
        code_registry.add_quantity(quantity="charges", unit=base_registry.e, data_type=float)
        code_registry.add_quantity(quantity="positions", unit=base_registry.angstrom, data_type=float)


        code_registry.add_quantity(quantity="time", unit=base_registry.fs, data_type=float)
        code_registry.add_quantity(quantity="volume", unit=base_registry.angstrom**3, data_type=float)
        code_registry.add_quantity(quantity="energy", unit=base_registry.eV, data_type=float)

        # Define code labels
        code_registry.add_labels(labels=["energy_tot", "energy_pot"], quantity="energy")



        self.unit_convertor = BaseUnitConverter(base_registry,code_registry)
