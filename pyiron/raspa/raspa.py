# coding: utf-8
from pyiron.base.generic.parameters import GenericParameters
from pyiron.base.job.generic import GenericJob as GenericJobCore
from pyiron.base.settings.generic import Settings

from molmod.units import *
from molmod.constants import *
from molmod.periodic import periodic as pt
import subprocess

import os, posixpath, numpy as np, h5py, matplotlib.pyplot as pp, stat, warnings, glob, re


s = Settings()
KELVIN_TO_KJ_PER_MOL = float(8.314464919 / 1000.0)  #exactly the same as Raspa

class Raspa(GenericJobCore):
    def __init__(self, project, job_name):
        super(Raspa, self).__init__(project, job_name)
        self.__name__ = "Raspa"
        self._executable_activate(enforce=True)
        self.input = RaspaInput()
        self.systems = None
        self.components = None
        self.definitions = None

    def write_input(self):
        input_dict = {
            'simulationtype': self.input['SimulationType'],
            'components': self.components,
            'systems': self.systems,
            'definitions': self.definitions,
            'settings': self.input['settings'],
        }
        write_input(input_dict,working_directory=self.working_directory)


    def collect_output(self):
        output_files = glob.glob(posixpath.join(self.working_directory,'Output/*/*.data'))
        output_dict = {}
        for out in output_files:
            output_dict.update(collect_output(output_file=out,ncomponents=len(self.components)))
        with self.project_hdf5.open("output") as hdf5_output:
            for k, v in output_dict.items():
                hdf5_output[k] = v

    def to_hdf(self, hdf=None, group_name=None):
        super(Raspa, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.to_hdf(hdf5_input)

    def from_hdf(self, hdf=None, group_name=None):
        super(Raspa, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.from_hdf(hdf5_input)

    def log(self):
        for fname in glob.glob(posixpath.join(self.working_directory, 'Output/*')):
            with open(fname) as f:
                print('Output for {}\n'.format(fname.split('/')[-2]))
                print(f.read())
                print('#'*20) # to separate

    def add_system(self,name,type,fileloc=None,options={}):
        '''
            Add a system to the raspa job.

            **Arguments**

            name (string): Name of the system
            type (string): 'Box' or 'Framework'
            fileloc (string): (only for framework type) path to the corresponding cif file
            options (dict): the options for the system
        '''

        if self.systems is None:
            self.systems = {} # dicts respect input order

        self.systems[(name,type)] = options

        if type=='Framework':
            assert fileloc is not None
            if self.definitions is None:
                self.definitions = {}
            self.definitions['system_{}_fname'.format(name)] = fileloc


    def add_component(self,name,fileloc,options={}):
        '''
            Add a system to the raspa job.

            **Arguments**

            name (string): Name of the molecule
            fileloc (string):  path to the corresponding molecule.def file
            options (dict): the options for the system
        '''

        if self.components is None:
            self.components = {} # dicts respect input order

        self.components[name] = options

        if self.definitions is None:
            self.definitions = {}
        self.definitions['component_{}_fname'.format(name)] = fileloc

    def add_definition(self,fname):
        name = fname.split('/')[-1].split('.')[0]
        print('Adding {}'.format(name))

        if self.definitions is None:
            self.definitions = {}
        self.definitions['def_{}'.format(name)] = fname


    def calc_minimize(self, max_iter=1000, n_print=5):
        '''
            Set up an optimization calculation.

            **Arguments**

            cell (bool): Set True if the cell also has to be optimized
            gpos_tol (float): Convergence criterion for RMS of gradients towards atomic coordinates
            dpos_tol (float): Convergence criterion for RMS of differences of atomic coordinates
            grvecs_tol (float): Convergence criterion for RMS of gradients towards cell parameters
            drvecs_tol (float): Convergence criterion for RMS of differences of cell parameters
            max_iter (int): Maximum number of optimization steps
            n_print (int):  Print frequency
        '''
        self.input['SimulationType'] = 'minimization'

        settings = {}
        settings['NumberOfCycles'] = max_iter
        settings['PrintEvery'] = n_print

        if self.input['settings'] is None:
            self.input['settings'] = {}

        self.input['settings'].update(settings)

        super(Raspa, self).calc_minimize(max_iter=max_iter, n_print=n_print)


    def calc_static(self):
        '''
            Set up a static force field calculation.
        '''

        self.input['SimulationType'] = 'minimization'

        settings = {}
        settings['NumberOfCycles'] = 1

        if self.input['settings'] is None:
            self.input['settings'] = {}

        self.input['settings'].update(settings)

        super(Raspa, self).calc_static()


    def calc_md(self, temperature=None, pressure=None, nsteps=1000, time_step=0.5*femtosecond, n_print=5,
                timecon_thermo=150.0*femtosecond, timecon_baro=150.0*femtosecond, ensemble='NVE'):

        '''
            Set up an MD calculation.

            **Arguments**

            temperature (None/float/list of floats): Target temperature(s) for each system.
            pressure (None/float/list of floats): Target pressure(s) for each system.
            nsteps (int): Number of md steps
            init_cycles (int): The number of cycles used to initialize the system using Monte Carlo. This can be used for both
                               Monte Carlo as well as Molecular Dynamics to quickly equilibrate the positions of the atoms in the system.
            eq_cycles (int): For Molecular Dynamics it is the number of MD steps to equilibrate the velocities in the systems.
                             After this equilibration the production run is started. For Monte Carlo, in particular CFMC, the
                             equilibration-phase is used to measure the biasing factors.
            time_step (float): Step size between two steps.
            n_print (int):  Print frequency
            timecon_thermo (float): The time associated with the thermostat adjusting the temperature.
            timecon_baro (float): The time associated with the barostat adjusting the temperature.
            ensemble (list of NVE|NVT|NPT|NPH|NPTPR|NPHPR or dict with keys ['init','run']):
                     Sets the ensemble for each system, if a single ensemble is given it is applied for all systems.
                     If a dict is provided different ensembles can be used for the initialization and production run.


        '''
        self.input['SimulationType'] = 'moleculardynamics'

        settings = {}
        settings['NumberOfCycles'] = nsteps
        settings['NumberOfInitializationCycles'] = init_cycles
        settings['NumberOfEquilibrationCycles'] = eq_cycles
        settings['PrintEvery'] = n_print

        settings['TimeStep'] = time_step

        if temperature is not None:
            settings['ExternalTemperature'] = temperature
        if pressure is not None:
            settings['ExternalPressure'] = pressure

        settings['TimeScaleParameterThermostat'] = timecon_thermo
        settings['TimeScaleParameterBarostat'] = timecon_baro


        if isinstance(ensemble,dict):
            assert 'init' in ensemble and 'run' in ensemble
            settings['InitEnsemble'] = ensemble['init']
            settings['RunEnsemble'] = ensemble['run']
        else:
            settings['Ensemble'] = ensemble

        if self.input['settings'] is None:
            self.input['settings'] = {}

        self.input['settings'].update(settings)

        super(Raspa, self).calc_md(temperature=temperature, pressure=pressure, n_ionic_steps=nsteps,
                                   time_step=time_step, n_print=n_print,
                                   temperature_damping_timescale=timecon_thermo,
                                   pressure_damping_timescale=timecon_baro)


class RaspaInput(GenericParameters):
    def __init__(self, input_file_name=None):
        super(RaspaInput, self).__init__(input_file_name=input_file_name,table_name="input_inp",comment_char="#")

    def load_default(self):
        '''
        There are no default settings for the Raspa plugin!
        '''
        input_str = """
"""
        self.load_string(input_str)


def write_input(input_dict, working_directory='.'):

    # Load dictionary
    simulationtype = input_dict['simulationtype']
    settings       = input_dict['settings'] # computational settings
    systems        = input_dict['systems'] # the box(es) or framework(s) in which the molecule(s) are simulated
    components     = input_dict['components'] # the component(s) under investigation
    definitions    = input_dict['definitions'] # the definition(s) which need to be copied

    assert simulationtype.lower() in ['montecarlo','moleculardynamics','minimization','barriercrossing','numerical']
    warnings.warn('The pyiron plugin for Raspa is still under construction, and the output will likely not be parsed correctly.')

    if systems is not None:
        assert isinstance(systems,dict)
        assert all(isinstance(v, dict) for k,v in systems.items())

    if components is not None:
        assert isinstance(components,dict)
        assert all(isinstance(v, dict) for k,v in components.items())


    def write_dict(f,d,width=0,leading_whitespaces=0):
        for k,v in d.items():
            if not isinstance(v,list):
                v = [v]
            f.write(' '*leading_whitespaces + '{:<{width}} {}\n'.format(k,' '.join([str(val) for val in v]),width=width))

    len_setting_names = [len(k) for k,v in settings.items()] if len(settings)>0 else 0
    len_system_settings_names = [len(k) if len(system)>0 else 0 for name,system in systems.items() for k,v in system.items()]
    len_component_settings_names = [len(k) if len(component)>0 else 0 for name,component in components.items() for k,v in component.items()]
    max_width = max([max(len_setting_names), max(len_system_settings_names), max(len_component_settings_names)+12]) # +12 comes from the length of 'Component i '

    # Write to file
    with open(os.path.join(working_directory, 'simulation.input'), 'w') as f:
        f.write('{: <{width}} {}\n'.format('SimulationType',simulationtype,width=max_width))

        # Write all settings
        if len(settings)>0:
            write_dict(f,settings,width=max_width)
        f.write('\n')

        # Write systems (assume only boxes or only frameworks are present)
        for n,(key,system) in enumerate(systems.items()):
            f.write('{: <{width}} {}\n'.format(key[1],n,width=max_width))
            if key[1]=='Framework':
                f.write('{: <{width}} {}\n'.format('FrameworkName',key[0],width=max_width))
            write_dict(f,system,width=max_width)
        f.write('\n')

        # Write components
        for n,(name,component) in enumerate(components.items()):
            f.write('{} {} {: <{width}} {}\n'.format('Component',n,'MoleculeName',name,width=max_width-12))
            write_dict(f,component,width=max_width-12,leading_whitespaces=12)

        f.write('\n\n')

    # Copy all the definition files
    if len(definitions)>0:
        for k,definition in definitions.items():
            new_file_loc = definition.split('/')[-1]
            with open(definition,'r') as f:
                text = f.read()
            with open(os.path.join(working_directory, new_file_loc), 'w') as f:
                f.write(text)


#####################################
# OUTPUT PARSING, BASED ON AIIDA CODE

def parse_block1(flines, result_dict, prop, value=1, unit=2, dev=4):
    """Parse block.

    Parses blocks that look as follows::

        Average Volume:
        =================
            Block[ 0]        12025.61229 [A^3]
            Block[ 1]        12025.61229 [A^3]
            Block[ 2]        12025.61229 [A^3]
            Block[ 3]        12025.61229 [A^3]
            Block[ 4]        12025.61229 [A^3]
            ------------------------------------------------------------------------------
            Average          12025.61229 [A^3] +/-            0.00000 [A^3]

    """
    for line in flines:
        if 'Average' in line:
            result_dict[prop + '_average'] = float(line.split()[value])
            result_dict[prop + '_unit'] = re.sub(r"[{}()\[\]]", '', line.split()[unit])
            result_dict[prop + '_dev'] = float(line.split()[dev])
            break

def parse_block_energy(flines, res_dict, prop):
    """Parse energy block.

    Parse block that looks as follows::

        Average Adsorbate-Adsorbate energy:
        ===================================
            Block[ 0] -443.23204         Van der Waals: -443.23204         Coulomb: 0.00000            [K]
            Block[ 1] -588.20205         Van der Waals: -588.20205         Coulomb: 0.00000            [K]
            Block[ 2] -538.43355         Van der Waals: -538.43355         Coulomb: 0.00000            [K]
            Block[ 3] -530.00960         Van der Waals: -530.00960         Coulomb: 0.00000            [K]
            Block[ 4] -484.15106         Van der Waals: -484.15106         Coulomb: 0.00000            [K]
            ------------------------------------------------------------------------------
            Average   -516.80566         Van der Waals: -516.805659        Coulomb: 0.00000            [K]
                  +/- 98.86943                      +/- 98.869430               +/- 0.00000            [K]
    """
    for line in flines:
        if 'Average' in line:
            res_dict["energy_{}_tot_average".format(prop)] = float(line.split()[1]) * KELVIN_TO_KJ_PER_MOL
            res_dict["energy_{}_vdw_average".format(prop)] = float(line.split()[5]) * KELVIN_TO_KJ_PER_MOL
            res_dict["energy_{}_coulomb_average".format(prop)] = float(line.split()[7]) * KELVIN_TO_KJ_PER_MOL
        if '+/-' in line:
            res_dict["energy_{}_tot_dev".format(prop)] = float(line.split()[1]) * KELVIN_TO_KJ_PER_MOL
            res_dict["energy_{}_vdw_dev".format(prop)] = float(line.split()[3]) * KELVIN_TO_KJ_PER_MOL
            res_dict["energy_{}_coulomb_dev".format(prop)] = float(line.split()[5]) * KELVIN_TO_KJ_PER_MOL
            return

def parse_lines_with_component(res_components, components, line, prop):
    """Parse lines that contain components"""
    # self.logger.info("analysing line: {}".format(line))
    for i, component in enumerate(components):
        if '[' + component + ']' in line:
            words = line.split()
            res_components[i][prop + '_unit'] = re.sub(r'[{}()\[\]]', '', words[-1])
            res_components[i][prop + '_dev'] = float(words[-2])
            res_components[i][prop + '_average'] = float(words[-4])

def parse_base_output(output_abs_path, system_name, ncomponents):
    """Parse RASPA output file: it is divided in different parts, whose start/end is carefully documented."""

    # manage block of the first type
    # --------------------------------------------------------------------------------------------
    BLOCK_1_LIST = [
        ("Average temperature:", "temperature", (1, 2, 4), 0), # misleading property!
        ("Average Pressure:", "pressure", (1, 2, 4), 0), # misleading property!
        ("Average Volume:", "cell_volume", (1, 2, 4), 0),
        ("Average Density:", "adsorbate_density", (1, 2, 4), 0),
        ("Average Heat Capacity", "framework_heat_capacity", (1, 2, 4), 0), # misleading property!
        ("Enthalpy of adsorption:", "enthalpy_of_adsorption", (1, 4, 3), 4),
        ("Tail-correction energy:", "tail_correction_energy", (1, 2, 4), 0),
        ("Total energy:", "total_energy", (1, 2, 4), 0), # not important property!
    ]

    # block of box properties.
    BOX_PROP_LIST = [
        ("Average Box-lengths:", 'box'),
    ]


    # manage energy reading
    # --------------------------------------------------------------------------------------------
    ENERGY_CURRENT_LIST = [
        ("Host/Adsorbate energy:", "host/ads", "tot"),
        ("Host/Adsorbate VDW energy:", "host/ads", "vdw"),
        ("Host/Adsorbate Coulomb energy:", "host/ads", "coulomb"),
        ("Adsorbate/Adsorbate energy:", "ads/ads", "tot"),
        ("Adsorbate/Adsorbate VDW energy:", "ads/ads", "vdw"),
        ("Adsorbate/Adsorbate Coulomb energy:", "ads/ads", "coulomb"),
    ]

    ENERGY_AVERAGE_LIST = [("Average Adsorbate-Adsorbate energy:", "ads/ads"),
                           ("Average Host-Adsorbate energy:", "host/ads")]

    # manage lines with components
    # --------------------------------------------------------------------------------------------
    LINES_WITH_COMPONENT_LIST = [
        (" Average Widom Rosenbluth-weight:", "widom_rosenbluth_factor"),
        (" Average chemical potential: ", "chemical_potential"),
        (" Average Henry coefficient: ", "henry_coefficient"),
        (" Average  <U_gh>_1-<U_h>_0:", "adsorption_energy_widom"),
    ]


    warnings = []
    res_per_component = []
    for i in range(ncomponents):
        res_per_component.append({})
    result_dict = {'exceeded_walltime': False}

    with open(output_abs_path, "r") as fobj:

        # 1st parsing part: input settings
        # --------------------------------
        # from: start of file
        # to: "Current (initial full energy) Energy Status"

        icomponent = 0
        component_names = []
        res_cmp = res_per_component[0]
        for line in fobj:
            if "Component" in line and "(Adsorbate molecule)" in line:
                component_names.append(line.split()[2][1:-1])
            # Consider to change it with parse_line()
            if "Conversion factor molecules/unit cell -> mol/kg:" in line:
                res_cmp['conversion_factor_molec_uc_to_mol_kg'] = float(line.split()[6])
                res_cmp['conversion_factor_molec_uc_to_mol_kg_unit'] = "(mol/kg)/(molec/uc)"
            # this line was corrected in Raspa's commit c1ad4de (Nov19), since "gr/gr" should read "mg/g"
            if "Conversion factor molecules/unit cell -> gr/gr:" in line \
            or "Conversion factor molecules/unit cell -> mg/g:" in line:
                res_cmp['conversion_factor_molec_uc_to_mg_g'] = float(line.split()[6])
                res_cmp['conversion_factor_molec_uc_to_mg_g_unit'] = "(mg/g)/(molec/uc)"
            if "Conversion factor molecules/unit cell -> cm^3 STP/gr:" in line:
                res_cmp['conversion_factor_molec_uc_to_cm3stp_gr'] = float(line.split()[7])
                res_cmp['conversion_factor_molec_uc_to_cm3stp_gr_unit'] = "(cm^3_STP/gr)/(molec/uc)"
            if "Conversion factor molecules/unit cell -> cm^3 STP/cm^3:" in line:
                res_cmp['conversion_factor_molec_uc_to_cm3stp_cm3'] = float(line.split()[7])
                res_cmp['conversion_factor_molec_uc_to_cm3stp_cm3_unit'] = "(cm^3_STP/cm^3)/(molec/uc)"
            if "MolFraction:" in line:
                res_cmp['mol_fraction'] = float(line.split()[1])
                res_cmp['mol_fraction_unit'] = "-"
            if "Partial pressure:" in line:
                res_cmp['partial_pressure'] = float(line.split()[2])
                res_cmp['partial_pressure_unit'] = "Pa"
            if "Partial fugacity:" in line:
                res_cmp['partial_fugacity'] = float(line.split()[2])
                res_cmp['partial_fugacity_unit'] = "Pa"
                icomponent += 1
                if icomponent < ncomponents:
                    res_cmp = res_per_component[icomponent]
            if "Framework Density" in line:
                result_dict['framework_density'] = line.split()[2]
                result_dict['framework_density_unit'] = re.sub(r'[{}()\[\]]', '', line.split()[3])
            if "Current (initial full energy) Energy Status" in line:
                break

        # 2nd parsing part: initial and final configurations
        # --------------------------------------------------
        # from: "Current (initial full energy) Energy Status"
        # to: "Average properties of the system"

        reading = 'initial'
        result_dict['energy_unit'] = 'kJ/mol'

        for line in fobj:
            # Understand if it is the initial or final "Current Energy Status" section
            if "Current (full final energy) Energy Status" in line:
                reading = 'final'

            # Read the entries of "Current Energy Status" section
            if reading:
                for parse in ENERGY_CURRENT_LIST:
                    if parse[0] in line:
                        result_dict['energy_{}_{}_{}'.format(parse[1], parse[2],
                                                             reading)] = float(line.split()[-1]) * KELVIN_TO_KJ_PER_MOL
                        if parse[1] == "ads/ads" and parse[2] == "coulomb":
                            reading = None

            if "Average properties of the system" in line:
                break

        # 3rd parsing part: average system properties
        # --------------------------------------------------
        # from: "Average properties of the system"
        # to: "Number of molecules"

        for line in fobj:
            for parse in BLOCK_1_LIST:
                if parse[0] in line:
                    parse_block1(fobj, result_dict, parse[1], *parse[2])
                    # I assume here that properties per component are present furhter in the output file.
                    # so I need to skip some lines:
                    skip_nlines_after = parse[3]
                    while skip_nlines_after > 0:
                        line = next(fobj)
                        skip_nlines_after -= 1
                    for i, cmpnt in enumerate(component_names):
                        # The order of properties per molecule is the same as the order of molecules in the
                        # input file. So if component name was not found in the next line, I break the loop
                        # immidiately as there is no reason to continue it
                        line = next(fobj)
                        if cmpnt in line:
                            parse_block1(fobj, res_per_component[i], parse[1], *parse[2])
                        else:
                            break
                        skip_nlines_after = parse[3]
                        while skip_nlines_after > 0:
                            line = next(fobj)
                            skip_nlines_after -= 1

                    continue  # no need to perform further checks, propperty has been found already
            for parse in ENERGY_AVERAGE_LIST:
                if parse[0] in line:
                    parse_block_energy(fobj, result_dict, prop=parse[1])
                    continue  # no need to perform further checks, propperty has been found already
            for parse in BOX_PROP_LIST:
                if parse[0] in line:
                    # parse three cell vectors
                    parse_block1(fobj, result_dict, prop='box_ax', value=2, unit=3, dev=5)
                    parse_block1(fobj, result_dict, prop='box_by', value=2, unit=3, dev=5)
                    parse_block1(fobj, result_dict, prop='box_cz', value=2, unit=3, dev=5)
                    # parsee angles between the cell vectors
                    parse_block1(fobj, result_dict, prop='box_alpha', value=3, unit=4, dev=6)
                    parse_block1(fobj, result_dict, prop='box_beta', value=3, unit=4, dev=6)
                    parse_block1(fobj, result_dict, prop='box_gamma', value=3, unit=4, dev=6)

            if "Number of molecules:" in line:
                break

        # 4th parsing part: average molecule properties
        # --------------------------------------------------
        # from: "Number of molecules"
        # to: end of file

        icomponent = 0
        for line in fobj:
            # Consider to change it with parse_line?
            if 'Average loading absolute [molecules/unit cell]' in line:
                res_per_component[icomponent]['loading_absolute_average'] = float(line.split()[5])
                res_per_component[icomponent]['loading_absolute_dev'] = float(line.split()[7])
                res_per_component[icomponent]['loading_absolute_unit'] = 'molecules/unit cell'
            elif 'Average loading excess [molecules/unit cell]' in line:
                res_per_component[icomponent]['loading_excess_average'] = float(line.split()[5])
                res_per_component[icomponent]['loading_excess_dev'] = float(line.split()[7])
                res_per_component[icomponent]['loading_excess_unit'] = 'molecules/unit cell'
                icomponent += 1
            if icomponent >= ncomponents:
                break

        for line in fobj:
            for to_parse in LINES_WITH_COMPONENT_LIST:
                if to_parse[0] in line:
                    parse_lines_with_component(res_per_component, component_names, line, to_parse[1])

        # Assigning to None all the quantities that are meaningless if not running a Widom insertion calculation
        for res_comp in res_per_component:
            for prop in ["henry_coefficient", "widom_rosenbluth_factor", "chemical_potential"]:
                if res_comp["{}_dev".format(prop)] == 0.0:
                    res_comp["{}_average".format(prop)] = None
                    res_comp["{}_dev".format(prop)] = None

            # The section "Adsorption energy from Widom-insertion" is not showing in the output if no widom is performed
            if not "adsorption_energy_widom_average" in res_comp:
                res_comp["adsorption_energy_widom_unit"] = "kJ/mol"
                res_comp["adsorption_energy_widom_dev"] = None
                res_comp["adsorption_energy_widom_average"] = None

    return_dictionary = {"general": result_dict, "components": {}}

    for name, value in zip(component_names, res_per_component):
        return_dictionary["components"][name] = value

    # Parsing all the warning that are printed in the output file, avoiding redoundancy
    with open(output_abs_path, "r") as fobj:
        for line in fobj:
            if "WARNING" in line:
                warning_touple = (system_name, line)
                if warning_touple not in warnings:
                    warnings.append(warning_touple)
    return return_dictionary, warnings


def flatten_dict(dd, separator ='/', prefix =''):
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }

def collect_output(output_file,ncomponents=1):
    name = output_file.split('/')[-2]
    output_dict,_ = parse_base_output(output_file,name,ncomponents)
    output_dict = flatten_dict(output_dict)
    
    return output_dict
