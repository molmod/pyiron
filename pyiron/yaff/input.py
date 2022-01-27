# coding: utf-8

# Input module for Yaff
# Consists of write functions to create the initial files for a Yaff calculation based on the job type

import posixpath, h5py, inspect
import numpy as np

from molmod.units import *
from molmod.constants import *

from yaff import System, log
log.set_level(log.silent)

from pyiron.base.generic.parameters import GenericParameters


class YaffInput(GenericParameters):
    def __init__(self, input_file_name=None):
        super(YaffInput, self).__init__(input_file_name=input_file_name,table_name="input_inp",comment_char="#")

    def load_default(self):
        '''
        Loading the default settings for the input file.
        '''

        input_str = inspect.cleandoc("""\
        rcut 28.345892008818783 #(FF) real space cutoff
        alpha_scale 3.2 #(FF) scale for ewald alpha parameter
        gcut_scale 1.5 #(FF) scale for ewald reciprocal cutoff parameter
        smooth_ei True #(FF) smoothen cutoff for real space electrostatics
        gpos_rms 1e-8 #(OPT) convergence criterion for RMS of gradients towards atomic coordinates
        dpos_rms 1e-6 #(OPT) convergence criterion for RMS of differences of atomic coordinates
        grvecs_rms 1e-8 #(OPT) convergence criterion for RMS of gradients towards cell parameters
        drvecs_rms 1e-6 #(OPT) convergence criterion for RMS of differences of cell parameters
        hessian_eps 1e-3 #(HESS) step size in finite differences for numerical derivatives of the forces
        timestep 41.341373336646825 #(MD) time step for verlet scheme
        temp None #(MD) temperature
        press None #(MD) pressure
        timecon_thermo 4134.137333664683 #(MD) timeconstant for thermostat
        timecon_baro 41341.37333664683 #(MD) timeconstant for barostat
        nsteps 1000 #(GEN) number of steps for opt or md
        h5step 5 #(GEN) stores system properties every h5step
        """)
        self.load_string(input_str)


class InputWriter(object):
    '''
    The Yaff InputWriter is called to write the Yaff specific input files.
    '''

    common = inspect.cleandoc("""#! /usr/bin/python

    from molmod.units import *
    from yaff import *
    import h5py, numpy as np

    #Setting up system and force field
    system = System.from_file('system.chk')
    ff = ForceField.generate(system, 'pars.txt', rcut={rcut}*angstrom, alpha_scale={alpha_scale}, gcut_scale={gcut_scale}, smooth_ei={smooth_ei})

    #Setting up output
    f = h5py.File('output.h5', mode='w')
    hdf5 = HDF5Writer(f, step={h5step})
    r = h5py.File('restart.h5', mode='w')
    restart = RestartWriter(r, step=10000)
    hooks = [hdf5, restart]

    #Setting up simulation
    """) + '\n'

    plumed_part = inspect.cleandoc("""
    plumed = ForcePartPlumed(ff.system, fn='plumed.dat')
    ff.add_part(plumed)
    hooks.append(plumed)
    """)+ '\n\n'

    tail = """\n"""

    def __init__(self,input_dict,working_directory='.'):
        self.input_dict = input_dict
        self.working_directory = working_directory

    def write_chk(self):
        # collect data and initialize Yaff system
        cell = None
        if 'cell' in self.input_dict.keys() and self.input_dict['cell'] is not None and self.input_dict['cell'].volume > 0:
            cell = self.input_dict['cell']*angstrom
        system = System(self.input_dict['numbers'], self.input_dict['pos']*angstrom, ffatypes=self.input_dict['ffatypes'], ffatype_ids=self.input_dict['ffatype_ids'], bonds=self.input_dict['bonds'], rvecs=cell, masses=self.input_dict['masses']*amu)

        if self.input_dict['bonds'] is None:
            system.detect_bonds()
            print('Warning: no bonds could be read and were automatically detected.')
        # write dictionary to MolMod CHK file
        system.to_file(posixpath.join(self.working_directory,'system.chk'))

    def write_pars(self):
        with open(posixpath.join(self.working_directory,'pars.txt'), 'w') as f:
            for line in self.input_dict['ffpars']:
                f.write(line)

    def write_jobscript(self,jobtype=None):
        if jobtype == 'sp':
            self.write_ysp()
        elif jobtype == 'opt':
            self.write_yopt()
        elif jobtype == 'opt_cell':
            self.write_yopt_cell()
        elif jobtype == 'hess':
            self.write_yhess()
        elif jobtype == 'nve':
            self.write_ynve()
        elif jobtype == 'nvt':
            self.write_ynvt()
        elif jobtype == 'npt':
            self.write_ynpt()
        elif jobtype == 'scan':
            self.write_scan()
        else:
            raise IOError('Invalid job type for Yaff job, received %s' %jobtype)

    def write_yopt(self):
        body = InputWriter.common.format(
            rcut=self.input_dict['rcut']/angstrom, alpha_scale=self.input_dict['alpha_scale'],
            gcut_scale=self.input_dict['gcut_scale'], smooth_ei=self.input_dict['smooth_ei'],
            h5step=1,
        )
        body += "dof = CartesianDOF(ff, gpos_rms={gpos_rms}, dpos_rms={dpos_rms})".format(
            gpos_rms=self.input_dict['gpos_rms'],dpos_rms=self.input_dict['dpos_rms']
        )
        body += inspect.cleandoc("""
        opt = CGOptimizer(dof, hooks=[hdf5])
        opt.run({nsteps})
        system.to_file('opt.chk')
        """.format(nsteps=self.input_dict['nsteps']))
        body+= InputWriter.tail
        with open(posixpath.join(self.working_directory,'yscript.py'), 'w') as f:
            f.write(body)

    def write_yopt_cell(self):
        body = InputWriter.common.format(
            rcut=self.input_dict['rcut']/angstrom, alpha_scale=self.input_dict['alpha_scale'],
            gcut_scale=self.input_dict['gcut_scale'], smooth_ei=self.input_dict['smooth_ei'],
            h5step=1,
        )
        body += "dof = StrainCellDOF(ff, gpos_rms={gpos_rms}, dpos_rms={dpos_rms}, grvecs_rms={grvecs_rms}, drvecs_rms={drvecs_rms}, do_frozen=False)".format(
            gpos_rms=self.input_dict['gpos_rms'],dpos_rms=self.input_dict['dpos_rms'],
            grvecs_rms=self.input_dict['grvecs_rms'],drvecs_rms=self.input_dict['drvecs_rms']
        )
        body += inspect.cleandoc("""
        opt = CGOptimizer(dof, hooks=[hdf5])
        opt.run({nsteps})
        system.to_file('opt.chk')
        """.format(nsteps=self.input_dict['nsteps']))
        body+= InputWriter.tail
        with open(posixpath.join(self.working_directory,'yscript.py'), 'w') as f:
            f.write(body)

    def __write_ysp(self):
        """Deprecated due to not making a trajectory group"""
        body = InputWriter.common.format(
            rcut=self.input_dict['rcut']/angstrom, alpha_scale=self.input_dict['alpha_scale'],
            gcut_scale=self.input_dict['gcut_scale'], smooth_ei=self.input_dict['smooth_ei'],
            h5step=1,
        )
        body += inspect.cleandoc("""
        energy = ff.compute()
        system.to_hdf5(f)
        f['system/energy'] = energy
        """)
        body+= InputWriter.tail
        with open(posixpath.join(self.working_directory,'yscript.py'), 'w') as f:
            f.write(body)

    def write_ysp(self):
        body = InputWriter.common.format(
            rcut=self.input_dict['rcut']/angstrom, alpha_scale=self.input_dict['alpha_scale'],
            gcut_scale=self.input_dict['gcut_scale'], smooth_ei=self.input_dict['smooth_ei'],
            h5step=1,
        )
        body += "dof = CartesianDOF(ff, gpos_rms={gpos_rms}, dpos_rms={dpos_rms})".format(
            gpos_rms=self.input_dict['gpos_rms'],dpos_rms=self.input_dict['dpos_rms']
        )
        body += inspect.cleandoc("""
        opt = CGOptimizer(dof, hooks=[hdf5])
        opt.run(0) # just caluculate the energy
        """)
        body+= InputWriter.tail
        with open(posixpath.join(self.working_directory,'yscript.py'), 'w') as f:
            f.write(body)

    def write_yhess(self):
        body = InputWriter.common.format(
            rcut=self.input_dict['rcut']/angstrom, alpha_scale=self.input_dict['alpha_scale'],
            gcut_scale=self.input_dict['gcut_scale'], smooth_ei=self.input_dict['smooth_ei'],
            h5step=1,
        )
        body +=inspect.cleandoc("""dof = CartesianDOF(ff)

        gpos  = np.zeros((len(system.numbers), 3), float)
        vtens = np.zeros((3, 3), float)
        energy = ff.compute(gpos, vtens)
        hessian = estimate_hessian(dof, eps={hessian_eps})

        system.to_hdf5(f)
        f['system/energy'] = energy
        f['system/gpos'] = gpos
        f['system/hessian'] = hessian""".format(hessian_eps=self.input_dict['hessian_eps']))
        body+= InputWriter.tail
        with open(posixpath.join(self.working_directory,'yscript.py'), 'w') as f:
            f.write(body)

    def write_ynve(self):
        body = InputWriter.common.format(
            rcut=self.input_dict['rcut']/angstrom, alpha_scale=self.input_dict['alpha_scale'],
            gcut_scale=self.input_dict['gcut_scale'], smooth_ei=self.input_dict['smooth_ei'],
            h5step=self.input_dict['h5step'],
        )
        if self.input_dict['enhanced'] is not None:
            body += InputWriter.plumed_part
        body += inspect.cleandoc("""
        hooks.append(VerletScreenLog(step=1000))
        md = VerletIntegrator(ff, {timestep}*femtosecond, hooks=hooks)
        md.run({nsteps})
        """.format(timestep=self.input_dict['timestep']/femtosecond, nsteps=self.input_dict['nsteps']))
        body+= InputWriter.tail
        with open(posixpath.join(self.working_directory,'yscript.py'), 'w') as f:
            f.write(body)

    def write_ynvt(self):
        body = InputWriter.common.format(
            rcut=self.input_dict['rcut']/angstrom, alpha_scale=self.input_dict['alpha_scale'],
            gcut_scale=self.input_dict['gcut_scale'], smooth_ei=self.input_dict['smooth_ei'],
            h5step=self.input_dict['h5step'],
        )
        if self.input_dict['enhanced'] is not None:
            body += InputWriter.plumed_part
        body += inspect.cleandoc("""
        temp = {temp}*kelvin
        thermo = NHCThermostat(temp, timecon={timecon_thermo}*femtosecond)
        hooks.append(thermo)

        hooks.append(VerletScreenLog(step=1000))
        md = VerletIntegrator(ff, {timestep}*femtosecond, hooks=hooks)
        md.run({nsteps})
        """.format(
                temp=self.input_dict['temp']/kelvin,timestep=self.input_dict['timestep']/femtosecond,
                timecon_thermo=self.input_dict['timecon_thermo']/femtosecond, nsteps=self.input_dict['nsteps']
            ))
        body+= InputWriter.tail
        with open(posixpath.join(self.working_directory,'yscript.py'), 'w') as f:
            f.write(body)

    def write_ynpt(self):
        body = InputWriter.common.format(
            rcut=self.input_dict['rcut']/angstrom, alpha_scale=self.input_dict['alpha_scale'],
            gcut_scale=self.input_dict['gcut_scale'], smooth_ei=self.input_dict['smooth_ei'],
            h5step=self.input_dict['h5step'],
        )
        if self.input_dict['enhanced'] is not None:
            body += InputWriter.plumed_part
        body += inspect.cleandoc("""
        temp = {temp}*kelvin
        press = {press}*bar
        thermo = NHCThermostat(temp, timecon={timecon_thermo}*femtosecond)
        baro = MTKBarostat(ff, temp, press, timecon={timecon_baro}*femtosecond)
        TBC = TBCombination(thermo, baro)
        hooks.append(TBC)

        hooks.append(VerletScreenLog(step=1000))
        md = VerletIntegrator(ff, {timestep}*femtosecond, hooks=hooks)
        md.run({nsteps})
        """.format(
                temp=self.input_dict['temp']/kelvin,timestep=self.input_dict['timestep']/femtosecond,
                press=self.input_dict['press']/bar,timecon_thermo=self.input_dict['timecon_thermo']/femtosecond,
                timecon_baro=self.input_dict['timecon_baro']/femtosecond, nsteps=self.input_dict['nsteps']
            ))
        body+= InputWriter.tail
        with open(posixpath.join(self.working_directory,'yscript.py'), 'w') as f:
            f.write(body)


    def write_scan(self):
        # Write target positions for each frame to the corresponding h5 file
        scan_data = self.input_dict['scan']
        h5_path = posixpath.join(self.working_directory,'structures.h5')
        h5 = h5py.File(h5_path,mode='w')
        grp = h5.create_group("scan")
        grp['positions'] = scan_data['positions']
        if not np.isnan(scan_data['rvecs']).any():
            grp['rvecs'] = scan_data['rvecs']
        h5.close()

        # Write script file which reads and adapts h5 file with energy of each snapshot
        body = inspect.cleandoc("""#! /usr/bin/python

        from molmod.units import *
        from yaff import *
        import h5py, numpy as np

        # Define scan iterator
        class ScanIntegrator(Iterative):
            default_state = [
                AttributeStateItem('counter'),
                AttributeStateItem('epot'),
                PosStateItem(),
                VolumeStateItem(),
                CellStateItem(),
                EPotContribStateItem(),
            ]

            log_name = 'SCAN'

            def __init__(self, ff, h5_path, state=None, hooks=None, counter0=1):
                self.h5 = h5py.File(h5_path,mode='r')
                self.do_cell = 'scan/rvecs' in self.h5
                self.assign_structure(ff, self.h5, self.do_cell, 0)
                Iterative.__init__(self, ff, state, hooks, counter0)

            @staticmethod
            def assign_structure(ff,h5,do_cell,counter):
                ff.update_pos(h5['scan/positions'][counter])
                if do_cell:
                    ff.update_rvecs(h5['scan/rvecs'][counter])

            def initialize(self):
                self.epot = self.ff.compute()
                Iterative.initialize(self)

            def propagate(self):
                # counter0 should start from 1 since the counter is updated after evaluation
                self.assign_structure(self.ff, self.h5, self.do_cell, self.counter)
                self.epot = self.ff.compute()
                Iterative.propagate(self)

            def finalize(self):
                self.h5.close()


        #Setting up system and force field
        system = System.from_file('system.chk')
        ff = ForceField.generate(system, 'pars.txt', rcut={rcut}*angstrom, alpha_scale={alpha_scale}, gcut_scale={gcut_scale}, smooth_ei={smooth_ei})

        #Setting up output
        f = h5py.File('output.h5', mode='w')
        hdf5 = HDF5Writer(f, step=1)

        #Setting up simulation
        scan = ScanIntegrator(ff,'structures.h5',hooks=[hdf5])
        scan.run({steps})

            """.format(
                rcut=self.input_dict['rcut']/angstrom, alpha_scale=self.input_dict['alpha_scale'],
                gcut_scale=self.input_dict['gcut_scale'], smooth_ei=self.input_dict['smooth_ei'],
                steps=scan_data['positions'].shape[0]-1
            ))

        with open(posixpath.join(self.working_directory,'yscript.py'), 'w') as f:
            f.write(body)

    def write_plumed(self):
        #make copy of self.input_dict['enhanced'] that includes lower case definitions
        #(allowing for case insenstive definition of self.input_dict['enhanced'])
        enhanced = {}
        for key, value in self.input_dict['enhanced'].items():
            enhanced[key.lower()] = value

        #write plumed.dat file
        with open(posixpath.join(self.working_directory, 'plumed.dat'), 'w') as f:
            #set units to atomic units
            f.write('UNITS LENGTH=Bohr ENERGY=kj/mol TIME=fs \n')
            #define cvs
            for cv in enhanced['cvs']:
                f.write(cv.plumed_lines)

            #define metadynamics run
            if 'sigma' in enhanced.keys():
                if len(enhanced['sigma'])==1:
                    sigma = '%.2f' %(enhanced['sigma'])
                else:
                    assert len(enhanced['sigma'])>1
                    sigma = ','.join(['%.2f' %s for s in enhanced['sigma']])
                if len(enhanced['height'])==1:
                    height = '%.2f' %(enhanced['height']/kjmol)
                else:
                    assert len(enhanced['height'])>1
                    height = ','.join(['%.2f' %h/kjmol for h in enhanced['height']])

                f.write('metad: METAD ARG=%s SIGMA=%s HEIGHT=%s PACE=%i FILE=%s \n' %(
                    ','.join([cv.name for cv in enhanced['cvs']]), sigma, height, enhanced['pace'], enhanced['file']
                ))
                #setup printing of colvar
                f.write('PRINT ARG=%s,metad.bias FILE=%s STRIDE=%i \n' %(
                    ','.join([cv.name for cv in enhanced['cvs']]), enhanced['file_colvar'], enhanced['stride']
                ))

            # define umbrella sampling run
            if 'kappa' in enhanced.keys():
                kappa = ','.join(['%.2f' %(float(s)/kjmol) for s in enhanced['kappa']])
                loc = ','.join(['%.2f' %(float(h)) for h in enhanced['loc']])

                f.write('umbrella: RESTRAINT ARG=%s KAPPA=%s AT=%s \n' %(
                    ','.join([cv.name for cv in enhanced['cvs']]), kappa, loc
                ))
                #setup printing of colvar
                f.write('PRINT ARG=%s,umbrella.bias FILE=%s STRIDE=%i \n' %(
                    ','.join([cv.name for cv in enhanced['cvs']]), enhanced['file_colvar'], enhanced['stride']
                ))
