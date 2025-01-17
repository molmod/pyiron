import os
import unittest
from pyiron.atomistics.structure.atoms import CrystalStructure
from pyiron.base.project.generic import Project


def convergence_goal(self, **qwargs):
    import numpy as np
    eps = 0.2
    if "eps" in qwargs:
        eps = qwargs["eps"]
    erg_lst = self.get_from_childs("output/generic/energy")
    var = 1000 * np.var(erg_lst)
    # print(var / len(erg_lst))
    if var / len(erg_lst) < eps:
        return True
    job_prev = self[-1]
    job_name = self.first_child_name() + "_" + str(len(self))
    job_next = job_prev.restart(job_name=job_name)
    return job_next


class TestMurnaghan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        cls.project = Project(os.path.join(cls.file_location, 'test_murnaghan'))
        cls.basis = CrystalStructure(element="Fe", bravais_basis='fcc', lattice_constant=3.5)
        cls.project.remove_jobs(recursive=True)

    @classmethod
    def tearDownClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        project = Project(os.path.join(cls.file_location, 'test_murnaghan'))
        project.remove(enable=True, enforce=True)

    def test_run(self):
        job = self.project.create_job(self.project.job_type.AtomisticExampleJob, "job_test")
        job.structure = self.basis
        job_ser = self.project.create_job(self.project.job_type.SerialMaster, "murn_iter")
        job_ser.append(job)
        job_ser.set_goal(convergence_goal, eps=0.4)
        murn = self.project.create_job("Murnaghan", "murnaghan")
        murn.ref_job = job_ser
        murn.input["num_points"] = 3
        murn.run()
        self.assertTrue(murn.status.finished)
        murn.remove()
        job_ser.remove()


if __name__ == '__main__':
    unittest.main()
