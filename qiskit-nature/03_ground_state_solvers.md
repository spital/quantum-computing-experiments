# Ground state solvers

## Introduction

<img src="aux_files/H2_gs.png" width="200">

In this tutorial we are going to discuss the ground state calculation interface of Qiskit Nature. The goal is to compute the ground state of a molecular Hamiltonian. This Hamiltonian can be electronic or vibrational. To know more about the preparation of the Hamiltonian, check out the Electronic structure and Vibrational structure tutorials. 

The first step is to define the molecular system. In the following we ask for the electronic part of a hydrogen molecule.


```python
from qiskit import Aer
from qiskit_nature.second_q.drivers import UnitsType, Molecule
from qiskit_nature.second_q.drivers import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.mappers import JordanWignerMapper

molecule = Molecule(
    geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.735]]], charge=0, multiplicity=1
)
driver = ElectronicStructureMoleculeDriver(
    molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF
)

es_problem = ElectronicStructureProblem(driver)
qubit_converter = QubitConverter(JordanWignerMapper())
```

## The Solver

Then we need to define a solver. The solver is the algorithm through which the ground state is computed. 

Let's first start with a purely classical example: the NumPy minimum eigensolver. This algorithm exactly diagonalizes the Hamiltonian. Although it scales badly, it can be used on small systems to check the results of the quantum algorithms. 


```python
from qiskit.algorithms import NumPyMinimumEigensolver

numpy_solver = NumPyMinimumEigensolver()
```

To find the ground state we coul also use the Variational Quantum Eigensolver (VQE) algorithm. The VQE algorithms works by exchanging information between a classical and a quantum computer as depicted in the following figure.

<img src="aux_files/vqe.png" width="600">

Let's initialize a VQE solver.


```python
from qiskit.providers.aer import StatevectorSimulator
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_nature.second_q.algorithms import VQEUCCFactory

quantum_instance = QuantumInstance(backend=Aer.get_backend("aer_simulator_statevector"))
vqe_solver = VQEUCCFactory(quantum_instance=quantum_instance)
```

To define the VQE solver one needs two essential elements:

1. A variational form: here we use the Unitary Coupled Cluster (UCC) ansatz (see for instance [Physical Review A 98.2 (2018): 022322]). Since it is a chemistry standard, a factory is already available allowing a fast initialization of a VQE with UCC. The default is to use all single and double excitations. However, the excitation type (S, D, SD) as well as other parameters can be selected.
2. An initial state: the initial state of the qubits. In the factory used above, the qubits are initialized in the Hartree-Fock (see the electronic structure tutorial) initial state (the qubits corresponding to occupied MOs are $|1\rangle$ and those corresponding to virtual MOs are $|0\rangle$.
3. The backend: this is the quantum machine on which the right part of the figure above will be performed. Here we ask for the perfect quantum emulator (```aer_simulator_statevector```). 

One could also use any available ansatz / initial state or even define one's own. For instance,


```python
from qiskit.algorithms import VQE
from qiskit.circuit.library import TwoLocal

tl_circuit = TwoLocal(
    rotation_blocks=["h", "rx"],
    entanglement_blocks="cz",
    entanglement="full",
    reps=2,
    parameter_prefix="y",
)

another_solver = VQE(
    ansatz=tl_circuit,
    quantum_instance=QuantumInstance(Aer.get_backend("aer_simulator_statevector")),
)
```

## The calculation and results

We are now ready to run the calculation.


```python
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

calc = GroundStateEigensolver(qubit_converter, vqe_solver)
res = calc.solve(es_problem)

print(res)
```

    /usr/local/lib/python3.8/dist-packages/qiskit_nature/second_q/problems/electronic_structure_problem.py:97: ListAuxOpsDeprecationWarning: List-based `aux_operators` are deprecated as of version 0.3.0 and support for them will be removed no sooner than 3 months after the release. Instead, use dict-based `aux_operators`. You can switch to the dict-based interface immediately, by setting `qiskit_nature.settings.dict_aux_operators` to `True`.
      second_quantized_ops = self._grouped_property_transformed.second_q_ops()


    === GROUND STATE ENERGY ===
     
    * Electronic ground state energy (Hartree): -1.857275030145
      - computed part:      -1.857275030145
    ~ Nuclear repulsion energy (Hartree): 0.719968994449
    > Total ground state energy (Hartree): -1.137306035696
     
    === MEASURED OBSERVABLES ===
     
      0:  # Particles: 2.000 S: 0.000 S^2: 0.000 M: 0.000
     
    === DIPOLE MOMENTS ===
     
    ~ Nuclear dipole moment (a.u.): [0.0  0.0  1.3889487]
     
      0: 
      * Electronic dipole moment (a.u.): [0.0  0.0  1.38894842]
        - computed part:      [0.0  0.0  1.38894842]
      > Dipole moment (a.u.): [0.0  0.0  0.00000028]  Total: 0.00000028
                     (debye): [0.0  0.0  0.00000072]  Total: 0.00000072
     


We can compare the VQE results to the NumPy exact solver and see that they match. 


```python
calc = GroundStateEigensolver(qubit_converter, numpy_solver)
res = calc.solve(es_problem)
print(res)
```

    === GROUND STATE ENERGY ===
     
    * Electronic ground state energy (Hartree): -1.857275030202
      - computed part:      -1.857275030202
    ~ Nuclear repulsion energy (Hartree): 0.719968994449
    > Total ground state energy (Hartree): -1.137306035753
     
    === MEASURED OBSERVABLES ===
     
      0:  # Particles: 2.000 S: 0.000 S^2: 0.000 M: 0.000
     
    === DIPOLE MOMENTS ===
     
    ~ Nuclear dipole moment (a.u.): [0.0  0.0  1.3889487]
     
      0: 
      * Electronic dipole moment (a.u.): [0.0  0.0  1.3889487]
        - computed part:      [0.0  0.0  1.3889487]
      > Dipole moment (a.u.): [0.0  0.0  0.0]  Total: 0.0
                     (debye): [0.0  0.0  0.0]  Total: 0.0
     


## Using a filter function

Sometimes the true ground state of the Hamiltonian is not of interest because it lies in a different symmetry sector of the Hilbert space. In this case the NumPy eigensolver can take a filter function to return only the eigenstates with for example the correct number of particles. This is of particular importance in the case of vibrational structure calculations where the true ground state of the Hamiltonian is the vacuum state. A default filter function to check the number of particles is implemented in the different transformations and can be used as follows:


```python
from qiskit_nature.second_q.drivers import GaussianForcesDriver
from qiskit_nature.second_q.algorithms import NumPyMinimumEigensolverFactory
from qiskit_nature.second_q.problems import VibrationalStructureProblem
from qiskit_nature.second_q.mappers import DirectMapper

driver = GaussianForcesDriver(logfile="aux_files/CO2_freq_B3LYP_631g.log")

vib_problem = VibrationalStructureProblem(driver, num_modals=2, truncation_order=2)

qubit_covnerter = QubitConverter(DirectMapper())

solver_without_filter = NumPyMinimumEigensolverFactory(use_default_filter_criterion=False)
solver_with_filter = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)

gsc_wo = GroundStateEigensolver(qubit_converter, solver_without_filter)
result_wo = gsc_wo.solve(vib_problem)

gsc_w = GroundStateEigensolver(qubit_converter, solver_with_filter)
result_w = gsc_w.solve(vib_problem)

print(result_wo)
print("\n\n")
print(result_w)
```

    === GROUND STATE ===
     
    * Vibrational ground state energy (cm^-1): (-4e-12+0j)
    The number of occupied modals for each mode is: 
    - Mode 0: 0.0
    - Mode 1: 0.0
    - Mode 2: 0.0
    - Mode 3: 0.0
    
    
    
    === GROUND STATE ===
     
    * Vibrational ground state energy (cm^-1): (2435.501906392164+0j)
    The number of occupied modals for each mode is: 
    - Mode 0: 1.0
    - Mode 1: 1.0
    - Mode 2: 1.0
    - Mode 3: 1.0



```python
import qiskit.tools.jupyter

%qiskit_version_table
%qiskit_copyright
```


<h3>Version Information</h3><table><tr><th>Qiskit Software</th><th>Version</th></tr><tr><td><code>qiskit-terra</code></td><td>0.21.1</td></tr><tr><td><code>qiskit-aer</code></td><td>0.10.4</td></tr><tr><td><code>qiskit-ibmq-provider</code></td><td>0.19.2</td></tr><tr><td><code>qiskit-nature</code></td><td>0.5.0</td></tr><tr><td><code>qiskit-finance</code></td><td>0.4.0</td></tr><tr><td><code>qiskit-optimization</code></td><td>0.4.0</td></tr><tr><td><code>qiskit-machine-learning</code></td><td>0.5.0</td></tr><tr><th>System information</th></tr><tr><td>Python version</td><td>3.8.10</td></tr><tr><td>Python compiler</td><td>GCC 9.4.0</td></tr><tr><td>Python build</td><td>default, Jun 22 2022 20:18:18</td></tr><tr><td>OS</td><td>Linux</td></tr><tr><td>CPUs</td><td>12</td></tr><tr><td>Memory (Gb)</td><td>31.267112731933594</td></tr><tr><td colspan='2'>Sat Jul 30 15:18:46 2022 CEST</td></tr></table>



<div style='width: 100%; background-color:#d5d9e0;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'><h3>This code is a part of Qiskit</h3><p>&copy; Copyright IBM 2017, 2022.</p><p>This code is licensed under the Apache License, Version 2.0. You may<br>obtain a copy of this license in the LICENSE.txt file in the root directory<br> of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.<p>Any modifications or derivative works of this code must retain this<br>copyright notice, and modified files need to carry a notice indicating<br>that they have been altered from the originals.</p></div>

