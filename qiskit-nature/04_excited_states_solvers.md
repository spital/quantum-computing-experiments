# Excited states solvers

## Introduction

<img src="aux_files/H2_es.png" width="200">

In this tutorial we are going to discuss the excited states calculation interface of Qiskit Nature. The goal is to compute the excited states of a molecular Hamiltonian. This Hamiltonian can be electronic or vibrational. To know more about the preparation of the Hamiltonian, check out the Electronic structure and Vibrational structure tutorials. 

The first step is to define the molecular system. In the following we ask for the electronic part of a hydrogen molecule.


```python
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

Let's first start with a purely classical example: the NumPy eigensolver. This algorithm exactly diagonalizes the Hamiltonian. Although it scales badly, it can be used on small systems to check the results of the quantum algorithms. 
Here, we are only interested to look at eigenstates with a given number of particle. To compute only those states a filter function can be passed to the NumPy eigensolver. A default filter function is already implemented in Qiskit and can be used in this way:


```python
from qiskit_nature.second_q.algorithms import NumPyEigensolverFactory

numpy_solver = NumPyEigensolverFactory(use_default_filter_criterion=True)
```

The excitation energies can also be accessed with the qEOM algorithm [arXiv preprint arXiv:1910.12890 (2019)]. The EOM method finds the excitation energies (differences in energy between the ground state and all $n$th excited states) by solving the following pseudo-eigenvalue problem.

$$
\begin{pmatrix}
    \text{M} & \text{Q}\\ 
    \text{Q*} & \text{M*}
\end{pmatrix}
\begin{pmatrix}
    \text{X}_n\\ 
    \text{Y}_n
\end{pmatrix}
= E_{0n}
\begin{pmatrix}
    \text{V} & \text{W}\\ 
    -\text{W*} & -\text{V*}
\end{pmatrix}
\begin{pmatrix}
    \text{X}_n\\ 
    \text{Y}_n
\end{pmatrix}
$$

with 

$$
M_{\mu_{\alpha}\nu_{\beta}} = \langle0| [(\hat{\text{E}}_{\mu_{\alpha}}^{(\alpha)})^{\dagger},\hat{\text{H}}, \hat{\text{E}}_{\nu_{\beta}}^{(\beta)}]|0\rangle
$$
$$
Q_{\mu_{\alpha}\nu_{\beta}} = -\langle0| [(\hat{\text{E}}_{\mu_{\alpha}}^{(\alpha)})^{\dagger}, \hat{\text{H}}, (\hat{\text{E}}_{\nu_{\beta}}^{(\beta)})^{\dagger}]|0\rangle
$$
$$
V_{\mu_{\alpha}\nu_{\beta}} = \langle0| [(\hat{\text{E}}_{\mu_{\alpha}}^{(\alpha)})^{\dagger}, \hat{\text{E}}_{\nu_{\beta}}^{(\beta)}]|0\rangle
$$
$$
W_{\mu_{\alpha}\nu_{\beta}} = -\langle0| [(\hat{\text{E}}_{\mu_\alpha}^{(\alpha)})^{\dagger}, (\hat{\text{E}}_{\nu_{\beta}}^{(\beta)})^{\dagger}]|0\rangle
$$

Although the previous equation can be solved classically, each matrix element must be measured on the quantum computer with the corresponding ground state. 
To use the qEOM as a solver in Qiskit, we have to define a ground state calculation first, providing to the algorithm information on how to find the ground state. With this the qEOM solver can be initialized:


```python
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_nature.second_q.algorithms import GroundStateEigensolver, QEOM, VQEUCCFactory

# This first part sets the ground state solver
# see more about this part in the ground state calculation tutorial
quantum_instance = QuantumInstance(Aer.get_backend("aer_simulator_statevector"))
solver = VQEUCCFactory(quantum_instance=quantum_instance)
gsc = GroundStateEigensolver(qubit_converter, solver)

# The qEOM algorithm is simply instantiated with the chosen ground state solver
qeom_excited_states_calculation = QEOM(gsc, "sd")
```

## The calculation and results

The results are computed and printed


```python
from qiskit_nature.second_q.algorithms import ExcitedStatesEigensolver

numpy_excited_states_calculation = ExcitedStatesEigensolver(qubit_converter, numpy_solver)
numpy_results = numpy_excited_states_calculation.solve(es_problem)

qeom_results = qeom_excited_states_calculation.solve(es_problem)

print(numpy_results)
print("\n\n")
print(qeom_results)
```

    /usr/local/lib/python3.8/dist-packages/qiskit_nature/second_q/problems/electronic_structure_problem.py:97: ListAuxOpsDeprecationWarning: List-based `aux_operators` are deprecated as of version 0.3.0 and support for them will be removed no sooner than 3 months after the release. Instead, use dict-based `aux_operators`. You can switch to the dict-based interface immediately, by setting `qiskit_nature.settings.dict_aux_operators` to `True`.
      second_quantized_ops = self._grouped_property_transformed.second_q_ops()


    === GROUND STATE ENERGY ===
     
    * Electronic ground state energy (Hartree): -1.857275030202
      - computed part:      -1.857275030202
    ~ Nuclear repulsion energy (Hartree): 0.719968994449
    > Total ground state energy (Hartree): -1.137306035753
     
    === EXCITED STATE ENERGIES ===
     
      1: 
    * Electronic excited state energy (Hartree): -0.882722150245
    > Total excited state energy (Hartree): -0.162753155796
      2: 
    * Electronic excited state energy (Hartree): -0.224911252831
    > Total excited state energy (Hartree): 0.495057741618
     
    === MEASURED OBSERVABLES ===
     
      0:  # Particles: 2.000 S: 0.000 S^2: 0.000 M: 0.000
      1:  # Particles: 2.000 S: 0.000 S^2: 0.000 M: 0.000
      2:  # Particles: 2.000 S: 0.000 S^2: 0.000 M: 0.000
     
    === DIPOLE MOMENTS ===
     
    ~ Nuclear dipole moment (a.u.): [0.0  0.0  1.3889487]
     
      0: 
      * Electronic dipole moment (a.u.): [0.0  0.0  1.3889487]
        - computed part:      [0.0  0.0  1.3889487]
      > Dipole moment (a.u.): [0.0  0.0  0.0]  Total: 0.0
                     (debye): [0.0  0.0  0.0]  Total: 0.0
     
      1: 
      * Electronic dipole moment (a.u.): [0.0  0.0  1.3889487]
        - computed part:      [0.0  0.0  1.3889487]
      > Dipole moment (a.u.): [0.0  0.0  0.0]  Total: 0.0
                     (debye): [0.0  0.0  0.0]  Total: 0.0
     
      2: 
      * Electronic dipole moment (a.u.): [0.0  0.0  1.3889487]
        - computed part:      [0.0  0.0  1.3889487]
      > Dipole moment (a.u.): [0.0  0.0  0.0]  Total: 0.0
                     (debye): [0.0  0.0  0.0]  Total: 0.0
     
    
    
    
    === GROUND STATE ENERGY ===
     
    * Electronic ground state energy (Hartree): -1.857275030145
      - computed part:      -1.857275030145
    ~ Nuclear repulsion energy (Hartree): 0.719968994449
    > Total ground state energy (Hartree): -1.137306035696
     
    === EXCITED STATE ENERGIES ===
     
      1: 
    * Electronic excited state energy (Hartree): -1.244586758556
    > Total excited state energy (Hartree): -0.524617764107
      2: 
    * Electronic excited state energy (Hartree): -0.882724358957
    > Total excited state energy (Hartree): -0.162755364508
      3: 
    * Electronic excited state energy (Hartree): -0.224913461551
    > Total excited state energy (Hartree): 0.495055532898
     
    === MEASURED OBSERVABLES ===
     
      0:  # Particles: 2.000 S: 0.000 S^2: 0.000 M: 0.000
     
    === DIPOLE MOMENTS ===
     
    ~ Nuclear dipole moment (a.u.): [0.0  0.0  1.3889487]
     
      0: 
      * Electronic dipole moment (a.u.): [0.0  0.0  1.38894842]
        - computed part:      [0.0  0.0  1.38894842]
      > Dipole moment (a.u.): [0.0  0.0  0.00000028]  Total: 0.00000028
                     (debye): [0.0  0.0  0.00000072]  Total: 0.00000072
     


One can see from these results that one state is missing from the NumPy results. The reason for this is because the spin is also used as a filter and only singlet states are shown. 
In the following we use a custom filter function to check consistently our results and only filter out states with incorrect number of particle (in this case the number of particle is 2).


```python
import numpy as np


def filter_criterion(eigenstate, eigenvalue, aux_values):
    return np.isclose(aux_values[0][0], 2.0)


new_numpy_solver = NumPyEigensolverFactory(filter_criterion=filter_criterion)
new_numpy_excited_states_calculation = ExcitedStatesEigensolver(qubit_converter, new_numpy_solver)
new_numpy_results = new_numpy_excited_states_calculation.solve(es_problem)

print(new_numpy_results)
```

    === GROUND STATE ENERGY ===
     
    * Electronic ground state energy (Hartree): -1.857275030202
      - computed part:      -1.857275030202
    ~ Nuclear repulsion energy (Hartree): 0.719968994449
    > Total ground state energy (Hartree): -1.137306035753
     
    === EXCITED STATE ENERGIES ===
     
      1: 
    * Electronic excited state energy (Hartree): -1.244584549813
    > Total excited state energy (Hartree): -0.524615555364
      2: 
    * Electronic excited state energy (Hartree): -1.244584549813
    > Total excited state energy (Hartree): -0.524615555364
      3: 
    * Electronic excited state energy (Hartree): -1.244584549813
    > Total excited state energy (Hartree): -0.524615555364
      4: 
    * Electronic excited state energy (Hartree): -0.882722150245
    > Total excited state energy (Hartree): -0.162753155796
      5: 
    * Electronic excited state energy (Hartree): -0.224911252831
    > Total excited state energy (Hartree): 0.495057741618
     
    === MEASURED OBSERVABLES ===
     
      0:  # Particles: 2.000 S: 0.000 S^2: 0.000 M: 0.000
      1:  # Particles: 2.000 S: 1.000 S^2: 2.000 M: 0.000
      2:  # Particles: 2.000 S: 1.000 S^2: 2.000 M: -1.000
      3:  # Particles: 2.000 S: 1.000 S^2: 2.000 M: 1.000
      4:  # Particles: 2.000 S: 0.000 S^2: 0.000 M: 0.000
      5:  # Particles: 2.000 S: 0.000 S^2: 0.000 M: 0.000
     
    === DIPOLE MOMENTS ===
     
    ~ Nuclear dipole moment (a.u.): [0.0  0.0  1.3889487]
     
      0: 
      * Electronic dipole moment (a.u.): [0.0  0.0  1.3889487]
        - computed part:      [0.0  0.0  1.3889487]
      > Dipole moment (a.u.): [0.0  0.0  0.0]  Total: 0.0
                     (debye): [0.0  0.0  0.0]  Total: 0.0
     
      1: 
      * Electronic dipole moment (a.u.): [0.0  0.0  1.3889487]
        - computed part:      [0.0  0.0  1.3889487]
      > Dipole moment (a.u.): [0.0  0.0  0.0]  Total: 0.0
                     (debye): [0.0  0.0  0.0]  Total: 0.0
     
      2: 
      * Electronic dipole moment (a.u.): [0.0  0.0  1.3889487]
        - computed part:      [0.0  0.0  1.3889487]
      > Dipole moment (a.u.): [0.0  0.0  0.0]  Total: 0.0
                     (debye): [0.0  0.0  0.0]  Total: 0.0
     
      3: 
      * Electronic dipole moment (a.u.): [0.0  0.0  1.3889487]
        - computed part:      [0.0  0.0  1.3889487]
      > Dipole moment (a.u.): [0.0  0.0  0.0]  Total: 0.0
                     (debye): [0.0  0.0  0.0]  Total: 0.0
     
      4: 
      * Electronic dipole moment (a.u.): [0.0  0.0  1.3889487]
        - computed part:      [0.0  0.0  1.3889487]
      > Dipole moment (a.u.): [0.0  0.0  0.0]  Total: 0.0
                     (debye): [0.0  0.0  0.0]  Total: 0.0
     
      5: 
      * Electronic dipole moment (a.u.): [0.0  0.0  1.3889487]
        - computed part:      [0.0  0.0  1.3889487]
      > Dipole moment (a.u.): [0.0  0.0  0.0]  Total: 0.0
                     (debye): [0.0  0.0  0.0]  Total: 0.0
     



```python
import qiskit.tools.jupyter

%qiskit_version_table
%qiskit_copyright
```


<h3>Version Information</h3><table><tr><th>Qiskit Software</th><th>Version</th></tr><tr><td><code>qiskit-terra</code></td><td>0.21.1</td></tr><tr><td><code>qiskit-aer</code></td><td>0.10.4</td></tr><tr><td><code>qiskit-ibmq-provider</code></td><td>0.19.2</td></tr><tr><td><code>qiskit-nature</code></td><td>0.5.0</td></tr><tr><td><code>qiskit-finance</code></td><td>0.4.0</td></tr><tr><td><code>qiskit-optimization</code></td><td>0.4.0</td></tr><tr><td><code>qiskit-machine-learning</code></td><td>0.5.0</td></tr><tr><th>System information</th></tr><tr><td>Python version</td><td>3.8.10</td></tr><tr><td>Python compiler</td><td>GCC 9.4.0</td></tr><tr><td>Python build</td><td>default, Jun 22 2022 20:18:18</td></tr><tr><td>OS</td><td>Linux</td></tr><tr><td>CPUs</td><td>12</td></tr><tr><td>Memory (Gb)</td><td>31.267112731933594</td></tr><tr><td colspan='2'>Sat Jul 30 15:18:58 2022 CEST</td></tr></table>



<div style='width: 100%; background-color:#d5d9e0;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'><h3>This code is a part of Qiskit</h3><p>&copy; Copyright IBM 2017, 2022.</p><p>This code is licensed under the Apache License, Version 2.0. You may<br>obtain a copy of this license in the LICENSE.txt file in the root directory<br> of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.<p>Any modifications or derivative works of this code must retain this<br>copyright notice, and modified files need to carry a notice indicating<br>that they have been altered from the originals.</p></div>

