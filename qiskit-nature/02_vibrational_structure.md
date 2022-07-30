# Vibrational structure

## Introduction 

The molecular Hamiltonian is 

$$
\mathcal{H} = - \sum_I \frac{\nabla_{R_I}^2}{M_I} - \sum_i \frac{\nabla_{r_i}^2}{m_e} - \sum_I\sum_i  \frac{Z_I e^2}{|R_I-r_i|} + \sum_i \sum_{j>i} \frac{e^2}{|r_i-r_j|} + \sum_I\sum_{J>I} \frac{Z_I Z_J e^2}{|R_I-R_J|}
$$

Because the nuclei are much heavier than the electrons they do not move on the same time scale and therefore, the behavior of nuclei and electrons can be decoupled. This is the Born-Oppenheimer approximation.

Within the Born-Oppenheimer approximation, a molecular wave function is factorized as a product of an electronic part, which is the solution of the electronic Schroedinger equation, and a vibro-rotational one, which is the solution of the nuclear Schroedinger equation in the potential energy surface (PES) generated by sampling the eigenvalues of the electronic Schroedinger equation for different geometries.

The nuclear Schroedinger equation is usually solved in two steps, in analogy with its electronic counterpart. 
A single-particle basis (the basis functions are called, in this case, modals) is obtained either by the harmonic approximation applied to the PES or from a vibrational self-consistent field (VSCF) calculation. 
Vibrational anharmonic correlations are added a-posteriori with perturbative or variational approaches.
The latter include Vibrational Configuration Interaction (VCI) and Vibrational Coupled Cluster (VCC) for highly-accurate anharmonic energies. 
The main advantage of VCI and VCC over alternative approaches (such as perturbation theories) is that their accuracy can be systematically improved towards the complete basis set limit for a given PES. 
However, their applicability is limited to small molecules with up to about 10 atoms due to their unfavorable scaling with system size.

To tackle the scaling problem we would like to use quantum algorithms.

The nuclear Schroedinger equation is
$$
\mathcal{H}_{\text{vib}} |\Psi_{n}\rangle = E_{n} |\Psi_{n}\rangle
$$

The so-called Watson Hamiltonian (neglecting vibro-rotational coupling terms) is
$$
  \mathcal{H}_\text{vib}(Q_1, \ldots, Q_L) 
    = - \frac{1}{2} \sum_{l=1}^{L} \frac{\partial^2}{\partial Q_l^2} + V(Q_1, \ldots, Q_L)
$$
where $Q_l$ are the harmonic mass-weighted normal coordinates.

$\mathcal{H}_\text{vib}$ must be mapped to an operator that acts on the states of a given set of $N_q$ qubits in order to calculate its eigenfunctions on quantum hardware.
In electronic structure calculations, the mapping is achieved by expressing the non-relativistic electronic Hamiltonian in second quantization, \textit{i.e.} by projecting it onto the complete set of antisymmetrized occupation number vectors (ONV) generated by a given (finite) set of orbitals.
To encode the vibrational Hamiltonian in an analogous second quantization operators, we expand the potential $V(Q_1, \ldots, Q_L)$ with the $n$-body expansion as follows:

$$
  V(Q_1, \ldots, Q_L) = V_0 + \sum_{l=1}^L V^{[l]}(Q_l) 
    + \sum_{l<m}^L V^{[l,m]}(Q_l, Q_m) + \sum_{l<m<n}^L V^{[l,m,n]}(Q_l, Q_m, Q_n) + \ldots
$$

where $V_0$ is the electronic energy of the reference geometry, the one-mode term $V^{[l]}(Q_l)$ represents the variation of the PES upon change of the $l$-th normal coordinate from the equilibrium position.
Similarly, the two-body potential $V^{[l,m]}(Q_l, Q_m)$ represents the change in the exact PES upon a simultaneous displacement along the $l$-th and $m$-th coordinates. 
Often, including terms up to three-body in the $L$-body expansion is sufficient to obtain an accuracy of about 1~cm$^{-1}$. We highlight that the many-body expansion of the potential operator defining the Watson Hamiltonian contains arbitrarily high coupling terms. This is a crucial difference compared to the non-relativistic electronic-structure Hamiltonian that contains only pairwise interactions.

A flexible second quantization form of the Watson Hamiltonian is obtained within the so-called n-mode representation. Let us assume that each mode $l$ is described by a $N_l$-dimensional basis set $S_l$ defined as follows:

$$
  \mathcal{S}_l = \{ \phi_1^{(l)} (Q_l) , \ldots , \phi_{N_l}^{(l)} (Q_l) \} \, .
$$

The $n$-mode wave function can be expanded in the product basis $\mathcal{S} = \otimes_{i=1}^L \mathcal{S}_i$ as the following CI-like expansion:

$$
  |\Psi\rangle = \sum_{k_1=1}^{N_1} \cdots \sum_{k_L=1}^{N_L} C_{k_1,\ldots,k_L} 
    \phi_{k_1}^{(1)}(Q_1) \cdots \phi_{k_L}^{(L)}(Q_L) \, ,
$$

The many-body basis function $\phi_{k_1}^{(1)}(Q_1) \cdots \phi_{k_L}^{(L)}(Q_L)$ are encoded within the so-called $n$-mode second quantization as occupation-number vectors (ONVs) as follows:

$$
  \phi_{k_1}(Q_1) \cdots \phi_{k_L}(Q_L)
                      \equiv  |0_1 \cdots 1_{k_1} \cdots 0_{N_1},
                                   0_1 \cdots 1_{k_2} \cdots 0_{N_2}, 
                                   \cdots , 
                                   0_1 \cdots 1_{k_L} \cdots 0_{N_L}\rangle \, .
$$

The ONV defined above is, therefore, the product of $L$ mode-specific ONVs, each one describing an individual mode. Since each mode is described by one and only one basis function, the occupation of each mode-specific ONV is one.
From a theoretical perspective, each mode can be interpreted as a distinguishable quasi-particle (defined as phonons in solid-state physics). Distinguishability arises from the fact that the PES is not invariant by permutation of two modes, also in this case unlike the Coulomb interaction between two equal particles. From this perspective, a molecule can be interpreted as a collection of $L$ indistinguishable particles that interact through the PES operator.

Based on this second-quantization representation we introduce a pair of creation and annihilation operators per mode $l$ \textit{and} per basis function $k_l$ defined as:

$$
  \begin{aligned}
    a_{k_l}^\dagger |\cdots, 0_1 \cdots 0_{k_l} \cdots 0_{N_l}, \cdots\rangle 
      &=  | \cdots, 0_1 \cdots 1_{k_l} \cdots 0_{N_l}, \cdots\rangle \\
    a_{k_l}^\dagger | \cdots, 0_1 \cdots 1_{k_l} \cdots 0_{N_l}, \cdots\rangle &=  0 \\
    a_{k_l} | \cdots, 0_1 \cdots 1_{k_l} \cdots 0_{N_l}, \cdots\rangle
     &= | \cdots, 0_1 \cdots 0_{k_l} \cdots 0_{N_l}, \cdots\rangle \\
    a_{k_l} | \cdots, 0_1 \cdots 0_{k_l} \cdots 0_{N_l}, \cdots\rangle &=  0 \\
  \end{aligned}
$$

with

$$
  \begin{aligned}
    \left[ a_{k_l}^\dagger, a_{h_m}^\dagger \right] &= 0 \\
    \left[ a_{k_l}, a_{h_m} \right] &= 0 \\
    \left[ a_{k_l}^\dagger, a_{h_m} \right] &= \delta_{l,m} \, , \delta_{k_l,h_m}
  \end{aligned}
$$

The second quantization form is obtained by expressing the potential as 

$$
 \begin{aligned}
  \mathcal{H}_\text{vib}^{SQ} =& \sum_{l=1}^L 
    \sum_{k_l,h_l}^{N_l} \langle \phi_{k_l} | T(Q_l) + V^{[l]}(Q_l) | \phi_{h_l} \rangle a_{k_l}^+ a_{h_l} \\
 +& \sum_{l<m}^L \sum_{k_l,h_l}^{N_l} \sum_{k_m,h_m}^{N_m}
    \langle \phi_{k_l} \phi_{k_m} | V^{[l,m]}(Q_l, Q_m) | \phi_{h_l} \phi_{h_m} \rangle 
    a_{k_l}^+ a_{k_m}^+ a_{h_l} a_{h_m} + \cdots
 \end{aligned}
$$

We highlight here the difference between the operators defined here above and the electronic structure one. First, as we already mentioned, the potential contains (in principle) three- and higher-body coupling terms that lead to strings with six (or more) second-quantization operators. 
Moreover, the Hamiltonian conserves the number of particles for each mode, as can be seen from the fact that the number of creation and annihilation operators for a given mode is the same in each term. Nevertheless, different modes are coupled by two- (and higher) body terms containing SQ operators belonging to different modes $l$ and $m$.

Reference: Ollitrault, Pauline J., et al., arXiv:2003.12578 (2020).

Compute the electronic potential

Solving the ESE for different nuclear configurations to obtain the PES function $V(Q_1, \ldots, Q_L)$. So far Qiskit gives the possibility to approximate the PES with a quartic force field. 
$$
V(Q_1, \ldots, Q_L) = \frac{1}{2}  \sum_{ij} k_{ij} Q_i Q_j
                  + \frac{1}{6}  \sum_{ijk} k_{ijk} Q_i Q_j Q_k
                  + \frac{1}{16} \sum_{ijkl} k_{ijkl} Q_i Q_j Q_k Q_l
$$
The advantage of such form for the PES is that the anharmonic force fields ($k_{ij}$, $k_{ijk}$, $k_{ijkl}$) can be calculated by finite-difference approaches. For methods for which the nuclear energy Hessian can be calculated analytically with response theory-based methods (such as HF and DFT), the quartic force field can be calculated by semi-numerical differentiation as:
$$
k_{ijk} = \frac{H_{ij}(+\delta Q_k) - H_{ij}(-\delta Q_k)}{2\delta Q_k}
$$
and
$$
k_{ijkl} = \frac{H_{ij}(+\delta Q_k+\delta Q_l) - H_{ij}(+\delta Q_k-\delta Q_l)
                    -H_{ij}(-\delta Q_k+\delta Q_l) + H_{ij}(-\delta Q_k+\delta Q_l)}
                    {4\delta Q_k \delta Q_l}
$$
Such numerical procedure is implemented, for instance, in the Gaussian suite of programs.

In practice this can be done with Qiskit using the GaussianForceDriver. 
       


```python
from qiskit_nature.second_q.drivers import GaussianForcesDriver

# if you ran Gaussian elsewhere and already have the output file
driver = GaussianForcesDriver(logfile="aux_files/CO2_freq_B3LYP_631g.log")

# if you want to run the Gaussian job from Qiskit
# driver = GaussianForcesDriver(
#                 ['#p B3LYP/6-31g Freq=(Anharm) Int=Ultrafine SCF=VeryTight',
#                  '',
#                  'CO2 geometry optimization B3LYP/6-31g',
#                  '',
#                  '0 1',
#                  'C  -0.848629  2.067624  0.160992',
#                  'O   0.098816  2.655801 -0.159738',
#                  'O  -1.796073  1.479446  0.481721',
#                  '',
#                  ''
```

## Map to a qubit Hamiltonian

Now that we have an approximation for the potential, we need to write the Hamiltonian in second quantization. To this end we need to select a modal basis to calculate the one-body integrals $\langle\phi_{k_i}| V(Q_i) | \phi_{h_i} \rangle$, two-body integrals $\langle\phi_{k_i} \phi_{k_j}| V(Q_i,Q_j) | \phi_{h_i} \phi_{h_j} \rangle$... 

In the simplest case, the $\phi$ functions are the harmonic-oscillator eigenfunctions for each mode. The main advantage of this choice is that the integrals of a PES expressed as a Taylor expansion are easy to calculate with such basis. A routine for computing these integrals is implemented in Qiskit. 

The bosonic operator, $\mathcal{H}_\text{vib}^{SQ}$, is then created and must be mapped to a qubit operator. The direct mapping introduced in the first section of this tutorial can be used is Qiskit as follows:


```python
from qiskit_nature.second_q.problems import VibrationalStructureProblem
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.mappers import DirectMapper

vibrational_problem = VibrationalStructureProblem(driver, num_modals=2, truncation_order=2)
second_q_ops = vibrational_problem.second_q_ops()
```

    /usr/local/lib/python3.8/dist-packages/qiskit_nature/second_q/problems/vibrational_structure_problem.py:94: ListAuxOpsDeprecationWarning: List-based `aux_operators` are deprecated as of version 0.3.0 and support for them will be removed no sooner than 3 months after the release. Instead, use dict-based `aux_operators`. You can switch to the dict-based interface immediately, by setting `qiskit_nature.settings.dict_aux_operators` to `True`.
      second_quantized_ops = self._grouped_property_transformed.second_q_ops()


The Vibrational operator for the problem now reads as


```python
print(second_q_ops[0])
```

      NIIIIIII * (1268.0676746875001+0j)
    + INIIIIII * (3813.8767834375008+0j)
    + IINIIIII * (705.8633818750001+0j)
    + II+-IIII * (-46.025705898886045+0j)
    + II-+IIII * (-46.025705898886045+0j)
    + IIINIIII * (2120.1145593750007+0j)
    + IIIINIII * (238.31540750000005+0j)
    + IIIIINII * (728.9613775000003+0j)
    + IIIIIINI * (238.31540750000005+0j)
    + IIIIIIIN * (728.9613775000003+0j)
    + NINIIIII * (4.942542500000002+0j)
    + NI+-IIII * (-88.20174216876333+0j)
    + NI-+IIII * (-88.20174216876333+0j)
    + NIINIIII * (14.827627500000007+0j)
    + INNIIIII * (14.827627500000007+0j)
    + IN+-IIII * (-264.60522650629+0j)
    + IN-+IIII * (-264.60522650629+0j)
    + ININIIII * (44.482882500000024+0j)
    + NIIINIII * (-10.205891250000004+0j)
    + INIINIII * (-30.617673750000016+0j)
    + IININIII * (-4.194299375000002+0j)
    + II+-NIII * (42.67527310283147+0j)
    + II-+NIII * (42.67527310283147+0j)
    + IIINNIII * (-12.582898125000007+0j)
    + NIIIINII * (-30.61767375000002+0j)
    + INIIINII * (-91.85302125000007+0j)
    + IINIINII * (-12.582898125000007+0j)
    + II+-INII * (128.02581930849442+0j)
    + II-+INII * (128.02581930849442+0j)
    + IIININII * (-37.74869437500002+0j)
    + NIIIIINI * (-10.205891250000004+0j)
    + INIIIINI * (-30.617673750000016+0j)
    + IINIIINI * (-4.194299375000002+0j)
    + II+-IINI * (42.67527310283147+0j)
    + II-+IINI * (42.67527310283147+0j)
    + IIINIINI * (-12.582898125000007+0j)
    + IIIININI * (7.0983500000000035+0j)
    + IIIIINNI * (21.29505000000001+0j)
    + NIIIIIIN * (-30.61767375000002+0j)
    + INIIIIIN * (-91.85302125000007+0j)
    + IINIIIIN * (-12.582898125000007+0j)
    + II+-IIIN * (128.02581930849442+0j)
    + II-+IIIN * (128.02581930849442+0j)
    + IIINIIIN * (-37.74869437500002+0j)
    + IIIINIIN * (21.29505000000001+0j)
    + IIIIININ * (63.88515000000004+0j)


In the previous cell we defined a bosonic transformation to express the Hamiltonian in the harmonic modal basis, with 2 modals per mode with the potential truncated at order 2 and the 'direct' boson to qubit mapping. 
The calculation is then ran as:


```python
qubit_converter = QubitConverter(mapper=DirectMapper())
qubit_op = qubit_converter.convert(second_q_ops[0])

print(qubit_op)
```

    4854.2000296874985 * IIIIIIII
    - 618.5645973437499 * IIIIIIIZ
    - 1860.5306717187505 * IIIIIIZI
    - 349.4856346875001 * IIIIIZII
    - 25.86404891254341 * IIIIXXII
    - 25.86404891254341 * IIIIYYII
    - 1049.7191109375005 * IIIIZIII
    - 111.85586312500003 * IIIZIIII
    - 342.57516687500015 * IIZIIIII
    - 111.85586312500003 * IZIIIIII
    - 342.57516687500015 * ZIIIIIII
    + 1.2356356250000005 * IIIIIZIZ
    + 22.050435542190833 * IIIIXXIZ
    + 22.050435542190833 * IIIIYYIZ
    + 3.706906875000002 * IIIIZIIZ
    + 3.706906875000002 * IIIIIZZI
    + 66.1513066265725 * IIIIXXZI
    + 66.1513066265725 * IIIIYYZI
    + 11.120720625000006 * IIIIZIZI
    - 2.551472812500001 * IIIZIIIZ
    - 7.654418437500004 * IIIZIIZI
    - 1.0485748437500004 * IIIZIZII
    - 10.668818275707867 * IIIZXXII
    - 10.668818275707867 * IIIZYYII
    - 3.1457245312500017 * IIIZZIII
    - 7.654418437500005 * IIZIIIIZ
    - 22.963255312500017 * IIZIIIZI
    - 3.1457245312500017 * IIZIIZII
    - 32.006454827123605 * IIZIXXII
    - 32.006454827123605 * IIZIYYII
    - 9.437173593750005 * IIZIZIII
    - 2.551472812500001 * IZIIIIIZ
    - 7.654418437500004 * IZIIIIZI
    - 1.0485748437500004 * IZIIIZII
    - 10.668818275707867 * IZIIXXII
    - 10.668818275707867 * IZIIYYII
    - 3.1457245312500017 * IZIIZIII
    + 1.7745875000000009 * IZIZIIII
    + 5.323762500000003 * IZZIIIII
    - 7.654418437500005 * ZIIIIIIZ
    - 22.963255312500017 * ZIIIIIZI
    - 3.1457245312500017 * ZIIIIZII
    - 32.006454827123605 * ZIIIXXII
    - 32.006454827123605 * ZIIIYYII
    - 9.437173593750005 * ZIIIZIII
    + 5.323762500000003 * ZIIZIIII
    + 15.97128750000001 * ZIZIIIII


To have a different number of modals per mode:


```python
vibrational_problem = VibrationalStructureProblem(driver, num_modals=3, truncation_order=2)
second_q_ops = vibrational_problem.second_q_ops()

qubit_converter = QubitConverter(mapper=DirectMapper())

qubit_op = qubit_converter.convert(second_q_ops[0])

print(qubit_op)
```

    10788.719982656254 * IIIIIIIIIIII
    - 599.2280473437501 * IIIIIIIIIIIZ
    - 42.382439413480434 * IIIIIIIIIXIX
    - 42.382439413480434 * IIIIIIIIIYIY
    - 1802.5210217187505 * IIIIIIIIIIZI
    - 3015.48775546875 * IIIIIIIIIZII
    - 345.17806437500013 * IIIIIIIIZIII
    - 29.428043866418914 * IIIIIIIXXIII
    - 29.428043866418914 * IIIIIIIYYIII
    - 9.180253761118223 * IIIIIIXIXIII
    - 9.180253761118223 * IIIIIIYIYIII
    - 1036.7964000000004 * IIIIIIIZIIII
    - 74.16262749999984 * IIIIIIXXIIII
    - 74.16262749999984 * IIIIIIYYIIII
    - 1730.939149375 * IIIIIIZIIIII
    - 102.72856234375003 * IIIIIZIIIIII
    - 13.324103454983595 * IIIXIXIIIIII
    - 13.324103454983595 * IIIYIYIIIIII
    - 315.1932645312501 * IIIIZIIIIIII
    - 541.6731217187502 * IIIZIIIIIIII
    - 102.72856234375003 * IIZIIIIIIIII
    - 13.32410345498359 * XIXIIIIIIIII
    - 13.32410345498359 * YIYIIIIIIIII
    - 315.1932645312501 * IZIIIIIIIIII
    - 541.6731217187502 * ZIIIIIIIIIII
    + 1.2356356250000005 * IIIIIIIIZIIZ
    + 22.050435542190833 * IIIIIIIXXIIZ
    + 22.050435542190833 * IIIIIIIYYIIZ
    - 1.7474526590263566 * IIIIIIXIXIIZ
    - 1.7474526590263566 * IIIIIIYIYIIZ
    + 3.706906875000002 * IIIIIIIZIIIZ
    + 31.184025000000005 * IIIIIIXXIIIZ
    + 31.184025000000005 * IIIIIIYYIIIZ
    + 6.178178125000002 * IIIIIIZIIIIZ
    - 1.7474526590263566 * IIIIIIIIZXIX
    - 1.7474526590263566 * IIIIIIIIZYIY
    - 31.184025000000013 * IIIIIIIXXXIX
    - 31.184025000000013 * IIIIIIIYYXIX
    - 31.184025000000013 * IIIIIIIXXYIY
    - 31.184025000000013 * IIIIIIIYYYIY
    + 2.471271250000001 * IIIIIIXIXXIX
    + 2.471271250000001 * IIIIIIYIYXIX
    + 2.471271250000001 * IIIIIIXIXYIY
    + 2.471271250000001 * IIIIIIYIYYIY
    - 5.24235797707907 * IIIIIIIZIXIX
    - 5.24235797707907 * IIIIIIIZIYIY
    - 44.100871084381666 * IIIIIIXXIXIX
    - 44.100871084381666 * IIIIIIYYIXIX
    - 44.100871084381666 * IIIIIIXXIYIY
    - 44.100871084381666 * IIIIIIYYIYIY
    - 8.737263295131783 * IIIIIIZIIXIX
    - 8.737263295131783 * IIIIIIZIIYIY
    + 3.706906875000002 * IIIIIIIIZIZI
    + 66.1513066265725 * IIIIIIIXXIZI
    + 66.1513066265725 * IIIIIIIYYIZI
    - 5.24235797707907 * IIIIIIXIXIZI
    - 5.24235797707907 * IIIIIIYIYIZI
    + 11.120720625000006 * IIIIIIIZIIZI
    + 93.55207500000002 * IIIIIIXXIIZI
    + 93.55207500000002 * IIIIIIYYIIZI
    + 18.534534375000007 * IIIIIIZIIIZI
    + 6.178178125000003 * IIIIIIIIZZII
    + 110.25217771095417 * IIIIIIIXXZII
    + 110.25217771095417 * IIIIIIIYYZII
    - 8.737263295131783 * IIIIIIXIXZII
    - 8.737263295131783 * IIIIIIYIYZII
    + 18.53453437500001 * IIIIIIIZIZII
    + 155.920125 * IIIIIIXXIZII
    + 155.920125 * IIIIIIYYIZII
    + 30.89089062500001 * IIIIIIZIIZII
    - 2.551472812500001 * IIIIIZIIIIIZ
    + 3.608327455463727 * IIIIIZIIIXIX
    + 3.608327455463727 * IIIIIZIIIYIY
    - 7.654418437500004 * IIIIIZIIIIZI
    - 12.757364062500006 * IIIIIZIIIZII
    - 1.0485748437500004 * IIIIIZIIZIII
    - 10.668818275707867 * IIIIIZIXXIII
    - 10.668818275707867 * IIIIIZIYYIII
    + 1.4829087651944997 * IIIIIZXIXIII
    + 1.4829087651944997 * IIIIIZYIYIII
    - 3.1457245312500017 * IIIIIZIZIIII
    - 15.087987500000002 * IIIIIZXXIIII
    - 15.087987500000002 * IIIIIZYYIIII
    - 5.242874218750002 * IIIIIZZIIIII
    + 3.608327455463727 * IIIXIXIIIIIZ
    + 3.608327455463727 * IIIYIYIIIIIZ
    - 5.102945625000002 * IIIXIXIIIXIX
    - 5.102945625000002 * IIIYIYIIIXIX
    - 5.102945625000002 * IIIXIXIIIYIY
    - 5.102945625000002 * IIIYIYIIIYIY
    + 10.82498236639118 * IIIXIXIIIIZI
    + 10.82498236639118 * IIIYIYIIIIZI
    + 18.04163727731863 * IIIXIXIIIZII
    + 18.04163727731863 * IIIYIYIIIZII
    + 1.4829087651944997 * IIIXIXIIZIII
    + 1.4829087651944997 * IIIYIYIIZIII
    + 15.087987500000006 * IIIXIXIXXIII
    + 15.087987500000006 * IIIYIYIXXIII
    + 15.087987500000006 * IIIXIXIYYIII
    + 15.087987500000006 * IIIYIYIYYIII
    - 2.097149687500001 * IIIXIXXIXIII
    - 2.097149687500001 * IIIYIYXIXIII
    - 2.097149687500001 * IIIXIXYIYIII
    - 2.097149687500001 * IIIYIYYIYIII
    + 4.4487262955835 * IIIXIXIZIIII
    + 4.4487262955835 * IIIYIYIZIIII
    + 21.337636551415734 * IIIXIXXXIIII
    + 21.337636551415734 * IIIYIYXXIIII
    + 21.337636551415734 * IIIXIXYYIIII
    + 21.337636551415734 * IIIYIYYYIIII
    + 7.4145438259724985 * IIIXIXZIIIII
    + 7.4145438259724985 * IIIYIYZIIIII
    - 7.654418437500005 * IIIIZIIIIIIZ
    + 10.824982366391183 * IIIIZIIIIXIX
    + 10.824982366391183 * IIIIZIIIIYIY
    - 22.963255312500017 * IIIIZIIIIIZI
    - 38.27209218750002 * IIIIZIIIIZII
    - 3.1457245312500017 * IIIIZIIIZIII
    - 32.006454827123605 * IIIIZIIXXIII
    - 32.006454827123605 * IIIIZIIYYIII
    + 4.4487262955835 * IIIIZIXIXIII
    + 4.4487262955835 * IIIIZIYIYIII
    - 9.437173593750005 * IIIIZIIZIIII
    - 45.26396250000001 * IIIIZIXXIIII
    - 45.26396250000001 * IIIIZIYYIIII
    - 15.728622656250007 * IIIIZIZIIIII
    - 12.757364062500006 * IIIZIIIIIIIZ
    + 18.04163727731863 * IIIZIIIIIXIX
    + 18.04163727731863 * IIIZIIIIIYIY
    - 38.27209218750002 * IIIZIIIIIIZI
    - 63.786820312500026 * IIIZIIIIIZII
    - 5.242874218750002 * IIIZIIIIZIII
    - 53.34409137853934 * IIIZIIIXXIII
    - 53.34409137853934 * IIIZIIIYYIII
    + 7.414543825972498 * IIIZIIXIXIII
    + 7.414543825972498 * IIIZIIYIYIII
    - 15.728622656250007 * IIIZIIIZIIII
    - 75.43993750000001 * IIIZIIXXIIII
    - 75.43993750000001 * IIIZIIYYIIII
    - 26.21437109375001 * IIIZIIZIIIII
    - 2.551472812500001 * IIZIIIIIIIIZ
    + 3.608327455463727 * IIZIIIIIIXIX
    + 3.608327455463727 * IIZIIIIIIYIY
    - 7.654418437500004 * IIZIIIIIIIZI
    - 12.757364062500006 * IIZIIIIIIZII
    - 1.0485748437500004 * IIZIIIIIZIII
    - 10.668818275707867 * IIZIIIIXXIII
    - 10.668818275707867 * IIZIIIIYYIII
    + 1.4829087651944997 * IIZIIIXIXIII
    + 1.4829087651944997 * IIZIIIYIYIII
    - 3.1457245312500017 * IIZIIIIZIIII
    - 15.087987500000002 * IIZIIIXXIIII
    - 15.087987500000002 * IIZIIIYYIIII
    - 5.242874218750002 * IIZIIIZIIIII
    + 1.7745875000000009 * IIZIIZIIIIII
    - 2.509645710117766 * IIZXIXIIIIII
    - 2.509645710117766 * IIZYIYIIIIII
    + 5.323762500000003 * IIZIZIIIIIII
    + 8.872937500000004 * IIZZIIIIIIII
    - 2.8170754092577175 * IXXXXIIIIIII
    - 2.8170754092577175 * IYYXXIIIIIII
    - 2.8170754092577175 * IXXYYIIIIIII
    - 2.8170754092577175 * IYYYYIIIIIII
    + 3.608327455463727 * XIXIIIIIIIIZ
    + 3.608327455463727 * YIYIIIIIIIIZ
    - 5.102945625000002 * XIXIIIIIIXIX
    - 5.102945625000002 * YIYIIIIIIXIX
    - 5.102945625000002 * XIXIIIIIIYIY
    - 5.102945625000002 * YIYIIIIIIYIY
    + 10.82498236639118 * XIXIIIIIIIZI
    + 10.82498236639118 * YIYIIIIIIIZI
    + 18.04163727731863 * XIXIIIIIIZII
    + 18.04163727731863 * YIYIIIIIIZII
    + 1.4829087651944997 * XIXIIIIIZIII
    + 1.4829087651944997 * YIYIIIIIZIII
    + 15.087987500000006 * XIXIIIIXXIII
    + 15.087987500000006 * YIYIIIIXXIII
    + 15.087987500000006 * XIXIIIIYYIII
    + 15.087987500000006 * YIYIIIIYYIII
    - 2.097149687500001 * XIXIIIXIXIII
    - 2.097149687500001 * YIYIIIXIXIII
    - 2.097149687500001 * XIXIIIYIYIII
    - 2.097149687500001 * YIYIIIYIYIII
    + 4.4487262955835 * XIXIIIIZIIII
    + 4.4487262955835 * YIYIIIIZIIII
    + 21.337636551415734 * XIXIIIXXIIII
    + 21.337636551415734 * YIYIIIXXIIII
    + 21.337636551415734 * XIXIIIYYIIII
    + 21.337636551415734 * YIYIIIYYIIII
    + 7.4145438259724985 * XIXIIIZIIIII
    + 7.4145438259724985 * YIYIIIZIIIII
    - 2.509645710117766 * XIXIIZIIIIII
    - 2.509645710117766 * YIYIIZIIIIII
    + 3.5491750000000017 * XIXXIXIIIIII
    + 3.5491750000000017 * YIYXIXIIIIII
    + 3.5491750000000017 * XIXYIYIIIIII
    + 3.5491750000000017 * YIYYIYIIIIII
    - 7.528937130353299 * XIXIZIIIIIII
    - 7.528937130353299 * YIYIZIIIIIII
    - 12.54822855058883 * XIXZIIIIIIII
    - 12.54822855058883 * YIYZIIIIIIII
    - 7.654418437500005 * IZIIIIIIIIIZ
    + 10.824982366391183 * IZIIIIIIIXIX
    + 10.824982366391183 * IZIIIIIIIYIY
    - 22.963255312500017 * IZIIIIIIIIZI
    - 38.27209218750002 * IZIIIIIIIZII
    - 3.1457245312500017 * IZIIIIIIZIII
    - 32.006454827123605 * IZIIIIIXXIII
    - 32.006454827123605 * IZIIIIIYYIII
    + 4.4487262955835 * IZIIIIXIXIII
    + 4.4487262955835 * IZIIIIYIYIII
    - 9.437173593750005 * IZIIIIIZIIII
    - 45.26396250000001 * IZIIIIXXIIII
    - 45.26396250000001 * IZIIIIYYIIII
    - 15.728622656250007 * IZIIIIZIIIII
    + 5.323762500000003 * IZIIIZIIIIII
    - 7.528937130353299 * IZIXIXIIIIII
    - 7.528937130353299 * IZIYIYIIIIII
    + 15.97128750000001 * IZIIZIIIIIII
    + 26.618812500000015 * IZIZIIIIIIII
    + 2.8170754092577175 * XXIIXXIIIIII
    + 2.8170754092577175 * YYIIXXIIIIII
    + 2.8170754092577175 * XXIIYYIIIIII
    + 2.8170754092577175 * YYIIYYIIIIII
    - 12.757364062500006 * ZIIIIIIIIIIZ
    + 18.04163727731863 * ZIIIIIIIIXIX
    + 18.04163727731863 * ZIIIIIIIIYIY
    - 38.27209218750002 * ZIIIIIIIIIZI
    - 63.786820312500026 * ZIIIIIIIIZII
    - 5.242874218750002 * ZIIIIIIIZIII
    - 53.34409137853934 * ZIIIIIIXXIII
    - 53.34409137853934 * ZIIIIIIYYIII
    + 7.414543825972498 * ZIIIIIXIXIII
    + 7.414543825972498 * ZIIIIIYIYIII
    - 15.728622656250007 * ZIIIIIIZIIII
    - 75.43993750000001 * ZIIIIIXXIIII
    - 75.43993750000001 * ZIIIIIYYIIII
    - 26.21437109375001 * ZIIIIIZIIIII
    + 8.872937500000003 * ZIIIIZIIIIII
    - 12.548228550588828 * ZIIXIXIIIIII
    - 12.548228550588828 * ZIIYIYIIIIII
    + 26.61881250000001 * ZIIIZIIIIIII
    + 44.36468750000001 * ZIIZIIIIIIII


Now that the Hamiltonian is ready, it can be used in a quantum algorithm to find information about the vibrational structure of the corresponding molecule. Check out our tutorials on Ground State Calculation and Excited States Calculation to learn more about how to do that in Qiskit Nature!


```python
import qiskit.tools.jupyter

%qiskit_version_table
%qiskit_copyright
```


<h3>Version Information</h3><table><tr><th>Qiskit Software</th><th>Version</th></tr><tr><td><code>qiskit-terra</code></td><td>0.21.1</td></tr><tr><td><code>qiskit-aer</code></td><td>0.10.4</td></tr><tr><td><code>qiskit-ibmq-provider</code></td><td>0.19.2</td></tr><tr><td><code>qiskit-nature</code></td><td>0.5.0</td></tr><tr><td><code>qiskit-finance</code></td><td>0.4.0</td></tr><tr><td><code>qiskit-optimization</code></td><td>0.4.0</td></tr><tr><td><code>qiskit-machine-learning</code></td><td>0.5.0</td></tr><tr><th>System information</th></tr><tr><td>Python version</td><td>3.8.10</td></tr><tr><td>Python compiler</td><td>GCC 9.4.0</td></tr><tr><td>Python build</td><td>default, Jun 22 2022 20:18:18</td></tr><tr><td>OS</td><td>Linux</td></tr><tr><td>CPUs</td><td>12</td></tr><tr><td>Memory (Gb)</td><td>31.267112731933594</td></tr><tr><td colspan='2'>Sat Jul 30 15:16:10 2022 CEST</td></tr></table>



<div style='width: 100%; background-color:#d5d9e0;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'><h3>This code is a part of Qiskit</h3><p>&copy; Copyright IBM 2017, 2022.</p><p>This code is licensed under the Apache License, Version 2.0. You may<br>obtain a copy of this license in the LICENSE.txt file in the root directory<br> of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.<p>Any modifications or derivative works of this code must retain this<br>copyright notice, and modified files need to carry a notice indicating<br>that they have been altered from the originals.</p></div>
