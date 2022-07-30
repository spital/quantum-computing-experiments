# The Property Framework


```python
import numpy as np
```

Qiskit Nature 0.2.0 introduces the _Property_ framework. This framework replaces the legacy driver return types like `QMolecule` and `WatsonHamiltonian` with a more modular and extensible approach.

In this tutorial, we will walk through the framework, explain its most important components and show you, how you can extend it with a custom _property_ yourself.

## What is a `Property`?

At its core, a `Property` is an object complementing some raw data with functions that allow you to transform/manipulate/interpret this raw data. This "definition" is kept rather abstract on purpose, but what it means is essentially the following.
A `Property`:

* represents a physical observable (that's the raw data)
* can be expressed as an operator
* can be evaluated with a wavefunction
* provides an `interpret` method which gives meaning to the eigenvalue of the evaluated qubit operator


```python
from qiskit_nature.second_q.properties import Property, GroupedProperty
```

The `qiskit_nature.second_q.properties` module provides two classes:

1. `Property`: this is the basic interface. It requires only a `name` and an `interpret` method to be implemented.
2. `GroupedProperty`: this class is an implementation of the [Composite pattern](https://en.wikipedia.org/wiki/Composite_pattern) which allows you to _group_ multiple properties into one.

**Note:** Grouped properties must have unique `name` attributes!

## Second Quantization Properties

At the time of writing, Qiskit Nature ships with a single variant of properties: the `SecondQuantizedProperty` objects.

This sub-type adds one additional requirement because any `SecondQuantizedProperty`

* **must** implement a `second_q_ops` method which constructs a (list of) `SecondQuantizedOp`s.

The `qiskit_nature.second_q.properties` module is further divided into `electronic` and `vibrational` modules (similar to other modules of Qiskit Nature).
Let us dive into the `electronic` sub-module first.

### Electronic Second Quantization Properties

Out-of-the-box Qiskit Nature ships the following electronic properties:


```python
from qiskit_nature.second_q.properties import (
    ElectronicEnergy,
    ElectronicDipoleMoment,
    ParticleNumber,
    AngularMomentum,
    Magnetization,
)
```

`ElectronicEnergy` is arguably the most important property because it contains the _Hamiltonian_ representing the electronic structure problem whose eigenvalues we are interested in when solving an `ElectronicStructureProblem`.
The way in which it stores this Hamiltonian is, just like the rest of the framework, highly modular. The initializer of the `ElectronicEnergy` has a single required argument, `electronic_integrals`, which is a `List[ElectronicIntegrals]`.


#### ElectronicIntegrals


```python
from qiskit_nature.second_q.properties.integrals import (
    ElectronicIntegrals,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
    IntegralProperty,
)
```

The `ElectronicIntegrals` are a container class for _n-body_ interactions in a given basis, which can be any of the following:


```python
from qiskit_nature.second_q.properties.bases import ElectronicBasis
```


```python
list(ElectronicBasis)
```




    [<ElectronicBasis.AO: 'atomic'>,
     <ElectronicBasis.MO: 'molecular'>,
     <ElectronicBasis.SO: 'spin'>]



Let us take the `OneBodyElectronicIntegrals` as an example. As the name suggests, this container is for 1-body interaction integrals. You can construct an instance of it like so:


```python
one_body_ints = OneBodyElectronicIntegrals(
    ElectronicBasis.MO,
    (
        np.eye(2),
        2 * np.eye(2),
    ),
)
print(one_body_ints)
```

    (MO) 1-Body Terms:
    	Alpha
    	<(2, 2) matrix with 2 non-zero entries>
    	[0, 0] = 1.0
    	[1, 1] = 1.0
    	Beta
    	<(2, 2) matrix with 2 non-zero entries>
    	[0, 0] = 2.0
    	[1, 1] = 2.0


As you can see, the first argument simply specifies the basis of the integrals.
The second argument requires a little further explanation:

1. In the case of the `AO` or `MO` basis, this argument **must** be a pair of numpy arrays, where the first and second one are the alpha- and beta-spin integrals, respectively.

**Note:** The second argument may be `None`, for the case where the beta-spin integrals are the same as the alpha-spin integrals (so there is no need to provide the same values twice).

2. In the case of the `SO` basis, this argument **must** be a single numpy array, storing the alpha- and beta-spin integrals in blocked order, i.e. like so:
```python
spin_basis = np.block([[alpha_spin, zeros], [zeros, beta_spin]])
```

The `TwoBodyElectronicIntegrals` work pretty much the same except that they contain all possible spin-combinations in the tuple of numpy arrays. For example:


```python
two_body_ints = TwoBodyElectronicIntegrals(
    ElectronicBasis.MO,
    (
        np.arange(1, 17).reshape((2, 2, 2, 2)),
        np.arange(16, 32).reshape((2, 2, 2, 2)),
        np.arange(-16, 0).reshape((2, 2, 2, 2)),
        None,
    ),
)
print(two_body_ints)
```

    (MO) 2-Body Terms:
    	Alpha-Alpha
    	<(2, 2, 2, 2) matrix with 16 non-zero entries>
    	[0, 0, 0, 0] = 1.0
    	[0, 0, 0, 1] = 2.0
    	[0, 0, 1, 0] = 3.0
    	[0, 0, 1, 1] = 4.0
    	[0, 1, 0, 0] = 5.0
    	... skipping 11 entries
    	Beta-Alpha
    	<(2, 2, 2, 2) matrix with 16 non-zero entries>
    	[0, 0, 0, 0] = 16.0
    	[0, 0, 0, 1] = 17.0
    	[0, 0, 1, 0] = 18.0
    	[0, 0, 1, 1] = 19.0
    	[0, 1, 0, 0] = 20.0
    	... skipping 11 entries
    	Beta-Beta
    	<(2, 2, 2, 2) matrix with 16 non-zero entries>
    	[0, 0, 0, 0] = -16.0
    	[0, 0, 0, 1] = -15.0
    	[0, 0, 1, 0] = -14.0
    	[0, 0, 1, 1] = -13.0
    	[0, 1, 0, 0] = -12.0
    	... skipping 11 entries
    	Alpha-Beta
    	Same values as the corresponding primary-spin data.


We should take note of a few observations:

* the numpy arrays shall be ordered as `("alpha-alpha", "beta-alpha", "beta-beta", "alpha-beta")`
* the `alpha-alpha` matrix may **not** be `None`
* if the `alpha-beta` matrix is `None`, but `beta-alpha` is not, its transpose will be used (like above)
* in any other case, matrices which are `None` will be filled with the `alpha-alpha` matrix

* in the `SO` basis case, a single numpy array must be specified. Refer to `TwoBodyElectronicIntegrals.to_spin()` for its exact formatting.

#### ElectronicEnergy

Now we are ready to construct an `ElectronicEnergy` instance:


```python
electronic_energy = ElectronicEnergy(
    [one_body_ints, two_body_ints],
)
print(electronic_energy)
```

    ElectronicEnergy
    	(MO) 1-Body Terms:
    		Alpha
    		<(2, 2) matrix with 2 non-zero entries>
    		[0, 0] = 1.0
    		[1, 1] = 1.0
    		Beta
    		<(2, 2) matrix with 2 non-zero entries>
    		[0, 0] = 2.0
    		[1, 1] = 2.0
    	(MO) 2-Body Terms:
    		Alpha-Alpha
    		<(2, 2, 2, 2) matrix with 16 non-zero entries>
    		[0, 0, 0, 0] = 1.0
    		[0, 0, 0, 1] = 2.0
    		[0, 0, 1, 0] = 3.0
    		[0, 0, 1, 1] = 4.0
    		[0, 1, 0, 0] = 5.0
    		... skipping 11 entries
    		Beta-Alpha
    		<(2, 2, 2, 2) matrix with 16 non-zero entries>
    		[0, 0, 0, 0] = 16.0
    		[0, 0, 0, 1] = 17.0
    		[0, 0, 1, 0] = 18.0
    		[0, 0, 1, 1] = 19.0
    		[0, 1, 0, 0] = 20.0
    		... skipping 11 entries
    		Beta-Beta
    		<(2, 2, 2, 2) matrix with 16 non-zero entries>
    		[0, 0, 0, 0] = -16.0
    		[0, 0, 0, 1] = -15.0
    		[0, 0, 1, 0] = -14.0
    		[0, 0, 1, 1] = -13.0
    		[0, 1, 0, 0] = -12.0
    		... skipping 11 entries
    		Alpha-Beta
    		Same values as the corresponding primary-spin data.


This property can now be used to construct a `SecondQuantizedOp` (which can then be mapped to a `QubitOperator`):


```python
hamiltonian = electronic_energy.second_q_ops()[0]  # here, output length is always 1
print(hamiltonian)
```

    Fermionic Operator
    register length=4, number terms=68
      1.0 * ( +_0 -_0 )
    + 1.0 * ( +_1 -_1 )
    + 2.0 * ( +_2 -_2 )
    + 2.0 * ( +_3 -_3 )
    + -0.5 * ( +_0 +_0 -_0 -_0 )
    + -1.5 * ( +_0 +_0 -_1 -_0 )
    + -4.5 * ( +_0 +_1 -_0 -_0 )
    + -5.5 * ( +_0 +_1 -_1 -_0 )
     ...


    /tmp/ipykernel_107906/2798686583.py:1: ListAuxOpsDeprecationWarning: List-based `aux_operators` are deprecated as of version 0.3.0 and support for them will be removed no sooner than 3 months after the release. Instead, use dict-based `aux_operators`. You can switch to the dict-based interface immediately, by setting `qiskit_nature.settings.dict_aux_operators` to `True`.
      hamiltonian = electronic_energy.second_q_ops()[0]  # here, output length is always 1


#### Result interpretation

An additional benefit which we gain from the `Property` framework, is that the result interpretation of a computed eigenvalue can be handled by each property itself. This results in nice and logically consistent classes because the result gets interpreted in the same context where the raw data is available.


```python
from qiskit_nature.second_q.problems import ElectronicStructureResult

# some dummy result
result = ElectronicStructureResult()
result.eigenenergies = np.asarray([-1])
result.computed_energies = np.asarray([-1])


# now, let's interpret it
electronic_energy.interpret(result)
print(result)
```

    === GROUND STATE ENERGY ===
     
    * Electronic ground state energy (Hartree): -1.
      - computed part:      -1.


While this particular example is not yet very impressive, wait until we use more properties at once.

#### ParticleNumber

The `ParticleNumber` property also takes a special place among the builtin properties because it serves a dual purpose of:

* storing the number of particles in the electronic system
* providing the operators to evaluate the number of particles for a given eigensolution of the problem

Therefore, this property is required if you want to use additional functionality like the `ActiveSpaceTransformer` or the `ElectronicStructureProblem.default_filter_criterion()`.


```python
particle_number = ParticleNumber(
    num_spin_orbitals=4,
    num_particles=(1, 1),
)
print(particle_number)
```

    ParticleNumber:
    	4 SOs
    	1 alpha electrons
    		orbital occupation: [1. 0.]
    	1 beta electrons
    		orbital occupation: [1. 0.]


#### GroupedProperty

Rather than iterating all of the other properties one by one, let us simply consider a group of properties as provided by any `ElectronicStructureDriver` from the `qiskit_nature.second_q.drivers` module.


```python
from qiskit_nature.second_q.drivers.pyscfd import PySCFDriver
```


```python
electronic_driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto3g")
electronic_driver_result = electronic_driver.run()
```


```python
print(electronic_driver_result)
```

    ElectronicStructureDriverResult:
    	DriverMetadata:
    		Program: PYSCF
    		Version: 2.0.1
    		Config:
    			atom=H 0 0 0; H 0 0 0.735
    			unit=Angstrom
    			charge=0
    			spin=0
    			basis=sto3g
    			method=rhf
    			conv_tol=1e-09
    			max_cycle=50
    			init_guess=minao
    			max_memory=4000
    			
    	ElectronicBasisTransform:
    		Initial basis: atomic
    		Final basis: molecular
    		Alpha coefficients:
    		[0, 0] = 0.5483020229014732
    		[0, 1] = 1.2183273138546826
    		[1, 0] = 0.548302022901473
    		[1, 1] = -1.2183273138546826
    	ParticleNumber:
    		4 SOs
    		1 alpha electrons
    			orbital occupation: [1. 0.]
    		1 beta electrons
    			orbital occupation: [1. 0.]
    	ElectronicEnergy
    		(AO) 1-Body Terms:
    			Alpha
    			<(2, 2) matrix with 4 non-zero entries>
    			[0, 0] = -1.1242175791954514
    			[0, 1] = -0.9652573993472754
    			[1, 0] = -0.9652573993472754
    			[1, 1] = -1.1242175791954514
    			Beta
    			Same values as the corresponding primary-spin data.
    		(AO) 2-Body Terms:
    			Alpha-Alpha
    			<(2, 2, 2, 2) matrix with 16 non-zero entries>
    			[0, 0, 0, 0] = 0.7746059439198978
    			[0, 0, 0, 1] = 0.4474457245330949
    			[0, 0, 1, 0] = 0.447445724533095
    			[0, 0, 1, 1] = 0.5718769760004512
    			[0, 1, 0, 0] = 0.4474457245330951
    			... skipping 11 entries
    			Beta-Alpha
    			Same values as the corresponding primary-spin data.
    			Beta-Beta
    			Same values as the corresponding primary-spin data.
    			Alpha-Beta
    			Same values as the corresponding primary-spin data.
    		(MO) 1-Body Terms:
    			Alpha
    			<(2, 2) matrix with 2 non-zero entries>
    			[0, 0] = -1.2563390730032498
    			[1, 1] = -0.47189600728114245
    			Beta
    			Same values as the corresponding primary-spin data.
    		(MO) 2-Body Terms:
    			Alpha-Alpha
    			<(2, 2, 2, 2) matrix with 8 non-zero entries>
    			[0, 0, 0, 0] = 0.6757101548035163
    			[0, 0, 1, 1] = 0.6645817302552965
    			[0, 1, 0, 1] = 0.1809311997842313
    			[0, 1, 1, 0] = 0.1809311997842311
    			[1, 0, 0, 1] = 0.18093119978423147
    			... skipping 3 entries
    			Beta-Alpha
    			Same values as the corresponding primary-spin data.
    			Beta-Beta
    			Same values as the corresponding primary-spin data.
    			Alpha-Beta
    			Same values as the corresponding primary-spin data.
    	ElectronicDipoleMoment:
    		DipoleMomentX
    			(AO) 1-Body Terms:
    				Alpha
    				<(2, 2) matrix with 0 non-zero entries>
    				Beta
    				Same values as the corresponding primary-spin data.
    			(MO) 1-Body Terms:
    				Alpha
    				<(2, 2) matrix with 0 non-zero entries>
    				Beta
    				Same values as the corresponding primary-spin data.
    		DipoleMomentY
    			(AO) 1-Body Terms:
    				Alpha
    				<(2, 2) matrix with 0 non-zero entries>
    				Beta
    				Same values as the corresponding primary-spin data.
    			(MO) 1-Body Terms:
    				Alpha
    				<(2, 2) matrix with 0 non-zero entries>
    				Beta
    				Same values as the corresponding primary-spin data.
    		DipoleMomentZ
    			(AO) 1-Body Terms:
    				Alpha
    				<(2, 2) matrix with 3 non-zero entries>
    				[0, 1] = 0.46053770796603194
    				[1, 0] = 0.46053770796603194
    				[1, 1] = 1.3889487015553206
    				Beta
    				Same values as the corresponding primary-spin data.
    			(MO) 1-Body Terms:
    				Alpha
    				<(2, 2) matrix with 4 non-zero entries>
    				[0, 0] = 0.6944743507776598
    				[0, 1] = -0.927833470459232
    				[1, 0] = -0.9278334704592321
    				[1, 1] = 0.6944743507776604
    				Beta
    				Same values as the corresponding primary-spin data.
    	AngularMomentum:
    		4 SOs
    	Magnetization:
    		4 SOs
    Molecule:
    	Multiplicity: 1
    	Charge: 0
    	Unit: Angstrom
    	Geometry:
    		H	[0.0, 0.0, 0.0]
    		H	[0.0, 0.0, 0.735]
    	Masses:
    		H	1
    		H	1


There is a lot going on but with the explanations above you should not have any problems with understanding this output.

#### Constructing a Hamiltonian from raw integrals

If you have obtained some raw one- and two-body integrals by means other than through a driver provided by Qiskit Nature, you can still easily construct an `ElectronicEnergy` property to serve as your access point into the stack:


```python
one_body_ints = np.arange(1, 5).reshape((2, 2))
two_body_ints = np.arange(1, 17).reshape((2, 2, 2, 2))
electronic_energy_from_ints = ElectronicEnergy.from_raw_integrals(
    ElectronicBasis.MO, one_body_ints, two_body_ints
)
print(electronic_energy_from_ints)
```

    ElectronicEnergy
    	(MO) 1-Body Terms:
    		Alpha
    		<(2, 2) matrix with 4 non-zero entries>
    		[0, 0] = 1.0
    		[0, 1] = 2.0
    		[1, 0] = 3.0
    		[1, 1] = 4.0
    		Beta
    		Same values as the corresponding primary-spin data.
    	(MO) 2-Body Terms:
    		Alpha-Alpha
    		<(2, 2, 2, 2) matrix with 16 non-zero entries>
    		[0, 0, 0, 0] = 1.0
    		[0, 0, 0, 1] = 2.0
    		[0, 0, 1, 0] = 3.0
    		[0, 0, 1, 1] = 4.0
    		[0, 1, 0, 0] = 5.0
    		... skipping 11 entries
    		Beta-Alpha
    		Same values as the corresponding primary-spin data.
    		Beta-Beta
    		Same values as the corresponding primary-spin data.
    		Alpha-Beta
    		Same values as the corresponding primary-spin data.


### Vibrational Second Quantization Properties

The `vibrational` stack for vibrational structure calculations also integrates with the Property framework. After the above introduction you should be able to understand the following directly:


```python
from qiskit_nature.second_q.drivers.gaussiand import GaussianForcesDriver
```


```python
# if you ran Gaussian elsewhere and already have the output file
vibrational_driver = GaussianForcesDriver(logfile="aux_files/CO2_freq_B3LYP_631g.log")
vibrational_driver_result = vibrational_driver.run()
```

For vibrational structure calculations we always need to define the basis which we want to work in, separately:


```python
from qiskit_nature.second_q.properties.bases import HarmonicBasis
```


```python
harmonic_basis = HarmonicBasis([2] * 4)
```


```python
vibrational_driver_result.basis = harmonic_basis
print(vibrational_driver_result)
```

    VibrationalStructureDriverResult:
    	HarmonicBasis:
    		Modals: [2, 2, 2, 2]:
    	VibrationalEnergy:
    		HarmonicBasis:
    			Modals: [2, 2, 2, 2]
    		1-Body Terms:
    			<sparse integral list with 13 entries>
    			(2, 2) = 352.3005875
    			(-2, -2) = -352.3005875
    			(1, 1) = 631.6153975
    			(-1, -1) = -631.6153975
    			(4, 4) = 115.653915
    			... skipping 8 entries
    		2-Body Terms:
    			<sparse integral list with 11 entries>
    			(1, 1, 2) = -88.2017421687633
    			(4, 4, 2) = 42.675273102831454
    			(3, 3, 2) = 42.675273102831454
    			(1, 1, 2, 2) = 4.9425425
    			(4, 4, 2, 2) = -4.194299375
    			... skipping 6 entries
    		3-Body Terms:
    			<sparse integral list with 0 entries>
    	OccupiedModals:
    		HarmonicBasis:
    			Modals: [2, 2, 2, 2]


## Writing custom Properties

You can extend the Property framework with your own implementations. Here, we will walk through the simple example of constructing a Property for the _electronic density_. Since this observable is essentially based on matrices, we will be leveraging the `OneBodyElectronicIntegrals` container to store the actual density matrix.


```python
from itertools import product
from typing import List

import h5py

from qiskit_nature.second_q._qmolecule import QMolecule
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.properties.bases import ElectronicBasis
from qiskit_nature.second_q.properties.electronic_types import ElectronicProperty
from qiskit_nature.second_q.properties.integrals import (
    OneBodyElectronicIntegrals,
)
```


```python
class ElectronicDensity(ElectronicProperty):
    """A simple electronic density property.

    This basic example works only in the MO basis!
    """

    def __init__(self, num_molecular_orbitals: int) -> None:
        super().__init__(self.__class__.__name__)
        self._num_molecular_orbitals = num_molecular_orbitals

    def __str__(self) -> str:
        string = [super().__str__() + ":"]
        string += [f"\t{self._num_molecular_orbitals} MOs"]
        return "\n".join(string)

    def to_hdf5(self, parent: h5py.Group) -> None:
        super().to_hdf5(parent)
        group = parent.require_group(self.name)

        group.attrs["num_molecular_orbitals"] = self._num_molecular_orbitals

    @classmethod
    def from_hdf5(cls, h5py_group: h5py.Group) -> "ElectronicDensity":
        return ElectronicDensity(h5py_group.attrs["num_molecular_orbitals"])

    @classmethod
    def from_legacy_driver_result(cls, result) -> "ElectronicDensity":
        cls._validate_input_type(result, QMolecule)

        qmol = cast(QMolecule, result)

        return cls(qmol.num_molecular_orbitals)

    def second_q_ops(self) -> List[FermionicOp]:
        ops = []

        # iterate all pairs of molecular orbitals
        for mo_i, mo_j in product(range(self._num_molecular_orbitals), repeat=2):

            # construct an auxiliary matrix where the only non-zero entry is at the current pair of MOs
            number_op_matrix = np.zeros(
                (self._num_molecular_orbitals, self._num_molecular_orbitals)
            )
            number_op_matrix[mo_i, mo_j] = 1

            # leverage the OneBodyElectronicIntegrals to construct the corresponding FermionicOp
            one_body_ints = OneBodyElectronicIntegrals(
                ElectronicBasis.MO, (number_op_matrix, number_op_matrix)
            )
            ops.append(one_body_ints.to_second_q_op())

        return ops

    def interpret(self, result) -> None:
        # here goes the code which interprets the eigenvalues returned for the auxiliary operators
        pass
```


```python
density = ElectronicDensity(2)
```


```python
print(density)
```

    ElectronicDensity:
    	2 MOs



```python
for idx, op in enumerate(density.second_q_ops()):
    print(idx, ":", op)
```

    0 : Fermionic Operator
    register length=4, number terms=2
      1.0 * ( +_0 -_0 )
    + 1.0 * ( +_2 -_2 )
    1 : Fermionic Operator
    register length=4, number terms=2
      1.0 * ( +_0 -_1 )
    + 1.0 * ( +_2 -_3 )
    2 : Fermionic Operator
    register length=4, number terms=2
      1.0 * ( +_1 -_0 )
    + 1.0 * ( +_3 -_2 )
    3 : Fermionic Operator
    register length=4, number terms=2
      1.0 * ( +_1 -_1 )
    + 1.0 * ( +_3 -_3 )


Of course, the above example is very minimal and can be extended at will.

**Note:** as of Qiskit Nature version 0.2.0, the direct integration of custom Property objects into the stack is not implemented yet, due to limitations of the auxiliary operator parsing. See https://github.com/Qiskit/qiskit-nature/issues/312 for more details.

For the time being, you can still evaluate a custom Property, by passing it's generated operators directly to the `Eigensolver.solve` method by means of constructing the `aux_operators`. Note, however, that you will have to deal with transformations applied to your properties manually, until the above issue is resolved.


```python
# set up some problem
problem = ...
# set up a solver
solver = ...
# when solving the problem, pass additional operators in like so:
aux_ops = density.second_q_ops()
# solver.solve(problem, aux_ops)
```


```python
import qiskit.tools.jupyter

%qiskit_version_table
%qiskit_copyright
```


<h3>Version Information</h3><table><tr><th>Qiskit Software</th><th>Version</th></tr><tr><td><code>qiskit-terra</code></td><td>0.21.1</td></tr><tr><td><code>qiskit-aer</code></td><td>0.10.4</td></tr><tr><td><code>qiskit-ibmq-provider</code></td><td>0.19.2</td></tr><tr><td><code>qiskit-nature</code></td><td>0.5.0</td></tr><tr><td><code>qiskit-finance</code></td><td>0.4.0</td></tr><tr><td><code>qiskit-optimization</code></td><td>0.4.0</td></tr><tr><td><code>qiskit-machine-learning</code></td><td>0.5.0</td></tr><tr><th>System information</th></tr><tr><td>Python version</td><td>3.8.10</td></tr><tr><td>Python compiler</td><td>GCC 9.4.0</td></tr><tr><td>Python build</td><td>default, Jun 22 2022 20:18:18</td></tr><tr><td>OS</td><td>Linux</td></tr><tr><td>CPUs</td><td>12</td></tr><tr><td>Memory (Gb)</td><td>31.267112731933594</td></tr><tr><td colspan='2'>Sat Jul 30 15:19:16 2022 CEST</td></tr></table>



<div style='width: 100%; background-color:#d5d9e0;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'><h3>This code is a part of Qiskit</h3><p>&copy; Copyright IBM 2017, 2022.</p><p>This code is licensed under the Apache License, Version 2.0. You may<br>obtain a copy of this license in the LICENSE.txt file in the root directory<br> of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.<p>Any modifications or derivative works of this code must retain this<br>copyright notice, and modified files need to carry a notice indicating<br>that they have been altered from the originals.</p></div>

