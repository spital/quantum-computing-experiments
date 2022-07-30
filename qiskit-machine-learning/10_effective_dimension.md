# Effective Dimension of Qiskit Neural Networks
In this tutorial, we will take advantage of the `EffectiveDimension` and `LocalEffectiveDimension` classes to evaluate the power of Quantum Neural Network models. These are metrics based on information geometry that connect to notions such as trainability, expressibility or ability to generalize.

Before diving into the code example, we will briefly explain what is the difference between these two metrics, and why are they relevant to the study of Quantum Neural Networks. More information about global effective dimension can be found in [this paper](https://arxiv.org/pdf/2011.00027.pdf), while the local effective dimension was introduced in a [later work](https://arxiv.org/abs/2112.04807).

## 1. Global vs. Local Effective Dimension
Both classical and quantum machine learning models share a common goal: being good at **generalizing**, i.e. learning insights from data and applying them on unseen data.

Finding a good metric to assess this ability is a non-trivial matter. In [The Power of Quantum Neural Networks](https://arxiv.org/pdf/2011.00027.pdf), the authors introduce the **global** effective dimension as a useful indicator of how well a particular model will be able to perform on new data. In [Effective Dimension of Machine Learning Models](https://arxiv.org/pdf/2112.04807.pdf), the **local** effective dimension is proposed as a new capacity measure that bounds the generalization error of machine learning models.

The key difference between global (`EffectiveDimension` class) and **local** effective dimension (`LocalEffectiveDimension` class) is actually not in the way they are computed, but in the nature of the parameter space that is analyzed. The global effective dimension incorporates the **full parameter space** of the model, and is calculated from a **large number of parameter (weight) sets**. On the other hand, the local effective dimension focuses on how well the **trained** model can generalize to new data, and how **expressive** it can be. Therefore, the local effective dimension is calculated from **a single** set of weight samples (training result). This difference is small in terms of practical implementation, but quite relevant at a conceptual level.

## 2. The Effective Dimension Algorithm

Both the global and local effective dimension algorithms use the Fisher Information matrix to provide a measure of complexity. The details on how this matrix is calculated are provided in the [reference paper](https://arxiv.org/pdf/2011.00027.pdf), but in general terms, this matrix captures how sensitive a neural network's output is to changes in the network's parameter space.

In particular, this algorithm follows 4 main steps:

1. **Monte Carlo simulation:** the forward and backward passes (gradients) of the neural network are computed for each pair of input and weight samples.
2. **Fisher Matrix Computation:** these outputs and gradients are used to compute the Fisher Information Matrix.
3. **Fisher Matrix Normalization:** averaging over all input samples and dividing by the matrix trace
4. **Effective Dimension Calculation:** according to the formula from [*Abbas et al.*](https://arxiv.org/pdf/2011.00027.pdf)

## 3. Basic Example (CircuitQNN)

This example shows how to set up a QNN model problem and run the global effective dimension algorithm. Both Qiskit `CircuitQNN` (shown in this example) and `OpflowQNN` (shown in a later example) can be used with the `EffectiveDimension` class.

We start off from the required imports and a fixed seed for the random number generator for reproducibility purposes.


```python
# Necessary imports
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, RealAmplitudes
import matplotlib.pyplot as plt
import numpy as np

from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit import Aer, QuantumCircuit

from qiskit_machine_learning.neural_networks import EffectiveDimension, LocalEffectiveDimension

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit.algorithms.optimizers import COBYLA

from IPython.display import clear_output

# set random seed
algorithm_globals.random_seed = 42
```

### 3.1 Define QNN

The first step to create a `CircuitQNN` is to define a parametrized feature map and ansatz. In this toy example, we will use 3 qubits, and we will define the circuit used in the `TwoLayerQNN` class.


```python
num_qubits = 3
# create a feature map
feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=1)
# create a variational circuit
ansatz = RealAmplitudes(num_qubits, reps=1)

# combine feature map and ansatz into a single circuit
qc = QuantumCircuit(num_qubits)
qc.append(feature_map, range(num_qubits))
qc.append(ansatz, range(num_qubits))
qc.decompose().draw("mpl")
```




    
![png](10_effective_dimension_files/10_effective_dimension_6_0.png)
    



The parametrized circuit can then be sent together with an optional interpret map (parity in this case) to the `CircuitQNN` constructor.


```python
# parity maps bitstrings to 0 or 1
def parity(x):
    return "{:b}".format(x).count("1") % 2


output_shape = 2  # corresponds to the number of classes, possible outcomes of the (parity) mapping.
```


```python
# declare quantum instance
qi_sv = QuantumInstance(Aer.get_backend("aer_simulator_statevector"))

# construct QNN
qnn = CircuitQNN(
    qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    interpret=parity,
    output_shape=output_shape,
    sparse=False,
    quantum_instance=qi_sv,
)
```

### 3.2 Set up Effective Dimension calculation

In order to compute the effective dimension of our QNN using the `EffectiveDimension` class, we need a series of sets of input samples and weights, as well as the total number of data samples available in a dataset. The `input_samples` and `weight_samples` are set in the class constructor, while the number of data samples is given during the call to the effective dimension computation, to be able to test and compare how this measure changes with different dataset sizes.

We can define the number of input samples and weight samples and the class will randomly sample a corresponding array from a normal (for `input_samples`) or a uniform (for `weight_samples`) distribution. Instead of passing a number of samples we can pass an array, sampled manually.


```python
# we can set the total number of input samples and weight samples for random selection
num_input_samples = 10
num_weight_samples = 10

global_ed = EffectiveDimension(
    qnn=qnn, weight_samples=num_weight_samples, input_samples=num_input_samples
)
```

If we want to test a specific set of input samples and weight samples, we can provide it directly to the `EffectiveDimension` class as shown in the following snippet:


```python
# we can also provide user-defined samples and parameters
input_samples = algorithm_globals.random.normal(0, 1, size=(10, qnn.num_inputs))
weight_samples = algorithm_globals.random.uniform(0, 1, size=(10, qnn.num_weights))

global_ed = EffectiveDimension(qnn=qnn, weight_samples=weight_samples, input_samples=input_samples)
```

The effective dimension algorithm also requires a dataset size. In this example, we will define an array of sizes to later see how this input affects the result.


```python
# finally, we will define ranges to test different numbers of data, n
n = [5000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]
```

### 3.3 Compute Global Effective Dimension
Let's now calculate the effective dimension of our network for the previously defined set of input samples, weights, and a dataset size of 5000.


```python
global_eff_dim_0 = global_ed.get_effective_dimension(dataset_size=n[0])
```

The effective dimension values will range between 0 and `d`, where `d` represents the dimension of the model, and it's practically obtained from the number of weights of the QNN. By dividing the result by `d`, we can obtain the normalized effective dimension, which correlates directly with the capacity of the model.


```python
d = qnn.num_weights

print("Data size: {}, global effective dimension: {:.4f}".format(n[0], global_eff_dim_0))
print(
    "Number of weights: {}, normalized effective dimension: {:.4f}".format(d, global_eff_dim_0 / d)
)
```

    Data size: 5000, global effective dimension: 4.6657
    Number of weights: 6, normalized effective dimension: 0.7776


By calling the `EffectiveDimension` class with an array if input sizes `n`, we can monitor how the effective dimension changes with the dataset size.


```python
global_eff_dim_1 = global_ed.get_effective_dimension(dataset_size=n)
```


```python
print("Effective dimension: {}".format(global_eff_dim_1))
print("Number of weights: {}".format(d))
```

    Effective dimension: [4.66565096 4.7133723  4.73782922 4.89963559 4.94632272 5.00280009
     5.04530433 5.07408394 5.15786005 5.21349874]
    Number of weights: 6



```python
# plot the normalized effective dimension for the model
plt.plot(n, np.array(global_eff_dim_1) / d)
plt.xlabel("Number of data")
plt.ylabel("Normalized GLOBAL effective dimension")
plt.show()
```


    
![png](10_effective_dimension_files/10_effective_dimension_24_0.png)
    


## 4. Local Effective Dimension Example
As explained in the introduction, the local effective dimension algorithm only uses **one** set of weights, and it can be used to monitor how training affects the expressiveness of a neural network. The `LocalEffectiveDimension` class enforces this constraint to ensure that these calculations are conceptually separate, but the rest of the implementation is shared with `EffectiveDimension`.

This example shows how to leverage the `LocalEffectiveDimension` class to analyze the effect of training on QNN expressiveness.

### 4.1 Define Dataset and QNN

We start by creating a 3D binary classification dataset:


```python
num_inputs = 3
num_samples = 50
X = algorithm_globals.random.normal(0, 0.5, size=(num_samples, num_inputs))

y01 = 1 * (np.sum(X, axis=1) >= 0)  # in { 0,  1}
y = 2 * y01 - 1  # in {-1, +1}
```

The next step is to create a QNN, an instance of `TwoLayerQNN` in our case. Since we pass only the number of inputs, the network will continue with the default values for feature map and ansatz.


```python
opflow_qnn = TwoLayerQNN(num_inputs, quantum_instance=qi_sv)
```

### 4.2 Train QNN

We can now proceed to train the QNN. The training step may take some time, be patient. You can pass a callback to the classifier to observe how the training process is going on. We fix `initial_point` for reproducibility purposes as usual.


```python
# callback function that draws a live plot when the .fit() method is called
def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()
```


```python
# construct classifier
initial_point = algorithm_globals.random.random(opflow_qnn.num_weights)

opflow_classifier = NeuralNetworkClassifier(
    neural_network=opflow_qnn,
    optimizer=COBYLA(maxiter=150),
    initial_point=initial_point,
    callback=callback_graph,
)
```


```python
# create empty array for callback to store evaluations of the objective function (callback)
objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)

# fit classifier to data
opflow_classifier.fit(X, y)

# return to default figsize
plt.rcParams["figure.figsize"] = (6, 4)
```


    
![png](10_effective_dimension_files/10_effective_dimension_33_0.png)
    


The classifier can now differentiate between classes with an accuracy of:


```python
# score classifier
opflow_classifier.score(X, y)
```




    0.68



### 4.3 Compute Local Effective Dimension of trained QNN

Now that we have trained our network, let's evaluate the local effective dimension based on the trained weights. To do that we access the trained weights directly from the classifier.


```python
trained_weights = opflow_classifier.weights

# get Local Effective Dimension for set of trained weights
local_ed_trained = LocalEffectiveDimension(
    qnn=opflow_qnn, weight_samples=trained_weights, input_samples=X
)

local_eff_dim_trained = local_ed_trained.get_effective_dimension(dataset_size=n)

print(
    "normalized local effective dimensions for trained QNN: ",
    local_eff_dim_trained / opflow_qnn.num_weights,
)
```

    normalized local effective dimensions for trained QNN:  [0.79663244 0.80325759 0.80653351 0.82723511 0.83320702 0.84062917
     0.84641928 0.85045673 0.86276589 0.87134912]


### 4.4 Compute Local Effective Dimension of untrained QNN

We can compare this result with the effective dimension of the untrained network, using the `initial_point` as our weight sample:


```python
# get Local Effective Dimension for set of untrained weights
local_ed_untrained = LocalEffectiveDimension(
    qnn=opflow_qnn, weight_samples=initial_point, input_samples=X
)

local_eff_dim_untrained = local_ed_untrained.get_effective_dimension(dataset_size=n)

print(
    "normalized local effective dimensions for untrained QNN: ",
    local_eff_dim_untrained / opflow_qnn.num_weights,
)
```

    normalized local effective dimensions for untrained QNN:  [0.80896667 0.81612261 0.81966781 0.84219603 0.84864578 0.85651291
     0.86249025 0.86656428 0.8785217  0.88651616]


### 4.5 Plot and analyze results

If we plot the effective dimension values before and after training, we can see the following result:


```python
# plot the normalized effective dimension for the model
plt.plot(n, np.array(local_eff_dim_trained) / opflow_qnn.num_weights, label="trained weights")
plt.plot(n, np.array(local_eff_dim_untrained) / opflow_qnn.num_weights, label="untrained weights")

plt.xlabel("Number of data")
plt.ylabel("Normalized LOCAL effective dimension")
plt.legend()
plt.show()
```


    
![png](10_effective_dimension_files/10_effective_dimension_41_0.png)
    


In general, we should expect the value of the local effective dimension to decrease after training. This can be understood by looking back into the main goal of machine learning, which is to pick a model that is expressive enough to fit your data, but not too expressive that it overfits and performs badly on new data samples.  

Certain optimizers help regularize the overfitting of a model by learning parameters, and this action of learning inherently reduces a model’s expressiveness, as measured by the local effective dimension. Following this logic, a randomly initialized parameter set will most likely produce a higher effective dimension that the final set of trained weights, because that model with that particular parameterization is “using more parameters” unnecessarily to fit the data. After training (with the implicit regularization), a trained model will not need to use so many parameters and thus have more “inactive parameters” and a lower effective dimension. 

We must keep in mind though that this is the general intuition, and there might be cases where a randomly selected set of weights happens to provide a lower effective dimension than the trained weights for a specific model. 


```python
import qiskit.tools.jupyter

%qiskit_version_table
%qiskit_copyright
```


<h3>Version Information</h3><table><tr><th>Qiskit Software</th><th>Version</th></tr><tr><td><code>qiskit-terra</code></td><td>0.21.1</td></tr><tr><td><code>qiskit-aer</code></td><td>0.10.4</td></tr><tr><td><code>qiskit-ibmq-provider</code></td><td>0.19.2</td></tr><tr><td><code>qiskit-nature</code></td><td>0.5.0</td></tr><tr><td><code>qiskit-finance</code></td><td>0.4.0</td></tr><tr><td><code>qiskit-optimization</code></td><td>0.4.0</td></tr><tr><td><code>qiskit-machine-learning</code></td><td>0.5.0</td></tr><tr><th>System information</th></tr><tr><td>Python version</td><td>3.8.10</td></tr><tr><td>Python compiler</td><td>GCC 9.4.0</td></tr><tr><td>Python build</td><td>default, Jun 22 2022 20:18:18</td></tr><tr><td>OS</td><td>Linux</td></tr><tr><td>CPUs</td><td>12</td></tr><tr><td>Memory (Gb)</td><td>31.267112731933594</td></tr><tr><td colspan='2'>Sat Jul 30 14:36:07 2022 CEST</td></tr></table>



<div style='width: 100%; background-color:#d5d9e0;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'><h3>This code is a part of Qiskit</h3><p>&copy; Copyright IBM 2017, 2022.</p><p>This code is licensed under the Apache License, Version 2.0. You may<br>obtain a copy of this license in the LICENSE.txt file in the root directory<br> of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.<p>Any modifications or derivative works of this code must retain this<br>copyright notice, and modified files need to carry a notice indicating<br>that they have been altered from the originals.</p></div>

