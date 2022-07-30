# Torch Runtime

In this tutorial, we introduce Torch Runtime, and show how to use it via the `TorchRuntimeClient` class in Qiskit Machine Learning.
Torch Runtime leverages Qiskit Runtime for 
hybrid quantum-classical machine learning based on a PyTorch `Module`. It allows training models or predicting the outputs with trained models significantly faster. We show how to use Torch Runtime with two simple examples for regression and classification tasks in the following.

## 1. Regression

First, we show how to use Torch Runtime via `TorchRuntimeClient` using the simple regression example. In the example, we will perform 
a regression task on a randomly generated dataset following a sine wave.


```python
import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor, manual_seed, is_tensor
from torch.nn import Linear, CrossEntropyLoss, MSELoss
from torch.optim import LBFGS, Adam
from torch.utils.data import Dataset, DataLoader

from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow import AerPauliExpectation
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.runtime import TorchRuntimeClient, TorchRuntimeResult


# Set seed for random generators
seed = 42
manual_seed(seed)
algorithm_globals.random_seed = seed
```

    /usr/local/lib/python3.8/dist-packages/torch/cuda/__init__.py:83: UserWarning: CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero. (Triggered internally at  ../c10/cuda/CUDAFunctions.cpp:109.)
      return torch._C._cuda_getDeviceCount() > 0



```python
# Generate random dataset for the training
import matplotlib.pyplot as plt

np.random.seed(0)
num_samples = 20
eps = 0.2
lb, ub = -np.pi, np.pi
f = lambda x: np.sin(x)
X = (ub - lb) * np.random.rand(num_samples, 1) + lb
y = f(X) + eps * (2 * np.random.rand(num_samples, 1) - 1)

plt.figure()
plt.plot(np.linspace(lb, ub), f(np.linspace(lb, ub)), "r--")
plt.plot(X, y, "bo")
plt.show()
```


    
![png](06_torch_runtime_files/06_torch_runtime_3_0.png)
    


`TorchRuntimeClient` requires a PyTorch `DataLoader` as an input for training/predicting. For that purpose, we create a custom torch dataset class.


```python
# Create custom torch dataset class
class TorchDataset(Dataset):
    """Map-style dataset"""

    def __init__(self, X, y):
        self.X = Tensor(X).float()
        self.y = Tensor(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        import torch

        if torch.is_tensor(idx):
            idx = idx.tolist()

        X_i = self.X[idx]
        y_i = self.y[idx]

        # important: the dataset item must be returned as data,target
        return X_i, y_i


# Create a train loader
train_set = TorchDataset(X, y)
train_loader1 = DataLoader(train_set, batch_size=1, shuffle=False)
```

Create an instance of `TorchConnector` to wrap a QNN model and be able to use pytorch to train the model, then set up an optimizer and a loss function as usual.


```python
from qiskit.circuit import Parameter

# Construct simple feature map
param_x = Parameter("x")
feature_map = QuantumCircuit(1, name="fm")
feature_map.ry(param_x, 0)

# Construct simple feature map
param_y = Parameter("y")
ansatz = QuantumCircuit(1, name="vf")
ansatz.ry(param_y, 0)

# Construct QNN
qnn1 = TwoLayerQNN(1, feature_map, ansatz)
print(qnn1.operator)


initial_weights = 0.1 * (2 * algorithm_globals.random.random(qnn1.num_weights) - 1)
model1 = TorchConnector(qnn1, initial_weights)

# Define optimizer and loss function
optimizer1 = Adam(model1.parameters(), lr=0.1)
loss_func1 = MSELoss(reduction="sum")
```

    ComposedOp([
      OperatorMeasurement(1.0 * Z),
      CircuitStateFn(
         ┌───────┐┌───────┐
      q: ┤ fm(x) ├┤ vf(y) ├
         └───────┘└───────┘
      )
    ])


Load a provider and specify a backend for the runtime service. 


```python
# Set up a provider and backend
from qiskit import IBMQ

IBMQ.load_account()
provider = IBMQ.get_provider(project="main")  # replace by your runtime provider

backend = provider.get_backend("ibmq_qasm_simulator")  # select a backend that supports the runtime

```

Create a Torch Runtime Client instance with the model, the optimizer, and other configurations.


```python
torch_runtime_client = TorchRuntimeClient(
    provider=provider,
    model=model1,
    optimizer=optimizer1,
    loss_func=loss_func1,
    epochs=5,
    backend=backend,
)
```

### Call `fit()` to train the model
Call the `fit` method in `TorchRuntimeClient` with the data loader to train the model.


```python
fit_result = torch_runtime_client.fit(train_loader=train_loader1)
```

You can access training result information by querying properties of the `fit_result` variable, that is an instance of the `TorchRuntimeResult` class. Also, model's parameters in `TorchRuntimeClient` are updated with trained ones. 


```python
print("id: ", fit_result.job_id)
print("execution time: ", fit_result.execution_time)
print("model_state_dict: ", torch_runtime_client.model.state_dict())
```

    id:  cbimdmbsp2ol7h7jh6ag
    execution time:  2.2879841327667236
    model_state_dict:  OrderedDict([('weight', tensor([-1.5736])), ('_weights', tensor([-1.5736]))])


You can also query the `train_history` property, which is a list of dictionaries, each per epoch. In a dictionary you can find properties like: 
- `epoch`, epoch index
- `loss`, loss value at this epoch
- `forward_time`, time spent in the forward pass, in seconds
- `backward_time`, time spent in the backward pass, in seconds
- `epoch_time`, epoch time, in seconds."

#### Training with validation
Torch Runtime can also perform validation while training a model by passing a validation data loader to the `fit` method.


```python
# Create a validation dataloader
X_test = [[x] for x in np.linspace(lb, ub)]
y_test = [[y] for y in f(np.linspace(lb, ub))]
test_set = TorchDataset(X_test, y_test)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
```


```python
torch_runtime_client = TorchRuntimeClient(
    provider=provider,
    model=model1,
    optimizer=optimizer1,
    loss_func=loss_func1,
    epochs=5,
    backend=backend,
)
```


```python
# Pass a train data loader and a validation data loader
fit_result = torch_runtime_client.fit(train_loader=train_loader1, val_loader=test_loader)
```


```python
print("id: ", fit_result.job_id)
print("execution time: ", fit_result.execution_time)
```

    id:  cbimdqm5venaagr8h090
    execution time:  3.8626763820648193


You can query the `val_history` property, which is a list of dictionaries for the validation processes, each per epoch. In a dictionary you can find the same properties as `train_history`.

### Call `predict()` to perform prediction
Call the `predict` method in `TorchRuntimeClient` with the data loader to perform prediction on the passed data using the trained model.


```python
predict_result = torch_runtime_client.predict(data_loader=test_loader)
```


```python
print("id: ", predict_result.job_id)
print("execution time: ", predict_result.execution_time)
```

    id:  cbime07ltu41v8vk6nr0
    execution time:  0.40851449966430664



```python
# Plot the original function
plt.plot(np.linspace(lb, ub), f(np.linspace(lb, ub)), "r--")

# Plot the training data
plt.plot(X, y, "bo")
# # Plot the prediction result
y_ = []
for output in predict_result.prediction:
    y_.append(output.item())
plt.plot(X_test, y_, "g-")

plt.show()
```


    
![png](06_torch_runtime_files/06_torch_runtime_26_0.png)
    


A red line, blue dots, and a green line on the plot show the original function, the training data, 
and a function constructed out of the predicted values, respectively. 

### Call `score()` to calculate a score
Call the `score` method in `TorchRuntimeClient` with the data loader to calculate a score,
for the trained model. You should pass either `"regression\"` or `"classification"` to the `score_func` argument to use one of the pre-defined scores functions. Also, you can pass a custom scoring function defined as:

```
def score_func(model_output, target): -> score: float\n",
```
where:
- `model_output` are the values predicted by the model,
- `target` ground truth values.

Note that the result of the `score` call also contains predicted values that were calculated in the process of scoring the model.


```python
score_result = torch_runtime_client.score(data_loader=test_loader, score_func="regression")
```


```python
print("id: ", score_result.job_id)
print("execution time: ", score_result.execution_time)
print("score: ", score_result.score)
```

    id:  cbime465venaagr8h0l0
    execution time:  0.3851451873779297
    score:  0.0015488592195367801


## 2. Classification

Second, we show how to perform a simple classification task using Torch Runtime. In the example, we will perform binary classification on a randomly generated dataset.


```python
# Generate random dataset

# Select dataset dimension (num_inputs) and size (num_samples)
num_inputs = 2
num_samples = 20

# Generate random input coordinates (X) and binary labels (y)
X = 2 * algorithm_globals.random.random([num_samples, num_inputs]) - 1
y01 = 1 * (np.sum(X, axis=1) >= 0)  # in { 0,  1}, y01 will be used for CircuitQNN example
y = 2 * y01 - 1  # in {-1, +1}, y will be used for OplowQNN example

# Convert to torch Tensors
X_ = Tensor(X)
y01_ = Tensor(y01).reshape(len(y)).long()
y_ = Tensor(y).reshape(len(y), 1)

# Plot dataset
for x, y_target in zip(X, y):
    if y_target == 1:
        plt.plot(x[0], x[1], "bo")
    else:
        plt.plot(x[0], x[1], "go")
plt.plot([-1, 1], [1, -1], "--", color="black")
plt.show()
```


    
![png](06_torch_runtime_files/06_torch_runtime_32_0.png)
    



```python
# Create custom torch dataset class
class TorchDataset(Dataset):
    """Map-style dataset"""

    def __init__(self, X, y):
        self.X = Tensor(X)
        self.y = Tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        import torch

        if torch.is_tensor(idx):
            idx = idx.tolist()

        X_i = self.X[idx]
        y_i = self.y[idx]

        # important: the dataset item must be returned as data,target
        return X_i, y_i
```


```python
y = y.reshape(20, 1)
train_set = TorchDataset(X, y)
train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
```


```python
# Set up QNN
qnn = TwoLayerQNN(num_qubits=num_inputs)
print(qnn.operator)

# Set up PyTorch module
initial_weights = 0.1 * (2 * algorithm_globals.random.random(qnn.num_weights) - 1)
model = TorchConnector(qnn, initial_weights=initial_weights)
print("Initial weights: ", initial_weights)
```

    ComposedOp([
      OperatorMeasurement(1.0 * ZZ),
      CircuitStateFn(
           ┌──────────────────────────┐»
      q_0: ┤0                         ├»
           │  ZZFeatureMap(x[0],x[1]) │»
      q_1: ┤1                         ├»
           └──────────────────────────┘»
      «     ┌──────────────────────────────────────────────────────────┐
      «q_0: ┤0                                                         ├
      «     │  RealAmplitudes(θ[0],θ[1],θ[2],θ[3],θ[4],θ[5],θ[6],θ[7]) │
      «q_1: ┤1                                                         ├
      «     └──────────────────────────────────────────────────────────┘
      )
    ])
    Initial weights:  [ 0.00020896 -0.06953758  0.03926408 -0.01076874 -0.02379575 -0.03969758
      0.02605652 -0.02763748]



```python
# Define optimizer and loss
optimizer = Adam(model.parameters(), lr=0.1)
loss_func = MSELoss(reduction="sum")
```


```python
torch_runtime_client = TorchRuntimeClient(
    provider=provider,
    model=model,
    optimizer=optimizer,
    loss_func=loss_func,
    epochs=5,
    backend=backend,
)
```

### Call `fit()` to train the model


```python
fit_result = torch_runtime_client.fit(train_loader=train_loader, seed=42)
```


```python
print("id: ", fit_result.job_id)
print("execution time: ", fit_result.execution_time)
```

    id:  cbime8fltu41v8vk6o40
    execution time:  8.652519941329956


You can also query the `train_history` and the `val_history` properties.

### Call `predict()` to perform prediction


```python
# In this example, we use the same data loader for the prediction as well
predict_result = torch_runtime_client.predict(data_loader=train_loader)
```


```python
print("id: ", predict_result.job_id)
print("execution time: ", predict_result.execution_time)
```

    id:  cbimee3sp2ol7h7jh74g
    execution time:  0.26399731636047363



```python
# Plot results
# red == wrongly classified

y_predict = []
for out in predict_result.prediction:
    y_predict += [np.sign(out.item())]
y_predict = np.array(y_predict)
y_check = [i[0] for i in y]

for x, y_target, y_p in zip(X, y, y_predict):
    if y_target == 1:
        plt.plot(x[0], x[1], "bo")
    else:
        plt.plot(x[0], x[1], "go")
    if y_target != y_p:
        plt.scatter(x[0], x[1], s=200, facecolors="none", edgecolors="r", linewidths=2)
plt.plot([-1, 1], [1, -1], "--", color="black")
plt.show()
```


    
![png](06_torch_runtime_files/06_torch_runtime_45_0.png)
    


The red circles indicate wrongly classified data points.

### Call `score()` to calculate a score

In the example, we use the following custom scoring function to calculate a score. The scoring function returns 1, if the trained model successfully classified the input. Otherwise, it returns 0. An overall average is calculated in `score()` in the end.


```python
def score_func(out, target):
    from numpy import sign

    score = 0
    if sign(out.item()) == target.item():
        score = 1
    return score
```


```python
score_result = torch_runtime_client.score(data_loader=train_loader, score_func=score_func)
```


```python
print("id: ", score_result.job_id)
print("execution time: ", score_result.execution_time)
print("score: ", score_result.score)
```

    id:  cbimeinltu41v8vk6ogg
    execution time:  0.2289142608642578
    score:  0.8


## 3. How to use hooks in training
Qiskit Machine Learning offers a base hook class, `HookBase`. It is a base class for a hook that is a set of callback functions used in the training process. Users can implement their own hook classes from this base class to handle complicated callback processes. This structure provides high flexibility in the callback processes.
Each hook can implement 6 methods, and each method is called before/after the corresponding processes during training. The way they are called is demonstrated
in the following snippet:
```
    hook.before_train()
    for epoch in range(epochs):
        hook.before_epoch()
        for batch in train_loader:
            hook.before_step()
            train_step()
            hook.after_step()
            global_step += 1
        hook.after_epoch()
        epoch += 1
    hook.after_train()
```


 In the hook methods, users can access `TorchTrainer` via `self.trainer` to access more context properties(e.g., model, current iteration, or config). The following snippet describes available properties that might be useful for a hook.

- TorchTrainer has:
  - `model`: A model to be trained.
  - `optimizer`: An optimizer used for the training.
  - `loss_func`: A loss function for the training.
  - `train_loader`: A PyTorch data loader object containing a training dataset.
  - `val_loader`: A PyTorch data loader object containing a validation dataset.
  - `max_epoch`: The maximum number of training epochs.
  - `log_period`: A logging period for a train history and a validation history. By default, there will be logs every epoch (`log_period`=1).
  - `start_epoch`: An initial epoch for warm-start training. By default, 0.
  - `epoch`: The current number of epochs.
  - `global_step`: The current number of steps.
  - `train_logger`: A logger for a training history. Use `train_logger.metrics` to access a list of logs. A log for each epoch is stored as a dictionary similar to `TorchRuntimeResult.train_history`
  - `val_logger`: A logger for a validation history. Logs are stored in the same way as train_logger.

 Users can pass a single hook instance or a list of hook instances to `hooks` argument.


The following example is a hook for early stopping. If the current loss is smaller than the threshold after an epoch, the training will be terminated. 


```python
from qiskit_machine_learning.runtime import HookBase


class StopHook(HookBase):
    """For early stopping"""

    def __init__(self, loss_threshold):
        self._loss_threshold = loss_threshold

    def after_epoch(self):
        # This function is called after each epoch
        current_loss = self.trainer.train_logger.metrics[-1]["loss"]
        print("current loss: ", current_loss)
        # If current loss is smaller than the threshold,
        # set the current number of the epoch to the maximum number of the epochs to stop the training
        if current_loss < self._loss_threshold:
            self.trainer.epoch = self.trainer.max_epochs


stop_hook = StopHook(loss_threshold=0.05)
```


```python
torch_runtime_client = TorchRuntimeClient(
    provider=provider,
    model=model1,
    optimizer=optimizer1,
    loss_func=loss_func1,
    epochs=5,
    backend=backend,
)
```


```python
fit_result = torch_runtime_client.fit(train_loader=train_loader1, hooks=stop_hook)
```


```python
print("id: ", fit_result.job_id)
print("execution time: ", fit_result.execution_time)
print("train history: ")
for epoch_result in fit_result.train_history:
    print("  epoch", epoch_result["epoch"], ": loss", epoch_result["loss"])
```

    id:  cbimemu5venaagr8h1cg
    execution time:  0.4658372402191162
    train history: 
      epoch 0 : loss 0.03142204274699907


As we can see, training was interrupted after the first epoch despite we set the number of epochs to 5 in the `TorchRuntimeClient` configuration.


```python
import qiskit.tools.jupyter

%qiskit_version_table
%qiskit_copyright
```


<h3>Version Information</h3><table><tr><th>Qiskit Software</th><th>Version</th></tr><tr><td><code>qiskit-terra</code></td><td>0.21.1</td></tr><tr><td><code>qiskit-aer</code></td><td>0.10.4</td></tr><tr><td><code>qiskit-ibmq-provider</code></td><td>0.19.2</td></tr><tr><td><code>qiskit-nature</code></td><td>0.5.0</td></tr><tr><td><code>qiskit-finance</code></td><td>0.4.0</td></tr><tr><td><code>qiskit-optimization</code></td><td>0.4.0</td></tr><tr><td><code>qiskit-machine-learning</code></td><td>0.5.0</td></tr><tr><th>System information</th></tr><tr><td>Python version</td><td>3.8.10</td></tr><tr><td>Python compiler</td><td>GCC 9.4.0</td></tr><tr><td>Python build</td><td>default, Jun 22 2022 20:18:18</td></tr><tr><td>OS</td><td>Linux</td></tr><tr><td>CPUs</td><td>12</td></tr><tr><td>Memory (Gb)</td><td>31.267112731933594</td></tr><tr><td colspan='2'>Sat Jul 30 19:16:27 2022 CEST</td></tr></table>



<div style='width: 100%; background-color:#d5d9e0;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'><h3>This code is a part of Qiskit</h3><p>&copy; Copyright IBM 2017, 2022.</p><p>This code is licensed under the Apache License, Version 2.0. You may<br>obtain a copy of this license in the LICENSE.txt file in the root directory<br> of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.<p>Any modifications or derivative works of this code must retain this<br>copyright notice, and modified files need to carry a notice indicating<br>that they have been altered from the originals.</p></div>



```python

```
