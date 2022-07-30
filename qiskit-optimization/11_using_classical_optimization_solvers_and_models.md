# Using Classical Optimization Solvers and Models with Qiskit Optimization

We can use classical optimization solvers (CPLEX and Gurobi) with Qiskit Optimization.
Docplex and Gurobipy are the Python APIs for CPLEX and Gurobi, respectively.
We can load and save an optimization model by Docplex and Gurobipy and can apply CPLEX and Gurobi to `QuadraticProgram`.

If you want to use the CPLEX solver, you need to install `pip install 'qiskit-optimization[cplex]'`. Docplex is automatically installed, as a dependent, when you install Qiskit Optimization.

If you want to use Gurobi and Gurobipy, you need to install `pip install 'qiskit-optimization[gurobi]'`.

Note that these solvers installed via pip are free versions, which have some limitations such as number of variables. Also note that the latest version 20.1 of CPLEX is available only for Python 3.7 and 3.8 as of July 2021. See the following pages for details.

- https://pypi.org/project/cplex/
- https://pypi.org/project/gurobipy/

## CplexSolver and GurobiSolver

Qiskit Optimization supports the classical solvers of CPLEX and Gurobi as `CplexSolver` and `GurobiSolver`, respectively.
We can solve `QuadraticProgram` with `CplexSolver` and `GurobiSolver` as follows.


```python
from qiskit_optimization.problems import QuadraticProgram

# define a problem
qp = QuadraticProgram()
qp.binary_var("x")
qp.integer_var(name="y", lowerbound=-1, upperbound=4)
qp.maximize(quadratic={("x", "y"): 1})
qp.linear_constraint({"x": 1, "y": -1}, "<=", 0)
print(qp.prettyprint())
```

    Problem name: 
    
    Maximize
      x*y
    
    Subject to
      Linear constraints (1)
        x - y <= 0  'c0'
    
      Integer variables (1)
        -1 <= y <= 4
    
      Binary variables (1)
        x
    



```python
from qiskit_optimization.algorithms import CplexOptimizer, GurobiOptimizer

cplex_result = CplexOptimizer().solve(qp)
gurobi_result = GurobiOptimizer().solve(qp)

print("cplex")
print(cplex_result.prettyprint())
print()
print("gurobi")
print(gurobi_result.prettyprint())
```

    Restricted license - for non-production use only - expires 2023-10-25
    cplex
    objective function value: 4.0
    variable values: x=1.0, y=4.0
    status: SUCCESS
    
    gurobi
    objective function value: 4.0
    variable values: x=1.0, y=4.0
    status: SUCCESS


We can set the solver parameter of CPLEX as follows. We can display the solver message of CPLEX by setting `disp=True`.
See [Parameters of CPLEX](https://www.ibm.com/docs/en/icos/20.1.0?topic=cplex-parameters) for details of CPLEX parameters.


```python
result = CplexOptimizer(disp=True, cplex_parameters={"threads": 1, "timelimit": 0.1}).solve(qp)
print(result.prettyprint())
```

    Version identifier: 22.1.0.0 | 2022-03-25 | 54982fbec
    CPXPARAM_Read_DataCheck                          1
    CPXPARAM_Threads                                 1
    CPXPARAM_TimeLimit                               0.10000000000000001
    Found incumbent of value 0.000000 after 0.00 sec. (0.00 ticks)
    Found incumbent of value 4.000000 after 0.00 sec. (0.00 ticks)
    
    Root node processing (before b&c):
      Real time             =    0.00 sec. (0.00 ticks)
    Sequential b&c:
      Real time             =    0.00 sec. (0.00 ticks)
                              ------------
    Total (root+branch&cut) =    0.00 sec. (0.00 ticks)
    objective function value: 4.0
    variable values: x=1.0, y=4.0
    status: SUCCESS


We get the same optimal solution by QAOA as follows.


```python
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA

qins = QuantumInstance(backend=Aer.get_backend("aer_simulator"), shots=1000)
meo = MinimumEigenOptimizer(QAOA(COBYLA(maxiter=100), quantum_instance=qins))
result = meo.solve(qp)
print(result.prettyprint())
print("\ndisplay the best 5 solution samples")
for sample in result.samples[:5]:
    print(sample)
```

    objective function value: 4.0
    variable values: x=1.0, y=4.0
    status: SUCCESS
    
    display the best 5 solution samples
    SolutionSample(x=array([1., 4.]), fval=4.0, probability=0.025, status=<OptimizationResultStatus.SUCCESS: 0>)
    SolutionSample(x=array([1., 3.]), fval=3.0, probability=0.058, status=<OptimizationResultStatus.SUCCESS: 0>)
    SolutionSample(x=array([1., 2.]), fval=2.0, probability=0.107, status=<OptimizationResultStatus.SUCCESS: 0>)
    SolutionSample(x=array([1., 1.]), fval=1.0, probability=0.204, status=<OptimizationResultStatus.SUCCESS: 0>)
    SolutionSample(x=array([0., 0.]), fval=0.0, probability=0.083, status=<OptimizationResultStatus.SUCCESS: 0>)


## Translators between `QuadraticProgram` and Docplex/Gurobipy

Qiskit Optimization can load `QuadraticProgram` from a Docplex model and a Gurobipy model.

First, we define an optimization problem by Docplex and Gurobipy.


```python
# docplex model
from docplex.mp.model import Model

docplex_model = Model("docplex")
x = docplex_model.binary_var("x")
y = docplex_model.integer_var(-1, 4, "y")
docplex_model.maximize(x * y)
docplex_model.add_constraint(x <= y)
docplex_model.prettyprint()
```

    // This file has been generated by DOcplex
    // model name is: docplex
    // single vars section
    dvar bool x;
    dvar int y;
    
    maximize
     [ x*y ];
     
    subject to {
     x <= y;
    
    }



```python
# gurobi model
import gurobipy as gp

gurobipy_model = gp.Model("gurobi")
x = gurobipy_model.addVar(vtype=gp.GRB.BINARY, name="x")
y = gurobipy_model.addVar(vtype=gp.GRB.INTEGER, lb=-1, ub=4, name="y")
gurobipy_model.setObjective(x * y, gp.GRB.MAXIMIZE)
gurobipy_model.addConstr(x - y <= 0)
gurobipy_model.update()
gurobipy_model.display()
```

    Maximize
      <gurobi.QuadExpr: 0.0 + [ x * y ]>
    Subject To
      R0: <gurobi.LinExpr: x + -1.0 y> <= 0
    Bounds
      -1 <= y <= 4
    Binaries
      ['x']
    General Integers
      ['y']


We can generate `QuadraticProgram` object from both Docplex and Gurobipy models. We see that the two `QuadraticProgram` objects generated from Docplex and Gurobipy are identical.


```python
from qiskit_optimization.translators import from_docplex_mp, from_gurobipy

qp = from_docplex_mp(docplex_model)
print("QuadraticProgram obtained from docpblex")
print(qp.prettyprint())
print("-------------")
print("QuadraticProgram obtained from gurobipy")
qp2 = from_gurobipy(gurobipy_model)
print(qp2.prettyprint())
```

    QuadraticProgram obtained from docpblex
    Problem name: docplex
    
    Maximize
      x*y
    
    Subject to
      Linear constraints (1)
        x - y <= 0  'c0'
    
      Integer variables (1)
        -1 <= y <= 4
    
      Binary variables (1)
        x
    
    -------------
    QuadraticProgram obtained from gurobipy
    Problem name: gurobi
    
    Maximize
      x*y
    
    Subject to
      Linear constraints (1)
        x - y <= 0  'R0'
    
      Integer variables (1)
        -1 <= y <= 4
    
      Binary variables (1)
        x
    


We can generate a Docplex model and a Gurobipy model from `QuadraticProgram` too.


```python
from qiskit_optimization.translators import to_gurobipy, to_docplex_mp

gmod = to_gurobipy(from_docplex_mp(docplex_model))
print("convert docplex to gurobipy via QuadraticProgram")
gmod.display()

dmod = to_docplex_mp(from_gurobipy(gurobipy_model))
print("\nconvert gurobipy to docplex via QuadraticProgram")
print(dmod.export_as_lp_string())
```

    convert docplex to gurobipy via QuadraticProgram
    Maximize
      <gurobi.QuadExpr: 0.0 + [ x * y ]>
    Subject To
      c0: <gurobi.LinExpr: x + -1.0 y> <= 0
    Bounds
      -1 <= y <= 4
    Binaries
      ['x']
    General Integers
      ['y']
    
    convert gurobipy to docplex via QuadraticProgram
    \ This file has been generated by DOcplex
    \ ENCODING=ISO-8859-1
    \Problem name: gurobi
    
    Maximize
     obj: [ 2 x*y ]/2
    Subject To
     R0: x - y <= 0
    
    Bounds
     0 <= x <= 1
     -1 <= y <= 4
    
    Binaries
     x
    
    Generals
     y
    End
    


### Indicator constraints of Docplex

`from_docplex_mp` supports indicator constraints, e.g., `u = 0 => x + y <= z` (u: binary variable) when we convert a Docplex model into `QuadraticProgram`. It converts indicator constraints into linear constraints using the big-M formulation.


```python
ind_mod = Model("docplex")
x = ind_mod.binary_var("x")
y = ind_mod.integer_var(-1, 2, "y")
z = ind_mod.integer_var(-1, 2, "z")
ind_mod.maximize(3 * x + y - z)
ind_mod.add_indicator(x, y >= z, 1)
print(ind_mod.export_as_lp_string())
```

    \ This file has been generated by DOcplex
    \ ENCODING=ISO-8859-1
    \Problem name: docplex
    
    Maximize
     obj: 3 x + y - z
    Subject To
     lc1: x = 1 -> y - z >= 0
    
    Bounds
     0 <= x <= 1
     -1 <= y <= 2
     -1 <= z <= 2
    
    Binaries
     x
    
    Generals
     y z
    End
    


Let's compare the solutions of the model with an indicator constraint by

1. applying CPLEX directly to the Docplex model (without translating it to `QuadraticProgram`. CPLEX solver natively supports the indicator constraints),
2. applying QAOA to `QuadraticProgram` obtained by `from_docplex_mp`.

We see the solutions are same.


```python
qp = from_docplex_mp(ind_mod)
result = meo.solve(qp)  # apply QAOA to QuadraticProgram
print("QAOA")
print(result.prettyprint())
print("-----\nCPLEX")
print(ind_mod.solve())  # apply CPLEX directly to the Docplex model
```

    QAOA
    objective function value: 6.0
    variable values: x=1.0, y=2.0, z=-1.0
    status: SUCCESS
    -----
    CPLEX
    solution for: docplex
    objective: 6
    x=1
    y=2
    z=-1
    



```python
import qiskit.tools.jupyter

%qiskit_version_table
%qiskit_copyright
```


<h3>Version Information</h3><table><tr><th>Qiskit Software</th><th>Version</th></tr><tr><td><code>qiskit-terra</code></td><td>0.22.0</td></tr><tr><td><code>qiskit-aer</code></td><td>0.11.0</td></tr><tr><td><code>qiskit-ignis</code></td><td>0.7.0</td></tr><tr><td><code>qiskit</code></td><td>0.37.1</td></tr><tr><td><code>qiskit-nature</code></td><td>0.4.3</td></tr><tr><td><code>qiskit-finance</code></td><td>0.3.3</td></tr><tr><td><code>qiskit-optimization</code></td><td>0.5.0</td></tr><tr><td><code>qiskit-machine-learning</code></td><td>0.4.0</td></tr><tr><th>System information</th></tr><tr><td>Python version</td><td>3.8.13</td></tr><tr><td>Python compiler</td><td>GCC 10.3.0</td></tr><tr><td>Python build</td><td>default, Mar 25 2022 06:04:10</td></tr><tr><td>OS</td><td>Linux</td></tr><tr><td>CPUs</td><td>12</td></tr><tr><td>Memory (Gb)</td><td>31.267112731933594</td></tr><tr><td colspan='2'>Fri Jul 29 20:04:18 2022 UTC</td></tr></table>



<div style='width: 100%; background-color:#d5d9e0;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'><h3>This code is a part of Qiskit</h3><p>&copy; Copyright IBM 2017, 2022.</p><p>This code is licensed under the Apache License, Version 2.0. You may<br>obtain a copy of this license in the LICENSE.txt file in the root directory<br> of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.<p>Any modifications or derivative works of this code must retain this<br>copyright notice, and modified files need to carry a notice indicating<br>that they have been altered from the originals.</p></div>

