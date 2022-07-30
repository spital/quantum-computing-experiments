# Converters for Quadratic Programs

Optimization problems in Qiskit's optimization module are represented with the `QuadraticProgram` class, which is a generic and powerful representation for optimization problems. In general, optimization algorithms are defined for a certain formulation of a quadratic program, and we need to convert our problem to the right type.

For instance, Qiskit provides several optimization algorithms that can handle [Quadratic Unconstrained Binary Optimization](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization) (QUBO) problems. These are mapped to Ising Hamiltonians, for which Qiskit uses the `qiskit.opflow` module, and then their ground state is approximated. For this optimization, commonly known algorithms such as VQE or QAOA can be used as underlying routine. See the following tutorial about the [Minimum Eigen Optimizer](./03_minimum_eigen_optimizer.ipynb) for more detail. Note that also other algorithms exist that work differently, such as the `GroverOptimizer`.

To map a problem to the correct input format, the optimization module of Qiskit offers a variety of converters. In this tutorial we're providing an overview on this functionality. Currently, Qiskit contains the following converters.

- `InequalityToEquality`: convert inequality constraints into equality constraints with additional slack variables.
- `IntegerToBinary`: convert integer variables into binary variables and corresponding coefficients.
- `LinearEqualityToPenalty`: convert equality constraints into additional terms of the objective function.
- `LinearInequalityToPenalty`: convert inequality constraints into additional terms of the objective function.
- `MaximizeToMinimize`: convert to the equivalent minimization problem.
- `MinimizeToMaximize`: convert to the equivalent maximization problem.
- `QuadraticProgramToQubo`: a wrapper that includes `InequalityToEquality`, `IntegerToBinary`, `LinearEqualityToPenalty`, `LinearInequalityToPenalty`, and `MaximizeToMinimize` for convenience.

## InequalityToEquality
`InequalityToEqualityConverter` converts inequality constraints into equality constraints with additional slack variables to remove inequality constraints from `QuadraticProgram`. The upper bounds and the lower bounds of slack variables will be calculated from the difference between the left sides and the right sides of constraints. Signs of slack variables depend on symbols in constraints such as $\leq$ and $\geq$.

The following is an example of a maximization problem with two inequality constraints. Variable $x$ and $y$ are binary variables and variable $z$ is an integer variable.

\begin{aligned}
   & \text{maximize}
       & 2x + y + z\\
   & \text{subject to:}
       & x+y+z \leq 5.5\\
       & & x+y+z \geq 2.5\\
       & & x, y \in \{0,1\}\\
       & & z \in \{0,1,2,3,4,5,6,7\} \\
\end{aligned}

With `QuadraticProgram`, an optimization model of the problem is written as follows.


```python
from qiskit_optimization import QuadraticProgram
```


```python
qp = QuadraticProgram()
qp.binary_var("x")
qp.binary_var("y")
qp.integer_var(lowerbound=0, upperbound=7, name="z")

qp.maximize(linear={"x": 2, "y": 1, "z": 1})
qp.linear_constraint(linear={"x": 1, "y": 1, "z": 1}, sense="LE", rhs=5.5, name="xyz_leq")
qp.linear_constraint(linear={"x": 1, "y": 1, "z": 1}, sense="GE", rhs=2.5, name="xyz_geq")
print(qp.prettyprint())
```

    Problem name: 
    
    Maximize
      2*x + y + z
    
    Subject to
      Linear constraints (2)
        x + y + z <= 5.5  'xyz_leq'
        x + y + z >= 2.5  'xyz_geq'
    
      Integer variables (1)
        0 <= z <= 7
    
      Binary variables (2)
        x y
    


Call `convert` method of `InequalityToEquality` to convert.


```python
from qiskit_optimization.converters import InequalityToEquality
```


```python
ineq2eq = InequalityToEquality()
qp_eq = ineq2eq.convert(qp)
print(qp_eq.prettyprint())
```

    Problem name: 
    
    Maximize
      2*x + y + z
    
    Subject to
      Linear constraints (2)
        x + xyz_leq@int_slack + y + z == 5  'xyz_leq'
        x - xyz_geq@int_slack + y + z == 3  'xyz_geq'
    
      Integer variables (3)
        0 <= z <= 7
        0 <= xyz_leq@int_slack <= 5
        0 <= xyz_geq@int_slack <= 6
    
      Binary variables (2)
        x y
    


After converting, the formulation of the problem looks like the above output. As we can see, the inequality constraints are replaced with equality constraints with additional integer slack variables, $xyz\_leg\text{@}int\_slack$ and $xyz\_geq\text{@}int\_slack$. 

Let us explain how the conversion works. For example, the lower bound of the left side of the first constraint is $0$ which is the case of $x=0$, $y=0$, and $z=0$. Thus, the upper bound of the additional integer variable must be $5$ to be able to satisfy even the case of $x=0$, $y=0$, and $z=0$. Note that we cut off the part after the decimal point in the converted formulation since the left side of the first constraint in the original formulation can be only integer values. For the second constraint, basically we apply the same approach. However, the symbol in the second constraint is $\geq$, so we add minus before $xyz\_geq\text{@}int\_slack$ to be able to satisfy even the case of $x=1, y=1$, and $z=7$.

\begin{aligned}
   & \text{maximize}
       & 2x + y + z\\
   & \text{subject to:}
       & x+y+z+ xyz\_leg\text{@}int\_slack= 5\\
       & & x+y+z+xyz\_geq\text{@}int\_slack= 3\\
       & & x, y \in \{0,1\}\\
       & & z \in \{0,1,2,3,4,5,6,7\} \\
       & & xyz\_leg\text{@}int\_slack \in \{0,1,2,3,4,5\} \\
       & & xyz\_geq\text{@}int\_slack \in \{0,1,2,3,4,5,6\} \\
\end{aligned}





## IntegerToBinary

`IntegerToBinary` converts integer variables into binary variables and coefficients to remove integer variables from `QuadraticProgram`. For converting, bounded-coefficient encoding proposed in [arxiv:1706.01945](https://arxiv.org/abs/1706.01945) (Eq. (5)) is used. For more detail of the encoding method, please see the paper.

We use the output of `InequalityToEquality` as a starting point. Variables $x$ and $y$ are binary variables, while the variable $z$ and the slack variables $xyz\_leq\text{@}int\_slack$ and $xyz\_geq\text{@}int\_slack$ are integer variables. We print the problem again for reference.


```python
print(qp_eq.prettyprint())
```

    Problem name: 
    
    Maximize
      2*x + y + z
    
    Subject to
      Linear constraints (2)
        x + xyz_leq@int_slack + y + z == 5  'xyz_leq'
        x - xyz_geq@int_slack + y + z == 3  'xyz_geq'
    
      Integer variables (3)
        0 <= z <= 7
        0 <= xyz_leq@int_slack <= 5
        0 <= xyz_geq@int_slack <= 6
    
      Binary variables (2)
        x y
    


Call `convert` method of `IntegerToBinary` to convert.


```python
from qiskit_optimization.converters import IntegerToBinary
```


```python
int2bin = IntegerToBinary()
qp_eq_bin = int2bin.convert(qp_eq)
print(qp_eq_bin.prettyprint())
```

    Problem name: 
    
    Maximize
      2*x + y + z@0 + 2*z@1 + 4*z@2
    
    Subject to
      Linear constraints (2)
        x + xyz_leq@int_slack@0 + 2*xyz_leq@int_slack@1 + 2*xyz_leq@int_slack@2 + y
        + z@0 + 2*z@1 + 4*z@2 == 5  'xyz_leq'
        x - xyz_geq@int_slack@0 - 2*xyz_geq@int_slack@1 - 3*xyz_geq@int_slack@2 + y
        + z@0 + 2*z@1 + 4*z@2 == 3  'xyz_geq'
    
      Binary variables (11)
        x y z@0 z@1 z@2 xyz_leq@int_slack@0 xyz_leq@int_slack@1 xyz_leq@int_slack@2
        xyz_geq@int_slack@0 xyz_geq@int_slack@1 xyz_geq@int_slack@2
    


After converting, the integer variable $z$ is replaced with three binary variables $z\text{@}0$, $z\text{@}1$ and $z\text{@}2$ with coefficients 1, 2 and 4, respectively as the above. 
The slack variables $xyz\_leq\text{@}int\_slack$ and $xyz\_geq\text{@}int\_slack$ that were introduced by `InequalityToEquality` are also both replaced with three binary variables with coefficients 1, 2, 2, and  1, 2, 3, respectively.

Note: Essentially the coefficients mean that the sum of these binary variables with coefficients can be the sum of a subset of $\{1, 2, 4\}$, $\{1, 2, 2\}$, and $\{1, 2, 3\}$ to represent that acceptable values $\{0, \ldots, 7\}$, $\{0, \ldots, 5\}$, and $\{0, \ldots, 6\}$, which respects the lower bound and the upper bound of original integer variables correctly.

`IntegerToBinary` also provides `interpret` method that is the functionality to translate a given binary result back to the original integer representation.

## LinearEqualityToPenalty

`LinearEqualityToPenalty` converts linear equality constraints into additional quadratic penalty terms of the objective function to map `QuadraticProgram` to an unconstrained form.
An input to the converter has to be a `QuadraticProgram` with only linear equality constraints. Those equality constraints, e.g. $\sum_i a_i x_i  = b$ where $a_i$ and $b$ are numbers and $x_i$ is a variable, will be added to the objective function in the form of $M(b - \sum_i a_i x_i)^2$ where $M$ is a large number as penalty factor. 
By default $M= 1e5$. The sign of the term depends on whether the problem type is a maximization or minimization.

We use the output of `IntegerToBinary` as a starting point, where all variables are binary variables and all inequality constraints have been mapped to equality constraints. 
We print the problem again for reference.


```python
print(qp_eq_bin.prettyprint())
```

    Problem name: 
    
    Maximize
      2*x + y + z@0 + 2*z@1 + 4*z@2
    
    Subject to
      Linear constraints (2)
        x + xyz_leq@int_slack@0 + 2*xyz_leq@int_slack@1 + 2*xyz_leq@int_slack@2 + y
        + z@0 + 2*z@1 + 4*z@2 == 5  'xyz_leq'
        x - xyz_geq@int_slack@0 - 2*xyz_geq@int_slack@1 - 3*xyz_geq@int_slack@2 + y
        + z@0 + 2*z@1 + 4*z@2 == 3  'xyz_geq'
    
      Binary variables (11)
        x y z@0 z@1 z@2 xyz_leq@int_slack@0 xyz_leq@int_slack@1 xyz_leq@int_slack@2
        xyz_geq@int_slack@0 xyz_geq@int_slack@1 xyz_geq@int_slack@2
    


Call `convert` method of `LinearEqualityToPenalty` to convert.


```python
from qiskit_optimization.converters import LinearEqualityToPenalty
```


```python
lineq2penalty = LinearEqualityToPenalty()
qubo = lineq2penalty.convert(qp_eq_bin)
print(qubo.prettyprint())
```

    Problem name: 
    
    Maximize
      -22*x^2 + 22*x*xyz_geq@int_slack@0 + 44*x*xyz_geq@int_slack@1
      + 66*x*xyz_geq@int_slack@2 - 22*x*xyz_leq@int_slack@0
      - 44*x*xyz_leq@int_slack@1 - 44*x*xyz_leq@int_slack@2 - 44*x*y - 44*x*z@0
      - 88*x*z@1 - 176*x*z@2 - 11*xyz_geq@int_slack@0^2
      - 44*xyz_geq@int_slack@0*xyz_geq@int_slack@1
      - 66*xyz_geq@int_slack@0*xyz_geq@int_slack@2 - 44*xyz_geq@int_slack@1^2
      - 132*xyz_geq@int_slack@1*xyz_geq@int_slack@2 - 99*xyz_geq@int_slack@2^2
      - 11*xyz_leq@int_slack@0^2 - 44*xyz_leq@int_slack@0*xyz_leq@int_slack@1
      - 44*xyz_leq@int_slack@0*xyz_leq@int_slack@2 - 44*xyz_leq@int_slack@1^2
      - 88*xyz_leq@int_slack@1*xyz_leq@int_slack@2 - 44*xyz_leq@int_slack@2^2
      + 22*y*xyz_geq@int_slack@0 + 44*y*xyz_geq@int_slack@1
      + 66*y*xyz_geq@int_slack@2 - 22*y*xyz_leq@int_slack@0
      - 44*y*xyz_leq@int_slack@1 - 44*y*xyz_leq@int_slack@2 - 22*y^2 - 44*y*z@0
      - 88*y*z@1 - 176*y*z@2 + 22*z@0*xyz_geq@int_slack@0
      + 44*z@0*xyz_geq@int_slack@1 + 66*z@0*xyz_geq@int_slack@2
      - 22*z@0*xyz_leq@int_slack@0 - 44*z@0*xyz_leq@int_slack@1
      - 44*z@0*xyz_leq@int_slack@2 - 22*z@0^2 - 88*z@0*z@1 - 176*z@0*z@2
      + 44*z@1*xyz_geq@int_slack@0 + 88*z@1*xyz_geq@int_slack@1
      + 132*z@1*xyz_geq@int_slack@2 - 44*z@1*xyz_leq@int_slack@0
      - 88*z@1*xyz_leq@int_slack@1 - 88*z@1*xyz_leq@int_slack@2 - 88*z@1^2
      - 352*z@1*z@2 + 88*z@2*xyz_geq@int_slack@0 + 176*z@2*xyz_geq@int_slack@1
      + 264*z@2*xyz_geq@int_slack@2 - 88*z@2*xyz_leq@int_slack@0
      - 176*z@2*xyz_leq@int_slack@1 - 176*z@2*xyz_leq@int_slack@2 - 352*z@2^2
      + 178*x - 66*xyz_geq@int_slack@0 - 132*xyz_geq@int_slack@1
      - 198*xyz_geq@int_slack@2 + 110*xyz_leq@int_slack@0 + 220*xyz_leq@int_slack@1
      + 220*xyz_leq@int_slack@2 + 177*y + 177*z@0 + 354*z@1 + 708*z@2 - 374
    
    Subject to
      No constraints
    
      Binary variables (11)
        x y z@0 z@1 z@2 xyz_leq@int_slack@0 xyz_leq@int_slack@1 xyz_leq@int_slack@2
        xyz_geq@int_slack@0 xyz_geq@int_slack@1 xyz_geq@int_slack@2
    


After converting, the equality constraints are added to the objective function as additional terms with the default penalty factor $M=1e5$.
The resulting problem is now a QUBO and compatible with many quantum optimization algorithms such as VQE, QAOA and so on.

This gives the same result as before.


```python
import qiskit.tools.jupyter

%qiskit_version_table
%qiskit_copyright
```


<h3>Version Information</h3><table><tr><th>Qiskit Software</th><th>Version</th></tr><tr><td><code>qiskit-terra</code></td><td>0.22.0</td></tr><tr><td><code>qiskit-aer</code></td><td>0.11.0</td></tr><tr><td><code>qiskit-ignis</code></td><td>0.7.0</td></tr><tr><td><code>qiskit</code></td><td>0.37.1</td></tr><tr><td><code>qiskit-nature</code></td><td>0.4.3</td></tr><tr><td><code>qiskit-finance</code></td><td>0.3.3</td></tr><tr><td><code>qiskit-optimization</code></td><td>0.5.0</td></tr><tr><td><code>qiskit-machine-learning</code></td><td>0.4.0</td></tr><tr><th>System information</th></tr><tr><td>Python version</td><td>3.8.13</td></tr><tr><td>Python compiler</td><td>GCC 10.3.0</td></tr><tr><td>Python build</td><td>default, Mar 25 2022 06:04:10</td></tr><tr><td>OS</td><td>Linux</td></tr><tr><td>CPUs</td><td>12</td></tr><tr><td>Memory (Gb)</td><td>31.267112731933594</td></tr><tr><td colspan='2'>Fri Jul 29 19:52:41 2022 UTC</td></tr></table>



<div style='width: 100%; background-color:#d5d9e0;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'><h3>This code is a part of Qiskit</h3><p>&copy; Copyright IBM 2017, 2022.</p><p>This code is licensed under the Apache License, Version 2.0. You may<br>obtain a copy of this license in the LICENSE.txt file in the root directory<br> of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.<p>Any modifications or derivative works of this code must retain this<br>copyright notice, and modified files need to carry a notice indicating<br>that they have been altered from the originals.</p></div>



```python

```
