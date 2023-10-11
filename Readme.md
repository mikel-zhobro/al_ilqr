# AL-iLQR
( AL-iLQR) Augmented Lagrangian - Iterative Linear Quadratic Regulator is a planning algorithm which can be used to solve optimal control problems with constraints. It is an iterative algorithm which uses the iLQR algorithm to solve the unconstrained problem and the Augmented Lagrangian method to solve the constrained problem.
The implementation is based on the paper [Bjack. et. al](https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf)

## Installation
``` bash
git clone git@gitlab.localnet:embodied-vision/mikel-zhobro/planning-with-differentiable-physics/al_ilqr.git
cd al_ilqr

# create virtual environment
python3 -m venv al_ilqr_env && source al_ilqr_env/bin/activate

# install dependencies
pip install .
```


## Get started
An introduction script can be found in `test_constrained_ilqr.py`. It is recommended to start there as it is a good example of how to use the package.
There we solve a constrained optimal control problem for the Inverted-Pendulum problem both with analytical and numerical gradients and assert the gradients to be equal.

``` bash
python test_constrained_ilqr.py
```

## Organization
```
README.md                                           # this file
setup.py                                            # setup file for the package

test_state.py                                       # script for testing the state class
test_constrained_ilqr.py                            # script for testing the constrained ilqr class

al_ilqr                                             # package containing the code
│   ├── al_ilqr.py                                  # implements Al-ILQR according to [1](https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf)
│   ├── config.py                                   # Holds the configuration dicts for iLQR and the AL (Augmented Lagrangian) classes
│   ├── constraints.py                              # Implements the constraints class (Inequality/Equality)
│   ├── dynamic_system.py                           # Implements the dynamic system abstractions (DynamicSystem, ILQRDynamicSystem, ILQRDynamicSystemClosedLoop and Controller)
│   ├── helpers.py                                  # Implements helper functions for the package (to setup the AL-ILQR planner)
│   ├── loss.py                                     # Implements the loss class (how to take its 1.st and 2.nd derivatives depending on the situation)
└── └── utils.py
```

## Explanation of the abstractions

## 1. State
State is designed as an abstract class `BaseState` which can be adjusted for different dynamic systems. It is used to store the state of the system and to calculate the derivatives of the state with respect to the control input and the previous state.
In the file [state.py](al_ilqr/state.py) we provide 3 different implementations of the state class: `BasicState`, `StackedState`, `MultiBodyState`.
- `BasicState`: is the most basic case where the whole state can be represented as a single vector/tensor.
- `StackedState`: represent a generic implementation where we can stack multiple states together to form a single state. Used for example to include the controller state in the state of the system.
- `MultiBodyState`: is a special case of `BaseState` where the state is composed of `SO3+position+veocities` states of multiple bodies. As the state can not be written as a single tensor for which we can take derivatices
the Lie-Algebra rules of taking derivatives have to be respected.

To implement a new state class the methods to be defined are:
1. 'size': size property
2. '_set_din': set the delta_input w.r.t each the derivative is taken (e.g. Lie-Algebra of the state for elements of Lie-Group)
3. 'detach': detach the state from the computational graph
4. 'clone': clone the state
5. 'd_out': get the derivative of the state w.r.t. the input
6. 'increment': increment the state by the given delta
7. 'diff': get the difference between two states (e.g. is a Lie-Algebra element for Lie-Group elements)
8.

## 2. DynamicSystem

The `DynamicSystem` class in [dynamic_system.py](al_ilqr/dynamic_system.py) is an abstract class which can be used to implement different dynamic systems. To implement a new dynamic system the steps are:
1. Implement a new state class which inherits from `BaseState`
2. Implement a new dynamic system class which inherits from `DynamicSystem` and implements the `step_abs` method.


The `BaseILQRDynSys` class is an abstract class which can be used to implements the rollout dynamics for the ILQR algorithm. We have implemented two instances of this class which can be found in [dynamic_system.py](al_ilqr/dynamic_system.py): `ILQRDynSys`, `ILQRDynSysClosedLoop`.

The `Controller` class is an abstract class which can be used to implement different controllers. To implement a new controller the following methods are to be implemented: `init_state`, `_get_err`, `_compute_input`, `_update_state`. Look at `PIController` in [test_constrained_ilqr.py](test_constrained_ilqr.py) for an example.

## 3. Loss
The `LossFunctionBase` class in [loss.py](al_ilqr/loss.py) is an abstract class which can be used to implement different loss functions. To implement a new loss function the steps are:
1. Implement a new state class which inherits from `BaseState`
2. Implement the `evaluate()` method which returns the loss value.
3. Implement the optional methods: `_dldx`, `_dldu`, `_dldxx`, `_dlduu`, `_dldux` which return the derivatives of the loss w.r.t. the state and the control input. If the derivatives are not implemented the numerical derivatives will be used.

## 4. Constraints
The `Constraint` class in [constraints.py](al_ilqr/constraints.py) is an abstract class which defines `InEqConstraint` and `EqConstraint` that can be used to implement different constraints. This class is not meant to be inherited from in the general case.

As shown in [test_constrained_ilqr.py](test_constrained_ilqr.py) the constraints can be implemented as follows:
``` python
my_constraints: list[Constraint] = []

def eq_c(x: BasicState, u, t):
    if t >= T // 3 and t < T // 2:
        return x.diff(x_des_traj[0]).view(-1)  # end state constraint
    else:
        return x.val.new_zeros(2)

my_constr = EqConstraint()
my_constr.build(N_out=2, c=eq_c)
my_constraints.append(my_constr)

def ineq_c(x: BasicState, u, t):
    return (x.val[1] - 15.0).view(-1)  # end state constraint

my_constr2 = InEqConstraint()
my_constr2.build(N_out=1, c=ineq_c)
my_constraints.append(my_constr2)
```